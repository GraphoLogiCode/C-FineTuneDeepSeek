"""Conservative, reproducible static validation of a generated C++ function repair.

The candidate is overlaid onto the *original absolute source path* with Clang's
virtual filesystem. Includes, declarations and compile flags remain those used
by collection; engine sources are never modified. A pass means successful static
analysis and removal of the requested diagnostics, NOT behavioral equivalence,
an engine build, or a successful test suite. Parser-only results are inconclusive.

    python -m src.evaluation.repair_validation --row-id fix-... --candidate fix.cpp
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "scripts/collect/collect_config.json"
WITNESS = "readability-function-size"
METRICS = {WITNESS, "readability-function-cognitive-complexity"}


class ValidationUnavailable(ValueError):
    """Required provenance, source context or compiler tooling is unavailable."""


def original_code(row: dict) -> str:
    """Read only the user input; never derive it from the assistant reference."""
    if isinstance(row.get("code"), str):
        return row["code"]
    users = [m["content"] for m in row.get("messages", []) if m.get("role") == "user"]
    if len(users) != 1 or "\n\n" not in users[0]:
        raise ValidationUnavailable("Cannot extract the original function from this row")
    return users[0].split("\n\n", 1)[1]


def _normal(code: str, indent: str = "") -> str:
    lines = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if indent:
        lines = lines[:1] + [line[len(indent):] if line.startswith(indent) else line for line in lines[1:]]
    return "\n".join(lines).strip()


def _is_defaulted_repair(before: str, candidate: str) -> bool:
    """An exact signature followed by '= default;' is a definition without a body.

    Clang validates whether that signature is a defaultable special member. No
    function-size witness is expected for the intentionally absent body.
    """
    match = re.fullmatch(r"\s*(.+?)\s*=\s*default\s*;\s*", candidate, re.S)
    if match is None or "{" not in before:
        return False
    signature = before.partition("{")[0]
    return re.sub(r"\s+", "", signature) == re.sub(r"\s+", "", match.group(1))


def _functions(raw: bytes):
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_cpp
    except ImportError as exc:
        raise ValidationUnavailable("tree-sitter C++ is unavailable; cannot verify function boundaries") from exc
    tree = Parser(Language(tree_sitter_cpp.language())).parse(raw)
    stack, found = [tree.root_node], []
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            outer = node
            while outer.parent is not None and outer.parent.type == "template_declaration":
                outer = outer.parent
            decl = node.child_by_field_name("declarator")
            while decl is not None and decl.type != "function_declarator":
                decl = decl.child_by_field_name("declarator")
            name_node = decl.child_by_field_name("declarator") if decl else None
            name = " ".join(raw[name_node.start_byte:name_node.end_byte].decode("utf-8").split()) if name_node else ""
            found.append((outer, name))
            continue
        stack.extend(reversed(node.children))
    return found


def splice_function(raw: bytes, row: dict, before: str, candidate: str):
    """Locate the exact recorded function and return (new_source, start, end, new_end).

    Offsets are bytes, not characters. Leading indentation and CRLF of the source
    are preserved. Line metadata alone is never trusted to overwrite source text.
    """
    matches = []
    for node, name in _functions(raw):
        start, end = node.start_byte, node.end_byte
        if raw[:start].count(b"\n") + 1 != row.get("start_line"):
            continue
        if raw[:end - 1].count(b"\n") + 1 != row.get("end_line"):
            continue
        lead = raw[raw.rfind(b"\n", 0, start) + 1:start]
        indent = lead.decode("ascii") if not lead.strip() else ""
        if _normal(raw[start:end].decode("utf-8"), indent) == _normal(before):
            matches.append((start, end, indent, name))
    if len(matches) != 1:
        raise ValidationUnavailable("Original function no longer matches the recorded lines and input text")
    start, end, indent, name = matches[0]
    candidate = candidate.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not _is_defaulted_repair(before, candidate):
        functions = _functions(candidate.encode("utf-8"))
        if len(functions) != 1 or functions[0][1] != name:
            raise ValueError("Candidate must contain exactly the same named function")
        node = functions[0][0]
        # Do not allow extra declarations or preprocessor directives outside the function.
        prefix = candidate.encode()[:node.start_byte].strip()
        suffix = candidate.encode()[node.end_byte:].strip()
        if prefix or suffix:
            raise ValueError("Candidate contains text outside its function definition")
    for marker in ("NOLINT", "#pragma", "_Pragma", "__pragma", "#if", "#define", "#undef"):
        if candidate.count(marker) > before.count(marker):
            raise ValueError("Candidate introduces diagnostic suppression or conditional compilation")
    lines = candidate.split("\n")
    restored = "\n".join(lines[:1] + [indent + line if line else line for line in lines[1:]])
    if b"\r\n" in raw:
        restored = restored.replace("\n", "\r\n")
    replacement = restored.encode("utf-8")
    return raw[:start] + replacement + raw[end:], start, end, start + len(replacement)


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


class RepairValidator:
    """Validate many rows with cached baseline analyses and scan provenance.

    Cached analyses are scoped to this object. Each call rechecks the main source
    bytes and git HEAD. The caller must not change engine headers while a run is
    in progress. Collection uses inferred flags and may use empty header stubs;
    this is recorded explicitly, and is not an engine build configuration.
    """

    def __init__(self, config_path: str | Path | None = None, clang_tidy: str | Path | None = None,
                 timeout: float = 120, check_all: bool = True):
        self.config_path = Path(config_path or DEFAULT_CONFIG).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.timeout = timeout
        self.check_all = check_all
        self.clang_tidy = str(clang_tidy) if clang_tidy else self._find_tool()
        self._scans, self._analyses = {}, {}
        self._cache_lock, self._analysis_locks = threading.Lock(), {}
        self._version = None

    @staticmethod
    def _find_tool():
        for candidate in (Path(sys.executable).parent / "clang-tidy.exe", ROOT / ".venv/Scripts/clang-tidy.exe"):
            if candidate.is_file():
                return str(candidate)
        return shutil.which("clang-tidy")

    def _context(self, row):
        engine = row.get("engine")
        if engine not in self.config["engines"]:
            raise ValidationUnavailable("Unknown engine in row")
        engine_root = (Path(self.config["engines_root"]) / engine).resolve()
        path = (engine_root / row["path"]).resolve()
        if not path.is_relative_to(engine_root):
            raise ValidationUnavailable("Source path escapes its engine root")
        raw = path.read_bytes()
        work = ROOT / self.config["work_dir"] / engine
        meta = json.loads((work / "scan_meta.json").read_text(encoding="utf-8"))
        if engine not in self._scans:
            records = {}
            with (work / "scan.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("cfg") == meta["cfg_hash"]:
                        records[record["path"]] = record
            self._scans[engine] = records
        record = self._scans[engine].get(row["path"])
        digest = hashlib.sha1(raw).hexdigest()
        if not record or record.get("sha1") != digest:
            raise ValidationUnavailable("Stale or missing scan provenance: source SHA1 differs from collection")
        if record.get("status") != "ok":
            raise ValidationUnavailable("Original collection analysis did not finish successfully")
        stats_path = ROOT / self.config["dataset_dir"] / "stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        expected = row.get("engine_commit") or stats.get("engines", {}).get(engine, {}).get("commit")
        git = subprocess.run(["git", "--git-dir=" + str(engine_root / ".git"), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=min(self.timeout, 20))
        commit = git.stdout.strip() if git.returncode == 0 else None
        if expected and commit != expected:
            raise ValidationUnavailable("Stale engine provenance: current git HEAD differs from dataset commit")
        flags_dir = work / "flags" / meta["std"]
        flags = (flags_dir / "compile_flags.txt").read_text(encoding="utf-8")
        tool_config = work / "clang-tidy.yaml"
        config_bytes = tool_config.read_bytes()
        if not self.clang_tidy:
            raise ValidationUnavailable("clang-tidy is unavailable; parser-only validation is inconclusive")
        if self._version is None:
            version = subprocess.run([self.clang_tidy, "--version"], capture_output=True, text=True,
                                     timeout=min(self.timeout, 20))
            match = re.search(r"version (\S+)", version.stdout)
            self._version = match.group(1) if match else "unknown"
        if self._version != meta.get("clang_tidy"):
            raise ValidationUnavailable("clang-tidy version differs from collection; regenerate scan or use the recorded version")
        # Collection hashes exactly this [version, tool configuration, flag list].
        import yaml
        tool_data = yaml.safe_load(config_bytes)
        cfg_hash = hashlib.sha1(json.dumps([self._version, tool_data, flags.splitlines()],
                                          sort_keys=True).encode()).hexdigest()[:12]
        if cfg_hash != meta["cfg_hash"]:
            raise ValidationUnavailable("Cached compile flags or clang-tidy configuration differ from scan provenance")
        provenance = {"engine": engine, "path": row["path"], "source_path": str(path), "source_sha1": digest,
                      "engine_commit": commit, "expected_engine_commit": expected,
                      "engine_commit_verified": bool(expected and commit == expected),
                      "scan_config_hash": cfg_hash, "clang_tidy_version": self._version,
                      "compile_flags_sha256": hashlib.sha256(flags.encode()).hexdigest(),
                      "compile_flags_file": str(flags_dir / "compile_flags.txt"),
                      "configuration": "collection inferred flags; may include empty header stubs",
                      "source_modified": False}
        return path, raw, flags_dir, tool_config, provenance

    def _analyse(self, path, flags_dir, tool_config, replacement, targets):
        import yaml
        temp_root = ROOT / "data/repair_validation_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="repair-validation-", dir=temp_root) as temp:
            folder = Path(temp)
            export = folder / "diagnostics.yaml"
            cmd = [self.clang_tidy, str(path), f"--config-file={tool_config}", f"--export-fixes={export}",
                   "--quiet", "-p", str(flags_dir)]
            # Always force the collection witness on; new suppression directives are rejected above.
            checks = ("," + WITNESS) if self.check_all else "-*," + ",".join(sorted(set(targets) | {WITNESS}))
            cmd += [f"--checks={checks}", "--allow-no-checks=false"]
            if replacement is not None:
                candidate_path = folder / "candidate.cpp"
                candidate_path.write_bytes(replacement)
                overlay = folder / "overlay.json"
                overlay.write_text(json.dumps({"version": 0, "use-external-names": False, "roots": [
                    {"type": "file", "name": path.as_posix(), "external-contents": candidate_path.as_posix()}
                ]}), encoding="utf-8")
                cmd.append(f"--vfsoverlay={overlay}")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=self.timeout, cwd=ROOT)
            except subprocess.TimeoutExpired:
                return {"ok": False, "reason": "clang-tidy timed out", "errors": [], "diagnostics": []}
            try:
                parsed = yaml.safe_load(export.read_text(encoding="utf-8")) if export.exists() else None
            except (yaml.YAMLError, UnicodeError) as exc:
                return {"ok": False, "reason": "Invalid clang-tidy diagnostic export: " + str(exc)[:200],
                        "errors": [], "diagnostics": []}
            if not isinstance(parsed, dict) or not isinstance(parsed.get("Diagnostics"), list):
                return {"ok": False, "reason": "Missing structured clang-tidy diagnostic export",
                        "errors": [], "diagnostics": [], "returncode": proc.returncode}
            diagnostics, errors = [], []
            for diag in parsed["Diagnostics"]:
                msg = diag.get("DiagnosticMessage") or {}
                item = {"check": diag.get("DiagnosticName", ""), "offset": msg.get("FileOffset"),
                        "path": msg.get("FilePath", ""), "message": msg.get("Message", "")}
                if diag.get("Level") in {"Error", "Fatal"}:
                    errors.append(item)
                elif _path_key(item["path"]) == _path_key(path) and item["offset"] is not None:
                    diagnostics.append(item)
            return {"ok": proc.returncode == 0 and not errors, "returncode": proc.returncode,
                    "reason": "" if proc.returncode == 0 and not errors else "clang-tidy reported compiler/tool errors",
                    "errors": errors, "diagnostics": diagnostics,
                    "stderr_tail": proc.stderr[-1500:] if proc.returncode else ""}

    def _cached_analysis(self, key, path, flags_dir, config, replacement, targets):
        with self._cache_lock:
            lock = self._analysis_locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._analyses:
                self._analyses[key] = self._analyse(path, flags_dir, config, replacement, targets)
            return self._analyses[key]

    def validate(self, row: dict, candidate_code: str) -> dict[str, Any]:
        result = {"id": row.get("id"), "status": "inconclusive", "static_validated": False,
                  "behavioral_correctness": "not_tested", "engine_build": "not_tested",
                  "validation_scope": "full original translation unit under collection flags",
                  "reasons": [], "provenance": {}}
        try:
            targets = sorted(set(row.get("fixed_checks", [])))
            if not targets:
                raise ValidationUnavailable("No requested clang-tidy checks supplied")
            if not isinstance(candidate_code, str) or not candidate_code.strip():
                result.update(status="rejected", reasons=["Empty repair candidate"])
                return result
            path, raw, flags_dir, config, provenance = self._context(row)
            result["provenance"] = {**provenance,
                                    "candidate_sha256": hashlib.sha256(candidate_code.encode()).hexdigest()}
            before_code = original_code(row)
            try:
                patched, start, end, new_end = splice_function(raw, row, before_code, candidate_code)
            except ValidationUnavailable:
                raise
            except ValueError as exc:
                result.update(status="rejected", reasons=[str(exc)])
                return result
            key = (str(path), provenance["source_sha1"], provenance["scan_config_hash"], self.check_all,
                   tuple(targets) if not self.check_all else ())
            before = self._cached_analysis(key + ("original",), path, flags_dir, config, None, targets)
            result["before"] = before
            if not before["ok"]:
                result["reasons"] = ["Original full-source analysis failed: " + before["reason"]]
                return result
            inside = lambda d, stop: start <= d["offset"] < stop
            witnessed = any(d["check"] == WITNESS and inside(d, end) for d in before["diagnostics"])
            present = {d["check"] for d in before["diagnostics"] if inside(d, end)}
            absent = set(targets) - present
            if not witnessed or absent:
                result["reasons"] = ["Original function lacks analysis witness or requested diagnostics: " + ", ".join(sorted(absent))]
                return result
            after = before if patched == raw else self._cached_analysis(
                key + (hashlib.sha256(patched).hexdigest(),), path, flags_dir, config, patched, targets)
            result["after"] = after
            if not after["ok"]:
                result.update(status="rejected" if after["errors"] else "inconclusive",
                              reasons=["Candidate full-source analysis failed: " + after["reason"]])
                return result
            defaulted = _is_defaulted_repair(before_code, candidate_code)
            if not defaulted and not any(d["check"] == WITNESS and inside(d, new_end) for d in after["diagnostics"]):
                result["reasons"] = ["Candidate function lacks analysis witness; warning disappearance is inconclusive"]
                return result
            remaining = sorted({d["check"] for d in after["diagnostics"]
                                if inside(d, new_end) and d["check"] in targets})
            delta = new_end - end

            def signatures(diags, stop, shift):
                counts = Counter()
                for d in diags:
                    if d["check"] in METRICS or d["check"].startswith("clang-diagnostic-"):
                        continue
                    # Inside a rewritten function offsets legitimately move; outside they must track.
                    location = "inside" if inside(d, stop) else d["offset"] - (shift if d["offset"] >= stop else 0)
                    counts[(d["check"], d["message"], location)] += 1
                return counts

            old = signatures(before["diagnostics"], end, 0)
            new = signatures(after["diagnostics"], new_end, delta)
            added = new - old
            new_diagnostics = [{"check": c, "message": m, "location": o, "count": n}
                               for (c, m, o), n in added.items()]
            result.update(requested_checks=targets, remaining_checks=remaining, new_diagnostics=new_diagnostics,
                          changed=_normal(before_code) != _normal(candidate_code),
                          candidate_analysis_witness="compiled defaulted definition" if defaulted else WITNESS,
                          compiler_errors_before=len(before["errors"]), compiler_errors_after=len(after["errors"]))
            reasons = []
            if remaining:
                reasons.append("Requested diagnostics remain: " + ", ".join(remaining))
            if new_diagnostics:
                reasons.append("Candidate introduces new enabled main-file clang-tidy diagnostics")
            if reasons:
                result.update(status="rejected", reasons=reasons)
            else:
                result.update(status="passed_static_validation", static_validated=True,
                              reasons=["Requested warnings removed with no new compiler errors or enabled main-file warnings"])
            # Guard against accidental mutation or concurrent changes, even though VFS is read-only.
            if path.read_bytes() != raw:
                result.update(status="inconclusive", static_validated=False,
                              reasons=["Source file changed during validation"])
            return result
        except (ValueError, OSError, KeyError, UnicodeError, subprocess.SubprocessError) as exc:
            result["reasons"] = [str(exc)]
            return result


def validate_repair(row: dict, candidate_code: str, *, config_path=None, clang_tidy=None,
                    timeout=120, check_all=True) -> dict:
    """Convenience API; reuse RepairValidator for baseline caching across candidates."""
    return RepairValidator(config_path, clang_tidy, timeout, check_all).validate(row, candidate_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/collected/fixing/test.jsonl")
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--clang-tidy", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8") as handle:
        row = next((r for line in handle if (r := json.loads(line)).get("id") == args.row_id), None)
    if row is None:
        parser.error("row ID not found")
    result = validate_repair(row, args.candidate.read_text(encoding="utf-8"), config_path=args.config,
                             clang_tidy=args.clang_tidy, timeout=args.timeout)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["static_validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
