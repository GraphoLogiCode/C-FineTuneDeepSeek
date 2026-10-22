"""Step 1 of 3: run clang-tidy on every engine source file and record what it reports.

    python scripts/collect/scan.py                 # every engine in the config
    python scripts/collect/scan.py godot ogre      # some engines
    python scripts/collect/scan.py irrlicht --limit 40

Writes data/collect/<engine>/scan.jsonl, one record per file. The run can be
interrupted and restarted; files already scanned with the same settings are skipped.

Design points (each answers a bug in the old GUI tool, see docs/COLLECTION.md):
  * Every file is analysed as its own main file and only diagnostics located in
    that file are kept, so a warning can never be attached to the wrong file.
  * Diagnostics are read from clang-tidy's structured YAML export, not scraped
    from its console output.
  * Every file gets a status. A timeout, a crash or a compile error is recorded
    as such and never mistaken for "no warnings".
  * Function-size thresholds are set to zero, so clang-tidy emits one diagnostic
    per function it really analysed. Step 2 uses that as proof of analysis.
  * No build of the engine is needed. Include paths are derived from the
    engine's own #include lines, and headers that cannot be found are replaced
    by empty stubs so that the damage shows up as ordinary errors at the point
    of use instead of silently stopping the analysis.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from common import (COGNITIVE_CHECK, SIZE_CHECK, engine_root, is_excluded, load_config, read_jsonl,
                    selected_engines, sha1, source_files, utf8_stdout, walk_files, work_dir)

INCLUDE_RE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"\r\n]+)[>"]', re.M)
# "class MACRO Name :", "struct MACRO Name {": the macro in the middle is an export/visibility annotation.
EXPORT_MACRO_RE = re.compile(
    rb"^[ \t]*(?:class|struct)[ \t]+([A-Za-z_]\w*)[ \t]+(?!final\b|sealed\b|abstract\b)[A-Za-z_]\w*[ \t]*"
    rb"(?:final[ \t]*)?(?::(?!:)|\{|\r?$)", re.M)
CPP_WORDS = {"final", "sealed", "abstract", "alignas", "const", "volatile", "typename", "class", "struct", "enum"}
MISSING_RE = re.compile(r"'([^']+)' file not found")
UNKNOWN_TYPE_RE = re.compile(r"unknown type name '(\w+)'")
ANNOTATION_WORDS = ("DEPRECATED", "API", "EXPORT", "IMPORT", "DLL", "INLINE", "ALIGN", "NODISCARD", "NORETURN",
                    "HIDDEN", "VISIBILITY", "ATTRIBUTE", "ATTR", "DECL", "UNUSED", "RESTRICT", "PUBLIC", "PRIVATE",
                    "LOCAL", "WARN", "PURE", "LIKELY", "EXTERN", "CALL", "OVERRIDE", "FINAL", "NOEXCEPT", "CONSTEXPR",
                    "THREAD", "FORCE")
METRIC_RE = re.compile(r"^(\d+) (lines|statements|branches|parameters|variables)\b")
NESTING_RE = re.compile(r"^nesting level (\d+) starts here")
COGNITIVE_RE = re.compile(r"cognitive complexity of (\d+)")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# Stubs sit this many folders deep so that an include such as "../../x.h" still lands inside the stub tree.
STUB_NEST = "/".join(["_"] * 8)
INDEX_EXTENSIONS = [".h", ".hpp", ".hh", ".hxx", ".inl", ".inc", ".ipp", ".tpp", ".cpp", ".cc", ".cxx", ".c"]
SYSTEM_HEADERS = {h + ".h" for h in (
    "assert complex ctype errno fenv float inttypes iso646 limits locale math setjmp signal stdarg stdbool "
    "stddef stdint stdio stdlib string tgmath time uchar wchar wctype memory malloc io process direct fcntl "
    "conio search share new windows winsock2 unistd dirent pthread").split()} | set(
    "algorithm any array atomic bitset cassert cctype cerrno cfloat chrono cinttypes climits cmath codecvt "
    "complex condition_variable csetjmp csignal cstdarg cstddef cstdint cstdio cstdlib cstring ctime cwchar "
    "deque exception filesystem forward_list fstream functional future initializer_list iomanip ios iosfwd "
    "iostream istream iterator limits list locale map memory mutex new numeric optional ostream queue random "
    "ratio regex set sstream stack stdexcept streambuf string string_view system_error thread tuple "
    "type_traits typeindex typeinfo unordered_map unordered_set utility valarray variant vector".split())


def find_clang_tidy():
    beside_python = Path(sys.executable).parent / ("clang-tidy.exe" if os.name == "nt" else "clang-tidy")
    exe = str(beside_python) if beside_python.exists() else shutil.which("clang-tidy")
    if not exe:
        sys.exit("clang-tidy not found. Install the collection requirements: pip install -r requirements-collect.txt")
    return exe


def clang_tidy_version(exe):
    out = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"version (\S+)", out)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------- preparation

def create_stub(stub_base, include):
    """Create an empty file for `include` under the stub tree; returns False if it cannot be placed there."""
    stub_root = stub_base.parents[STUB_NEST.count("/")]
    target = Path(os.path.normpath(stub_base / include))
    if stub_root not in target.parents:
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
        return True
    except OSError:
        return False


def system_header_names():
    """Lower-case names of standard and C-runtime headers, which an engine folder must not shadow."""
    names = set(SYSTEM_HEADERS)
    if os.name == "nt":  # add what this machine's compiler actually ships, when it can be located
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        folders = []
        try:
            vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
            folders += sorted(Path(vs, "VC/Tools/MSVC").glob("*/include"))[-1:] if vs else []
        except (OSError, subprocess.SubprocessError):
            pass
        kits = Path(os.environ.get("ProgramFiles(x86)", ""), "Windows Kits/10/Include")
        folders += sorted(kits.glob("*/ucrt"))[-1:]
        for folder in folders:
            names.update(p.name.lower() for p in folder.iterdir() if p.is_file())
    return names


def prepare(cfg, engine, refresh=False):
    """Derive include directories from the engine's own #include lines and create the header stubs."""
    work = work_dir(cfg, engine)
    prep_file = work / "prep.json"
    if prep_file.exists() and not refresh:
        return json.loads(prep_file.read_text(encoding="utf-8"))

    root = engine_root(cfg, engine)
    # Vendored code is indexed too: its headers let engine code parse, even though it is never labelled.
    files = list(walk_files(root, INDEX_EXTENSIONS))
    known = {rel.lower() for _, rel in files}
    by_name, children = defaultdict(list), defaultdict(set)
    for _, rel in files:
        folder, _, name = rel.rpartition("/")
        by_name[name.lower()].append(rel)
        children[folder].add(name.lower())

    engine_votes, vendor_votes, includes, export_macros = Counter(), Counter(), set(), Counter()
    for full, rel in files:
        try:
            data = full.read_bytes()
        except OSError:
            continue
        here = rel.rsplit("/", 1)[0] if "/" in rel else ""
        # Engine code decides the include path. Folders that only vendored libraries ask for are added
        # behind those, so that a library engine code includes can still find its own dependencies.
        is_engine_code = not is_excluded(rel, cfg, engine)
        votes = engine_votes if is_engine_code else vendor_votes
        if is_engine_code:
            export_macros.update(m.decode("latin-1") for m in EXPORT_MACRO_RE.findall(data))
        for raw in set(INCLUDE_RE.findall(data)):
            inc = raw.decode("latin-1").strip().replace("\\", "/")
            if not inc or inc.startswith("/") or ":" in inc:
                continue
            includes.add(inc)
            if ".." in inc.split("/"):
                continue
            if (f"{here}/{inc}" if here else inc).lower() in known:
                continue  # found next to the including file; no include directory needed
            suffix = "/" + inc.lower()
            candidates = [c for c in by_name.get(inc.rsplit("/", 1)[-1].lower(), ()) if ("/" + c.lower()).endswith(suffix)]
            # A bare file name that exists in several places says nothing about which directory is meant.
            if not candidates or ("/" not in inc and len(candidates) > 1):
                continue
            for c in candidates:
                votes[c[:len(c) - len(inc)].rstrip("/")] += 1.0 / len(candidates)

    system = system_header_names()
    limit = cfg["scan"]["max_include_dirs"]
    ranked = [""] + [d for d, _ in engine_votes.most_common(limit) if d]
    ranked += [d for d, _ in vendor_votes.most_common(limit) if d and d not in engine_votes][:limit // 2]
    # A folder holding a file named like a system header (math.h, signal.h ...) is searched after the
    # system folders, so the real header wins.
    after_dirs = [d for d in ranked if children[d] & system]
    include_dirs = [d for d in ranked if d not in after_dirs]
    stub_base = work / "stubs" / STUB_NEST
    stub_base.mkdir(parents=True, exist_ok=True)
    n_stubs = sum(create_stub(stub_base, inc) for inc in includes)

    # "class ENGINE_API Foo": such macros usually come from a header the build generates. Defined as
    # nothing here; if the engine does define one itself, its own definition simply replaces this one.
    auto_defines = sorted(m for m, n in export_macros.items() if n >= 3 and m not in CPP_WORDS)

    prep = {"include_dirs": include_dirs, "after_dirs": after_dirs, "auto_defines": auto_defines,
            "stub_base": stub_base.as_posix(), "n_stubs": n_stubs, "n_indexed_files": len(files),
            "n_distinct_includes": len(includes)}
    prep_file.write_text(json.dumps(prep, indent=1), encoding="utf-8")
    return prep


def write_tool_config(cfg, engine):
    """The clang-tidy configuration, written to a file so that a project's own .clang-tidy is ignored."""
    nesting = cfg["labels"]["thresholds"]["nesting"]
    options = {f"{SIZE_CHECK}.{k}": "0" for k in
               ("LineThreshold", "StatementThreshold", "BranchThreshold", "ParameterThreshold", "VariableThreshold")}
    options[f"{SIZE_CHECK}.NestingThreshold"] = str(nesting)
    options[f"{COGNITIVE_CHECK}.Threshold"] = "0"
    # Code that comes out of a macro (assert and error-check macros expand to if/else and do/while)
    # is not what the developer wrote; do not count it where the check allows that.
    options[f"{COGNITIVE_CHECK}.IgnoreMacros"] = "true"
    options["cppcoreguidelines-avoid-do-while.IgnoreMacros"] = "true"
    options["readability-simplify-boolean-expr.IgnoreMacros"] = "true"
    data = {"Checks": ",".join(cfg["scan"]["checks"] + [SIZE_CHECK, COGNITIVE_CHECK]), "WarningsAsErrors": "",
            "HeaderFilterRegex": "", "FormatStyle": "none", "CheckOptions": options}
    path = work_dir(cfg, engine) / "clang-tidy.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path, data


def write_flags(cfg, engine, prep, std, annotation_macros=()):
    """A compile_flags.txt for one language standard; clang-tidy reads it through -p."""
    root = engine_root(cfg, engine).as_posix()
    spec = cfg["engines"][engine]
    args = ["-xc++", f"-std={std}"] + cfg["scan"]["extra_args"]
    if os.name == "nt":
        args += ["-DWIN32", "-D_WINDOWS"]  # what CMake and Visual Studio projects define by default on Windows
    args += [f"-D{m}=" for m in list(prep.get("auto_defines", [])) + list(annotation_macros)]
    args += [f"-D{d}" for d in spec.get("defines", [])]
    for header in spec.get("force_include", []):  # a precompiled header the build includes in every file
        args += ["-include", f"{root}/{header}"]
    args += [f"-I{root}/{d}" if d else f"-I{root}" for d in prep["include_dirs"]]
    args += [f"-I{root}/{d}" for d in spec.get("include_dirs", [])]
    for d in prep.get("after_dirs", []):
        args += ["-idirafter", f"{root}/{d}" if d else root]
    args += ["-idirafter", prep["stub_base"]]
    flags_dir = work_dir(cfg, engine) / "flags" / std
    flags_dir.mkdir(parents=True, exist_ok=True)
    (flags_dir / "compile_flags.txt").write_text("\n".join(args) + "\n", encoding="utf-8")
    return flags_dir, args


# ---------------------------------------------------------------- one file

def parse_export(text):
    try:
        return yaml.load(text, Loader=YAML_LOADER)
    except yaml.YAMLError:
        # clang-tidy copies raw source bytes into messages and replacements; drop what YAML refuses.
        return yaml.load(CONTROL_CHARS.sub(" ", text), Loader=YAML_LOADER)


def same_file(a, b_norm):
    return bool(a) and os.path.normcase(os.path.normpath(a)) == b_norm


def to_record(diagnostics, main_norm):
    """Reduce clang-tidy's export to what steps 2 and 3 need. Only main-file diagnostics are kept."""
    diags, errors_main, missing, other_samples = [], [], [], []
    n_errors_other = 0
    unknown = Counter()
    for d in diagnostics:
        name = d.get("DiagnosticName") or ""
        msg = d.get("DiagnosticMessage") or {}
        text = (msg.get("Message") or "").strip()
        offset = msg.get("FileOffset")
        in_main = same_file(msg.get("FilePath"), main_norm)
        if d.get("Level") == "Error":
            m = MISSING_RE.search(text)
            if m:
                missing.append(m.group(1))
            m = UNKNOWN_TYPE_RE.search(text)
            if m:
                unknown[m.group(1)] += 1
            if in_main and offset is not None:
                errors_main.append([offset, text[:200]])
            else:
                n_errors_other += 1
                if len(other_samples) < 3:
                    other_samples.append([os.path.basename(msg.get("FilePath") or "?"), text[:120]])
            continue
        if not in_main or offset is None or name.startswith("clang-diagnostic"):
            continue
        notes = [(n.get("Message") or "") for n in (d.get("Notes") or [])]
        if name == SIZE_CHECK:
            metrics = {}
            for note in notes:
                m = METRIC_RE.match(note)
                if m:
                    metrics[m.group(2)] = int(m.group(1))
                elif NESTING_RE.match(note):
                    metrics["nesting"] = max(metrics.get("nesting", 0), int(NESTING_RE.match(note).group(1)))
            diags.append({"c": name, "o": offset, "k": metrics})
        elif name == COGNITIVE_CHECK:
            m = COGNITIVE_RE.search(text)
            diags.append({"c": name, "o": offset, "v": int(m.group(1)) if m else 0})
        else:
            entry = {"c": name, "o": offset, "m": text[:300]}
            reps = msg.get("Replacements") or []
            if reps:
                if all(same_file(r.get("FilePath"), main_norm) for r in reps):
                    entry["r"] = [[r["Offset"], r["Length"], r.get("ReplacementText") or ""] for r in reps]
                else:
                    entry["rx"] = 1  # the fix also edits another file, so it cannot be applied to this function alone
            diags.append(entry)
    return {"diags": diags, "errors_main": errors_main[:300], "n_errors_main": len(errors_main),
            "n_errors_other": n_errors_other, "other_error_samples": other_samples, "missing": missing,
            "unknown": dict(unknown.most_common(15))}


def scan_one(task):
    """Run clang-tidy on one file. Returns a record that always carries a status."""
    path, rel = task["path"], task["rel"]
    started = time.time()
    record = {"path": rel, "cfg": task["cfg_hash"]}
    try:
        record["sha1"] = sha1(Path(path).read_bytes())
    except OSError as e:
        return {**record, "status": "unreadable", "detail": str(e)[:200]}
    main_norm = os.path.normcase(os.path.normpath(path))
    stub_base = Path(task["stub_base"])
    result, attempts = None, 0
    for attempts in range(1, 5):
        fd, export = tempfile.mkstemp(suffix=".yaml", dir=task["tmp_dir"])
        os.close(fd)
        os.unlink(export)
        cmd = [task["clang_tidy"], path, f"--config-file={task['tool_config']}", f"--export-fixes={export}",
               "--quiet", "-p", task["flags_dir"]]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=task["timeout"])
        except subprocess.TimeoutExpired:
            if os.path.exists(export):
                os.unlink(export)
            return {**record, "status": "timeout", "sec": round(time.time() - started, 1)}
        if not os.path.exists(export):
            if proc.returncode == 0:
                result = to_record([], main_norm)  # nothing at all was reported
                break
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            return {**record, "status": "failed", "detail": " | ".join(tail)[:300],
                    "sec": round(time.time() - started, 1)}
        try:
            with open(export, encoding="utf-8", errors="replace") as f:
                parsed = parse_export(f.read())
        except yaml.YAMLError as e:
            return {**record, "status": "unparsable-output", "detail": str(e)[:200],
                    "sec": round(time.time() - started, 1)}
        finally:
            os.unlink(export)
        result = to_record((parsed or {}).get("Diagnostics") or [], main_norm)
        # A missing header is a fatal error: clang stops reporting after it. Stub it and analyse again.
        if not result["missing"] or not any([create_stub(stub_base, m) for m in result["missing"]]):
            break
    status = "missing-include" if result["missing"] else "ok"
    return {**record, "status": status, "sec": round(time.time() - started, 1), "attempts": attempts, **result}


# ---------------------------------------------------------------- one engine

def run_batch(tasks, jobs, on_result=None, label=""):
    results, started = [], time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(scan_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            results.append(rec)
            if on_result:
                on_result(rec)
            if label and (i % 200 == 0 or i == len(tasks)):
                rate = i / max(time.time() - started, 1e-6)
                print(f"  {label}: {i}/{len(tasks)} files, {rate:.1f} files/s, "
                      f"about {max(len(tasks) - i, 0) / rate / 60:.0f} min left", flush=True)
    return results


def usable_functions(records):
    """Functions clang-tidy analysed in files that compiled without any error: the yield that matters."""
    return sum(sum(d["c"] == SIZE_CHECK for d in r["diags"]) for r in records
               if r["status"] == "ok" and not r["n_errors_main"] and not r["n_errors_other"])


def looks_like_annotation(name):
    """DEPRECATED_API-style macro names: safe to define as nothing when the build would have supplied them."""
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", name)) and any(w in name for w in ANNOTATION_WORDS)


def calibrate(cfg, engine, prep, base_task, files, jobs):
    """On a sample of files, pick the language standard and find annotation macros the build would define.

    Returns (standard, macros, report). The choice is cached; --refresh-prep discards it.
    """
    spec = cfg["engines"][engine]
    cal_file = work_dir(cfg, engine) / "calibration.json"
    if cal_file.exists():
        saved = json.loads(cal_file.read_text(encoding="utf-8"))
        return saved["std"], saved.get("annotation_macros", []), None
    impl = [f for f in files if not f[1].lower().endswith((".h", ".hpp", ".hh", ".hxx"))] or files
    sample = random.Random(0).sample(impl, min(cfg["scan"]["calibration_files"], len(impl)))

    def trial(std, macros):
        flags_dir, _ = write_flags(cfg, engine, prep, std, macros)
        tasks = [{**base_task, "path": str(p), "rel": rel, "flags_dir": str(flags_dir)} for p, rel in sample]
        records = run_batch(tasks, jobs)
        return usable_functions(records), records

    candidates = [spec["std"]] if spec.get("std") else cfg["scan"]["std_candidates"]
    trials = {std: trial(std, []) for std in candidates}
    scores = {std: t[0] for std, t in trials.items()}
    best = max(candidates, key=lambda s: scores[s])
    score, records = trials[best]

    # An annotation macro (OGRE_DEPRECATED ...) that the build system would define shows up as an
    # "unknown type name" in many files. Define it as nothing and keep it only if the yield rises.
    macros = []
    for _ in range(3):
        seen = Counter(n for r in records if r["status"] == "ok" for n in r.get("unknown", {}))
        new = sorted(n for n, k in seen.items()
                     if k >= max(2, 0.15 * len(sample)) and looks_like_annotation(n) and n not in macros)
        if not new:
            break
        new_score, new_records = trial(best, macros + new)
        if new_score <= score:
            break  # no evidence that it helps; leave the code as the engine wrote it
        macros, score, records = macros + new, new_score, new_records
    report = {"std": best, "annotation_macros": macros, "usable_functions_per_standard": scores,
              "usable_functions_final": score, "sample_files": len(sample)}
    cal_file.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return best, macros, report


def scan_engine(cfg, engine, args, clang_tidy, version):
    work = work_dir(cfg, engine)
    files = source_files(cfg, engine)
    if not files:
        print(f"{engine}: no source files found under {engine_root(cfg, engine)}")
        return
    print(f"\n== {engine}: {len(files)} files to analyse", flush=True)
    if args.refresh_prep:
        (work / "calibration.json").unlink(missing_ok=True)  # the standard was chosen under the old include path
    prep = prepare(cfg, engine, args.refresh_prep)
    print(f"  include directories: {len(prep['include_dirs'])} (+{len(prep['after_dirs'])} searched after the "
          f"system headers); header stubs: {prep['n_stubs']}; export macros defined empty: "
          f"{', '.join(prep['auto_defines'][:8]) or 'none'}{' ...' if len(prep['auto_defines']) > 8 else ''}", flush=True)
    tool_config, tool_data = write_tool_config(cfg, engine)
    tmp_dir = work / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    jobs = args.jobs or cfg["scan"]["jobs"] or max(1, (os.cpu_count() or 4) - 2)
    base_task = {"clang_tidy": clang_tidy, "tool_config": str(tool_config), "stub_base": prep["stub_base"],
                 "tmp_dir": str(tmp_dir), "timeout": cfg["scan"]["timeout_seconds"], "cfg_hash": ""}

    std, macros, report = calibrate(cfg, engine, prep, base_task, files, jobs)
    if report:
        print(f"  calibrated on {report['sample_files']} files: standard {std} "
              f"(usable functions per standard: {report['usable_functions_per_standard']}); "
              f"annotation macros defined empty: {', '.join(macros) or 'none'} "
              f"-> {report['usable_functions_final']} usable functions", flush=True)
    flags_dir, flag_args = write_flags(cfg, engine, prep, std, macros)
    cfg_hash = sha1(json.dumps([version, tool_data, flag_args], sort_keys=True).encode())[:12]
    base_task.update(flags_dir=str(flags_dir), cfg_hash=cfg_hash)

    if args.limit and args.limit < len(files):
        step = len(files) / args.limit
        files = [files[int(i * step)] for i in range(args.limit)]

    scan_file = work / "scan.jsonl"
    done = {}
    if scan_file.exists() and not args.force:
        for rec in read_jsonl(scan_file):
            if rec.get("cfg") == cfg_hash:
                done[rec["path"]] = rec
    todo = []
    for path, rel in files:
        prev = done.get(rel)
        if prev is None or prev.get("sha1") != sha1(path.read_bytes()):
            todo.append({**base_task, "path": str(path), "rel": rel})
    print(f"  {len(files) - len(todo)} already scanned with these settings, {len(todo)} to do, {jobs} parallel jobs",
          flush=True)

    if args.force and scan_file.exists():
        scan_file.unlink()
    if todo:
        with open(scan_file, "a", encoding="utf-8", newline="\n") as out:
            def save(rec):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[rec["path"]] = rec
            run_batch(todo, jobs, save, label=engine)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    wanted = {rel for _, rel in files}
    records = [r for rel, r in done.items() if rel in wanted]
    status = Counter(r["status"] for r in records)
    ok = [r for r in records if r["status"] == "ok"]
    clean_parse = [r for r in ok if not r["n_errors_main"] and not r["n_errors_other"]]
    checks = Counter(d["c"] for r in ok for d in r["diags"] if d["c"] not in (SIZE_CHECK, COGNITIVE_CHECK))
    meta = {"engine": engine, "clang_tidy": version, "std": std, "cfg_hash": cfg_hash, "files": len(records),
            "status": dict(status), "files_without_any_error": len(clean_parse),
            "functions_analysed": sum(sum(d["c"] == SIZE_CHECK for d in r["diags"]) for r in ok),
            "functions_in_error_free_files": usable_functions(ok),
            "warnings": sum(checks.values()), "top_checks": dict(checks.most_common(25))}
    (work / "scan_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  status: {dict(status)}")
    print(f"  files with no compile error at all: {len(clean_parse)} of {len(records)}")
    print(f"  functions analysed: {meta['functions_analysed']}; in error-free files: {meta['functions_in_error_free_files']}")
    print("  most frequent checks: " + ", ".join(f"{c} {n}" for c, n in checks.most_common(8)), flush=True)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="*", help="engine folder names; default: all configured engines found on disk")
    ap.add_argument("--config", help="path to a config file (default: collect_config.json next to this script)")
    ap.add_argument("--limit", type=int, help="analyse only this many files per engine, evenly spread (for trials)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel clang-tidy processes (default: CPU count minus 2)")
    ap.add_argument("--force", action="store_true", help="discard earlier results and scan again")
    ap.add_argument("--refresh-prep", action="store_true", help="rebuild include directories and header stubs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    clang_tidy = find_clang_tidy()
    version = clang_tidy_version(clang_tidy)
    print(f"clang-tidy {version} at {clang_tidy}")
    for engine in selected_engines(cfg, args.engines):
        scan_engine(cfg, engine, args, clang_tidy, version)


if __name__ == "__main__":
    main()
