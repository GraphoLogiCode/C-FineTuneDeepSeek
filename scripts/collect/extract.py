"""Step 2 of 3: cut each scanned file into functions and attach the findings that fall inside each one.

    python scripts/collect/extract.py              # every engine that has been scanned
    python scripts/collect/extract.py godot ogre

Reads data/collect/<engine>/scan.jsonl, writes data/collect/<engine>/units.jsonl.

A unit is one function definition with a body. Smelly and clean code are cut
the same way, so the two classes cannot be told apart by how they were sliced.
A function is kept only if clang-tidy demonstrably analysed it and reported no
compile error inside it; everything else is dropped and counted, never
labelled clean by default.
"""
import argparse
import bisect
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import (COGNITIVE_CHECK, SIZE_CHECK, engine_root, load_config, read_jsonl, selected_engines, sha1,
                    utf8_stdout, walk_files, work_dir)

DEFINE_RE = re.compile(rb"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\()?", re.M)
CALL_AT = re.compile(rb"([A-Za-z_]\w*)[ \t]*\(")
# A macro used as a whole statement ("CHECK_GL_ERROR;"): the identifier alone on its line.
STATEMENT_AT = re.compile(rb"([A-Za-z_]\w*)[ \t]*;?[ \t]*(?://[^\n]*)?\r?$", re.M)
STANDARD_MACROS = {"assert", "va_start", "va_arg", "va_end", "va_copy", "offsetof"}
MACRO_EXTENSIONS = [".h", ".hpp", ".hh", ".hxx", ".inl", ".inc", ".cpp", ".cc", ".cxx", ".c"]

_parser = None
_function_macros = frozenset()
_object_macros = frozenset()


def init_worker(macros):
    global _function_macros, _object_macros
    _function_macros = frozenset(macros["function_like"])
    _object_macros = frozenset(macros["object_like"])


def engine_macros(cfg, engine):
    """Names of every macro defined anywhere in the engine tree, split by kind."""
    cache = work_dir(cfg, engine) / "macros.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    function_like, object_like = set(STANDARD_MACROS), set()
    for full, _ in walk_files(engine_root(cfg, engine), MACRO_EXTENSIONS):
        try:
            for name, paren in DEFINE_RE.findall(full.read_bytes()):
                (function_like if paren else object_like).add(name.decode("latin-1"))
        except OSError:
            pass
    macros = {"function_like": sorted(function_like), "object_like": sorted(object_like - function_like)}
    cache.write_text(json.dumps(macros), encoding="utf-8")
    return macros


def from_macro(raw, offset):
    """True if the finding at `offset` sits on a macro invocation rather than on written code."""
    call = CALL_AT.match(raw, offset)
    if call and call.group(1).decode("latin-1") in _function_macros:
        return True
    # An object-like macro counts only when it stands alone as a statement. One used as a value
    # (NULL, a constant) is something the developer chose to write.
    line_start = raw.rfind(b"\n", 0, offset) + 1
    if raw[line_start:offset].strip():
        return False
    statement = STATEMENT_AT.match(raw, offset)
    return bool(statement) and statement.group(1).decode("latin-1") in _object_macros


def parser():
    global _parser
    if _parser is None:
        import tree_sitter_cpp
        from tree_sitter import Language, Parser
        _parser = Parser(Language(tree_sitter_cpp.language()))
    return _parser


def function_nodes(root):
    """(outer node, definition node) for every function definition that has a body, outermost only."""
    found, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            body = node.child_by_field_name("body")
            if body is not None and body.type == "compound_statement":
                outer = node
                while outer.parent is not None and outer.parent.type == "template_declaration":
                    outer = outer.parent  # keep the template<...> header with the function
                found.append((outer, node))
            continue  # anything defined inside a function body belongs to that function
        stack.extend(node.children)
    return found


def function_name(node, raw):
    decl = node.child_by_field_name("declarator")
    while decl is not None and decl.type != "function_declarator":
        inner = decl.child_by_field_name("declarator")
        if inner is None:
            break
        decl = inner
    if decl is not None and decl.type == "function_declarator":
        decl = decl.child_by_field_name("declarator") or decl
    if decl is None:
        return ""
    return " ".join(raw[decl.start_byte:decl.end_byte].decode("utf-8", "replace").split())[:120]


def extract_file(task):
    """Units of one file plus a count of why functions were dropped."""
    rec, path, limits = task["record"], task["path"], task["limits"]
    dropped = Counter()
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], Counter({"file unreadable": 1})
    if sha1(raw) != rec["sha1"]:
        return [], Counter({"file changed since the scan": 1})

    diags = sorted(rec["diags"], key=lambda d: d["o"])
    offsets = [d["o"] for d in diags]
    error_offsets = sorted(e[0] for e in rec["errors_main"])
    newlines = [i for i, b in enumerate(raw) if b == 10]

    def line_of(offset):
        return bisect.bisect_left(newlines, offset) + 1

    units = []
    for outer, definition in function_nodes(parser().parse(raw).root_node):
        start, end = outer.start_byte, outer.end_byte
        inside = diags[bisect.bisect_left(offsets, start):bisect.bisect_left(offsets, end)]
        witness = next((d for d in inside if d["c"] == SIZE_CHECK), None)
        if witness is None:
            dropped["not analysed by clang-tidy (inactive #if branch, or skipped)"] += 1
            continue
        i = bisect.bisect_left(error_offsets, start)
        if i < len(error_offsets) and error_offsets[i] < end:
            dropped["compile error inside the function"] += 1
            continue
        # Indentation of the first line; also tells whether the function starts at the beginning of a line.
        line_start = raw.rfind(b"\n", 0, start) + 1
        lead = raw[line_start:start]
        # A macro in the signature (OVERRIDE, API export ...) confuses tree-sitter but not clang. Accept that
        # case when the body itself parsed cleanly; anything else may have the wrong boundaries.
        if outer.has_error and (definition.child_by_field_name("body").has_error or lead.strip()):
            dropped["tree-sitter could not parse the function cleanly"] += 1
            continue
        try:
            code = raw[start:end].decode("utf-8")
        except UnicodeDecodeError:
            dropped["not valid UTF-8"] += 1
            continue
        start_line, end_line = line_of(start), line_of(end - 1)
        n_lines = end_line - start_line + 1
        # Step 3 uses the indentation to shift the body of a class member back to the margin.
        indent = lead.decode("ascii") if lead.strip() == b"" else ""
        if n_lines < limits["min_lines"]:
            dropped[f"shorter than {limits['min_lines']} lines"] += 1
            continue
        if n_lines > limits["max_lines"]:
            dropped[f"longer than {limits['max_lines']} lines"] += 1
            continue

        cognitive = next((d["v"] for d in inside if d["c"] == COGNITIVE_CHECK), 0)
        findings = []
        for d in inside:
            if d["c"] in (SIZE_CHECK, COGNITIVE_CHECK):
                continue
            # A finding reported at a macro invocation is about the code the macro expands to
            # (an error-check macro's hidden if/else, say), not about what is written in this function.
            if from_macro(raw, d["o"]):
                dropped["findings ignored: reported at a macro invocation"] += 1
                continue
            f = {"check": d["c"], "line": line_of(d["o"]) - start_line + 1, "message": d["m"]}
            # A fix is usable only if every edit lies inside this function.
            if "r" in d and all(start <= o and o + n <= end for o, n, _ in d["r"]):
                f["fix"] = [[o - start, n, text] for o, n, text in d["r"]]
            findings.append(f)
        units.append({"path": rec["path"], "name": function_name(definition, raw), "start_line": start_line,
                      "end_line": end_line, "code": code, "indent": indent,
                      "metrics": {**witness["k"], "cognitive": cognitive},
                      "findings": findings, "file_errors_main": rec["n_errors_main"],
                      "file_errors_other": rec["n_errors_other"]})
    return units, dropped


def extract_engine(cfg, engine, jobs):
    work = work_dir(cfg, engine)
    scan_file = work / "scan.jsonl"
    if not scan_file.exists():
        print(f"{engine}: not scanned yet, run scan.py first")
        return
    meta_file = work / "scan_meta.json"
    cfg_hash = json.loads(meta_file.read_text(encoding="utf-8"))["cfg_hash"] if meta_file.exists() else None
    latest = {}
    for rec in read_jsonl(scan_file):
        if cfg_hash is None or rec.get("cfg") == cfg_hash:
            latest[rec["path"]] = rec
    root = engine_root(cfg, engine)
    statuses = Counter(r["status"] for r in latest.values())
    tasks = [{"record": r, "path": str(root / r["path"]), "limits": cfg["units"]}
             for r in latest.values() if r["status"] == "ok"]

    macros = engine_macros(cfg, engine)
    dropped, n_units, n_strict, n_smelly = Counter(), 0, 0, 0
    with open(work / "units.jsonl", "w", encoding="utf-8", newline="\n") as out, \
            ProcessPoolExecutor(max_workers=jobs, initializer=init_worker, initargs=(macros,)) as pool:
        for units, why in pool.map(extract_file, tasks, chunksize=8):
            dropped.update(why)
            for u in units:
                u["engine"] = engine
                out.write(json.dumps(u, ensure_ascii=False) + "\n")
                n_units += 1
                strict = not u["file_errors_main"] and not u["file_errors_other"]
                n_strict += strict
                n_smelly += bool(u["findings"])
    summary = {"engine": engine, "files_by_status": dict(statuses), "files_used": len(tasks), "units": n_units,
               "units_in_error_free_files": n_strict, "units_with_any_finding": n_smelly,
               "functions_dropped": dict(dropped.most_common())}
    (work / "units_meta.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"== {engine}: {n_units} functions kept from {len(tasks)} files "
          f"({n_strict} in files with no compile error); dropped: {dict(dropped.most_common())}", flush=True)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("engines", nargs="*")
    ap.add_argument("--config")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    jobs = args.jobs or max(1, (os.cpu_count() or 4) - 2)
    for engine in selected_engines(cfg, args.engines):
        extract_engine(cfg, engine, jobs)


if __name__ == "__main__":
    main()
