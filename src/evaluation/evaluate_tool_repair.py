"""Run a fresh, read-only clang-tidy repair baseline on held-out functions.

Only original code, original source context and requested check names enter
generation. Exported edits are applied to in-memory function bytes. This module
never asks Clang to write fixes and never writes an engine source file.

    python -m src.evaluation.evaluate_tool_repair --workers 4
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import yaml

from src.evaluation.repair_validation import ROOT, RepairValidator, original_code, splice_function


def _same_file(a, b):
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def apply_function_replacements(raw, start, end, replacements):
    """Apply unique, nonoverlapping byte edits wholly inside one function.

    An insertion sharing an offset with any other distinct edit is ambiguous and
    rejected. Identical edits are deduplicated, as Clang's replacement set does.
    """
    unique = sorted(set((int(o), int(n), str(text)) for o, n, text in replacements))
    for offset, length, _ in unique:
        if length < 0 or offset < start or offset + length > end:
            raise ValueError("Edit lies outside the original function")
    for left, right in zip(unique, unique[1:]):
        if left[0] + left[1] > right[0] or left[0] == right[0]:
            raise ValueError("Overlapping or ambiguously ordered replacements")
    output = raw[start:end]
    for offset, length, text in reversed(unique):
        relative = offset - start
        output = output[:relative] + text.encode("utf-8") + output[relative + length:]
    # Match dataset indentation normalization, without adding reference formatting.
    lead = raw[raw.rfind(b"\n", 0, start) + 1:start]
    indent = lead.decode("ascii") if not lead.strip() else ""
    lines = output.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if indent:
        lines = lines[:1] + [line[len(indent):] if line.startswith(indent) else line for line in lines[1:]]
    return "\n".join(lines), unique


def generate_tool_repair(row, validator):
    """Generate a candidate without consulting the stored assistant target."""
    started = time.perf_counter()
    output = {"id": row["id"], "generation_status": "unavailable", "candidate": None,
              "requested_checks": sorted(set(row["fixed_checks"])), "generation_diagnostics": [],
              "replacements": [], "generation_errors": [], "provenance": {}}
    try:
        path, raw, flags_dir, tool_config, provenance = validator._context(row)
        output["provenance"] = provenance
        before = original_code(row)
        _, start, end, _ = splice_function(raw, row, before, before)
        output["function_byte_bounds"] = [start, end]
        temp_root = ROOT / "data/repair_validation_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fresh-tool-", dir=temp_root) as temp:
            export = Path(temp) / "fixes.yaml"
            # --checks appends to configuration; the leading '-*' clears its checks.
            command = [validator.clang_tidy, str(path), f"--config-file={tool_config}",
                       "--checks=-*," + ",".join(output["requested_checks"]),
                       f"--export-fixes={export}", "--quiet", "-p", str(flags_dir)]
            output["generation_check_filter"] = "-*," + ",".join(output["requested_checks"])
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                     timeout=validator.timeout, cwd=ROOT)
            output["generation_returncode"] = process.returncode
            parsed = yaml.safe_load(export.read_text(encoding="utf-8")) if export.exists() else {}
        if not isinstance(parsed, dict) or not isinstance(parsed.get("Diagnostics"), list):
            output.update(generation_status="tool_failure", generation_errors=["No structured diagnostic export"])
            return output
        diagnostics = parsed["Diagnostics"]
        errors = [d for d in diagnostics if d.get("Level") in {"Error", "Fatal"}]
        if process.returncode != 0 or errors:
            output.update(generation_status="tool_failure", generation_errors=[
                (d.get("DiagnosticMessage") or {}).get("Message", "") for d in errors],
                generation_stderr=process.stderr[-1500:])
            return output
        replacements, outside, missing_fixes = [], [], []
        for diagnostic in diagnostics:
            message = diagnostic.get("DiagnosticMessage") or {}
            offset = message.get("FileOffset")
            if (not _same_file(message.get("FilePath", ""), path) or offset is None or not start <= offset < end
                    or diagnostic.get("DiagnosticName") not in output["requested_checks"]):
                continue
            finding = {"check": diagnostic["DiagnosticName"], "offset": offset,
                       "message": message.get("Message", "")}
            output["generation_diagnostics"].append(finding)
            edits = message.get("Replacements") or []
            if not edits:
                missing_fixes.append(finding)
            for edit in edits:
                if (not _same_file(edit.get("FilePath", ""), path) or edit["Offset"] < start
                        or edit["Length"] < 0 or edit["Offset"] + edit["Length"] > end):
                    outside.append(edit)
                else:
                    replacements.append((edit["Offset"], edit["Length"], edit.get("ReplacementText", "")))
        output["findings_without_fixes"] = missing_fixes
        if outside:
            output.update(generation_status="context_needed", outside_function_replacements=outside,
                          generation_errors=["A requested diagnostic requires edits outside the function"])
            return output
        try:
            candidate, unique = apply_function_replacements(raw, start, end, replacements)
        except ValueError as exc:
            output.update(generation_status="conflicting_edits", generation_errors=[str(exc)],
                          replacements=replacements)
            return output
        if path.read_bytes() != raw:
            output.update(generation_status="source_changed", generation_errors=["Source changed during analysis"])
            return output
        output.update(candidate=candidate, replacements=unique,
                      generation_status="generated" if unique else "no_applicable_edits",
                      candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest())
        return output
    except (OSError, ValueError, KeyError, UnicodeError, yaml.YAMLError, subprocess.SubprocessError) as exc:
        output["generation_errors"] = [str(exc)]
        return output
    finally:
        output["generation_seconds"] = round(time.perf_counter() - started, 6)


def _compact_validation(validation):
    result = dict(validation)
    for key in ("before", "after"):
        if key in result:
            part = dict(result[key])
            diagnostics = part.pop("diagnostics", None)
            if diagnostics is not None:
                part["diagnostic_counts"] = dict(Counter(d["check"] for d in diagnostics))
            result[key] = part
    return result


def _cached_validation(row, generation, reference_records):
    """Reuse a same-session result only after generation and exact byte comparison."""
    if row["id"] not in reference_records:
        return None, None
    candidate = generation["candidate"]
    # Reference code is used exclusively for evaluation/cache reuse, never generation.
    for name, text in (("no_op", original_code(row)), ("reference", row["messages"][-1]["content"])):
        if candidate.encode("utf-8") != text.encode("utf-8"):
            continue
        cached = reference_records[row["id"]].get(name, {})
        previous, current = cached.get("provenance", {}), generation["provenance"]
        fields = ("engine", "path", "source_sha1", "engine_commit", "scan_config_hash",
                  "clang_tidy_version", "compile_flags_sha256")
        if (all(previous.get(key) == current.get(key) for key in fields)
                and all(previous.get(key) is not None for key in fields)):
            return cached, name
    return None, None


def evaluate_row(row, validator, reference_records, dataset_sha256):
    from src.inference.fixing import syntax_check
    result = generate_tool_repair(row, validator)
    result.update(system="fresh_clang_tidy", system_label="fresh clang-tidy, same function-only edit scope",
                  edit_scope="function_only", engine=row["engine"], path=row["path"],
                  dataset_sha256=dataset_sha256, validation_reused_from=None, truncated=False)
    started = time.perf_counter()
    if result["candidate"] is None:
        result.update(changed=False, exact_reference=False)
        validation = {"status": "inconclusive", "static_validated": False,
                      "reasons": ["Fresh tool could not generate an in-function candidate: " + result["generation_status"]],
                      "behavioral_correctness": "not_tested", "engine_build": "not_tested"}
    else:
        result.update(changed=result["candidate"].encode() != original_code(row).encode(),
                      exact_reference=result["candidate"].encode() == row["messages"][-1]["content"].encode())
        validation, reused = _cached_validation(row, result, reference_records)
        if validation is None:
            validation = validator.validate(row, result["candidate"])
        else:
            result["validation_reused_from"] = reused
        validation = _compact_validation(validation)
    result["validation"] = validation
    syntactic = syntax_check(result["candidate"] or "")
    before_syntax = syntax_check(original_code(row))
    result["parse_ok"] = syntactic["parse_ok"] and syntactic["single_function"]
    result["parse_regression"] = before_syntax["parse_ok"] and not syntactic["parse_ok"]
    result["validation_seconds"] = round(time.perf_counter() - started, 6)
    return result


def summarize(results, dataset_path, dataset_sha256, elapsed):
    from src.evaluation.evaluate_fixing import summarize as common_summary
    summary = common_summary(results)
    candidates = [result for result in results if result["candidate"] is not None]
    abstentions = [result for result in results if result["candidate"] is None]
    return {"task": "oracle_check_conditional_repair", "system": "fresh_clang_tidy",
            "system_label": "fresh clang-tidy, same function-only edit scope", "edit_scope": "function_only",
            "dataset": str(dataset_path), "dataset_sha256": dataset_sha256,
            "summary": summary, "generation_status_counts": dict(Counter(r["generation_status"] for r in results)),
            "generation_abstentions": {"rows": len(abstentions),
                                       "status_counts": dict(Counter(r["generation_status"] for r in abstentions))},
            "candidate_static_validation": {"rows": len(candidates),
                                            "status_counts": dict(Counter(r["validation"]["status"] for r in candidates))},
            "validation_cache_reuse": dict(Counter("not_generated" if r["candidate"] is None else
                                                    (r["validation_reused_from"] or "fresh") for r in results)),
            "generation_seconds_total": round(sum(r["generation_seconds"] for r in results), 3),
            "generation_seconds_mean": round(sum(r["generation_seconds"] for r in results) / len(results), 4),
            "validation_seconds_total": round(sum(r["validation_seconds"] for r in results), 3),
            "wall_seconds": round(elapsed, 3),
            "method": {"generation_inputs": "original full source plus supplied check names",
                       "checks": "requested checks only, independently of broad validation checks",
                       "edits": "fresh structured Clang replacements, applied only in memory inside original function",
                       "source_write_mode": "never writes engine sources; export-fixes only",
                       "uses_reference_as_generation_input": False,
                       "identical_validation_reuse": "exact function bytes and matching source, engine commit, tool and flags provenance"},
            "limitations": ["Oracle check names are supplied; this is not an end-to-end detection score.",
                            "Collection uses inferred flags and may include empty header stubs.",
                            "Validation covers compiler success and enabled diagnostics, not behavioral correctness.",
                            "Standalone snippet parsing can fail on macros and defaulted members that compile in source context.",
                            "Overlapping or external edits are not guessed and count as unsuccessful attempts.",
                            "External include edits are scope abstentions; full-file tool repair performance is unmeasured.",
                            "Generation timing includes provenance loading and exports; total sums overlap across workers."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/fixing_pilot/test.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results/collected_fixing/tool_test")
    parser.add_argument("--reference-validation", type=Path, default=ROOT / "data/fixing_pilot/reference_static_validation.jsonl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--clang-tidy", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    reference_records = {}
    if args.reference_validation.is_file():
        reference_records = {r["id"]: r for line in args.reference_validation.read_text(encoding="utf-8").splitlines()
                             if line.strip() and (r := json.loads(line))}
    args.out.mkdir(parents=True, exist_ok=True)
    validator = RepairValidator(args.config, args.clang_tidy, args.timeout)
    started, results = time.perf_counter(), []
    with (args.out / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as output, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_row, row, validator, reference_records, digest) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if index % 8 == 0 or index == len(rows):
                print(f"Fresh tool {index}/{len(rows)}: " + str(dict(Counter(r["validation"]["status"] for r in results))), flush=True)
    metrics = summarize(results, args.dataset, digest, time.perf_counter() - started)
    metrics["reference_validation_file"] = str(args.reference_validation) if args.reference_validation.is_file() else None
    metrics["reference_validation_sha256"] = (hashlib.sha256(args.reference_validation.read_bytes()).hexdigest()
                                                if args.reference_validation.is_file() else None)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
