"""Evaluate detection -> supported learned repair -> independent static validation.

The evaluation uses every frozen positive TEST row whose recorded repair checks
are in both the detector vocabulary and repair training support. Reference checks
select this restricted diagnostic pilot and score the final patch; they are never
provided to either model. This is not a broad end-to-end benchmark and contains no
clean-function controls from which to estimate false-positive repair precision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.detection_demo import Detector
from src.inference.fixing import DEFAULT_ADAPTER, Fixer, syntax_check


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def original_code(row):
    """Extract model input only; assistant references never enter inference."""
    return row["messages"][1]["content"].split("\n\n", 1)[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_provenance(path):
    path = Path(path).resolve()
    files = ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "run_settings.json", "trained_checks.json")
    return {"path": str(path), "sha256": {name: sha256(path / name) for name in files if (path / name).is_file()}}


def select_rows(rows, detector_checks, repair_checks):
    eligible = set(detector_checks) & set(repair_checks)
    return [row for row in rows if row["fixed_checks"] and set(row["fixed_checks"]) <= eligible]


def infer_row(row, detector, fixer, detector_checks, repair_checks, max_new_tokens=1536):
    """Infer using source and predictions only, with explicit abstention gates.

    No access to fixed_checks or the assistant's reference is needed in this
    function. A raw snippet parse failure is not treated as a compiler verdict;
    independent full-context validation decides syntax when macros need context.
    """
    code = original_code(row)
    result = {"id": row["id"], "engine": row["engine"], "path": row["path"],
              "original_code": code, "candidate": code, "detected_checks": [],
              "unsupported_checks": [], "unknown_detector_checks": [],
              "detector": None, "repair": None, "changed": False,
              "repair_truncated": False, "inference_accepted": False, "pipeline_status": "not_run"}
    stage = "detector"
    started = time.perf_counter()
    try:
        trace = None
        for update in detector.stream(code):
            trace = update
        result["detector"] = trace
        if not trace or not trace.get("finished") or not trace.get("valid"):
            result["pipeline_status"] = "invalid_detector_output"
            return result
        if trace.get("stopped_at_limit"):
            result["pipeline_status"] = "truncated_detector_output"
            return result
        checks = trace.get("parsed", {}).get("checks")
        if not isinstance(checks, list) or any(not isinstance(check, str) for check in checks):
            result["pipeline_status"] = "invalid_detector_output"
            return result
        predicted = sorted(set(checks))
        result["detected_checks"] = predicted
        result["unknown_detector_checks"] = sorted(set(predicted) - set(detector_checks))
        result["unsupported_checks"] = sorted(set(predicted) - set(repair_checks))
        if result["unknown_detector_checks"] or result["unsupported_checks"]:
            result["pipeline_status"] = "abstained_unsupported_prediction"
            return result
        if not predicted:
            # Preserve the original rather than supplying reference labels to a fixer.
            result["pipeline_status"] = "detector_requested_no_repair"
            result["inference_accepted"] = True
            return result
        stage = "repair"
        repaired = fixer.repair(code, predicted, max_new_tokens=max_new_tokens)
        result["repair"] = repaired
        candidate = repaired.get("candidate")
        result["candidate"] = candidate if isinstance(candidate, str) else ""
        result["changed"] = code.strip() != result["candidate"].strip()
        result["repair_truncated"] = bool(repaired.get("truncated"))
        if result["repair_truncated"]:
            result["pipeline_status"] = "truncated_repair_output"
        elif not result["candidate"].strip():
            result["pipeline_status"] = "empty_repair_output"
        else:
            before_syntax = syntax_check(code)
            after_syntax = repaired.get("syntax") or syntax_check(result["candidate"])
            result["original_snippet_syntax"] = before_syntax
            if before_syntax["parse_ok"] and not after_syntax["parse_ok"]:
                result["pipeline_status"] = "fresh_snippet_parse_error"
            elif not after_syntax["single_function"]:
                result["pipeline_status"] = "invalid_repair_function_count"
            else:
                result["pipeline_status"] = "repair_generated"
                result["inference_accepted"] = True
    except Exception as exc:
        result["pipeline_status"] = stage + "_error"
        result["error"] = str(exc)
    finally:
        result["inference_seconds"] = round(time.perf_counter() - started, 3)
    return result


def score_result(result, row, validation):
    """Only scoring may inspect the reference repair labels and target."""
    scored = dict(result)
    gold = sorted(set(row["fixed_checks"]))
    predicted = set(result["detected_checks"])
    scored.update(reference_checks=gold, target_checks_detected=sorted(set(gold) & predicted),
                  missed_target_checks=sorted(set(gold) - predicted),
                  exact_recorded_tool_target=result["candidate"].strip() == row["messages"][2]["content"].strip(),
                  static_validation=validation,
                  pipeline_pass=bool(result["inference_accepted"] and validation.get("static_validated")))
    return scored


def summarize(results):
    n = len(results)
    passed = sum(row["pipeline_pass"] for row in results)
    checks = sorted({check for row in results for check in row["reference_checks"]})
    target_count = sum(len(row["reference_checks"]) for row in results)
    detected_count = sum(len(row["target_checks_detected"]) for row in results)

    def group_summary(group):
        return {"rows": len(group), "pipeline_passes": sum(r["pipeline_pass"] for r in group),
                "pipeline_pass_rate": round(sum(r["pipeline_pass"] for r in group) / len(group), 4)}

    return {"rows": n, "pipeline_passes": passed, "pipeline_static_pass_rate": round(passed / n, 4) if n else None,
            "pipeline_status_counts": dict(Counter(row["pipeline_status"] for row in results)),
            "static_validation_status_counts": dict(Counter(row["static_validation"]["status"] for row in results)),
            "changed_rows": sum(row["changed"] for row in results),
            "unsupported_prediction_rows": sum(bool(row["unsupported_checks"] or row["unknown_detector_checks"]) for row in results),
            "detector_requested_no_repair_rows": sum(row["pipeline_status"] == "detector_requested_no_repair" for row in results),
            "repair_truncation_rows": sum(row["repair_truncated"] for row in results),
            "recorded_target_check_recall": round(detected_count / target_count, 4) if target_count else None,
            "recorded_target_checks_detected": detected_count, "recorded_target_checks_total": target_count,
            "exact_recorded_tool_target_rows": sum(row["exact_recorded_tool_target"] for row in results),
            "per_engine": {engine: group_summary([r for r in results if r["engine"] == engine])
                           for engine in sorted({r["engine"] for r in results})},
            "per_reference_check_full_patch_validation": {check: group_summary([r for r in results if check in r["reference_checks"]])
                                                          for check in checks},
            "clean_function_precision": None, "behavioral_correctness": "not_tested"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/fixing_pilot/test.jsonl")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/collected_detection_lora")
    parser.add_argument("--fixer", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=ROOT / "results/collected_fixing/pipeline_test")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-inference", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.data.resolve()
    rows = read_rows(source)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset contains duplicate IDs")
    detector = Detector(args.detector)
    fixer = Fixer(args.fixer)
    support = fixer.supported_checks()
    if not support:
        raise ValueError("The fixer requires a nonempty trained_checks.json support manifest")
    selected = select_rows(rows, detector.vocab, support)
    if not selected:
        raise ValueError("No positive rows have target checks supported by both stages")
    design = {
        "task": "detector_to_learned_repair_with_static_validation",
        "source_dataset": str(source), "source_dataset_sha256": sha256(source),
        "source_manifest_sha256": sha256(source.parent / "manifest.json") if (source.parent / "manifest.json").exists() else None,
        "source_rows": len(rows), "selected_rows": len(selected),
        "selected_ids": [row["id"] for row in selected],
        "selection_rule": "All positive rows whose complete recorded fixed_checks are a subset of detector vocabulary intersect actual repair trained_checks",
        "selected_engines": dict(sorted(Counter(row["engine"] for row in selected).items())),
        "selected_reference_checks": dict(Counter(c for row in selected for c in row["fixed_checks"]).most_common()),
        "detector_vocabulary": sorted(detector.vocab), "repair_training_checks": sorted(support),
        "detector": adapter_provenance(args.detector), "fixer": adapter_provenance(args.fixer),
        "inference_code_sha256": {path: sha256(ROOT / path) for path in (
            "src/detection_prompt.py", "src/fixing_prompt.py", "src/inference/detection_demo.py", "src/inference/fixing.py")},
        "max_new_tokens": args.max_new_tokens,
        "model_inference_receives_oracle_checks": False,
        "acceptance": "Complete supported inference plus independent static validation against ORIGINAL row fixed_checks",
        "limitations": [
            "Only a small selected positive subset of the frozen stratified pilot TEST split; no broad end-to-end score.",
            "Selection uses reference check support eligibility; it is conditional on repairable known-check positives.",
            "No clean-function controls; false-positive repair precision and clean-function safety are unmeasured.",
            "Recorded fixed_checks are tool-edit targets, not every possible warning; detector precision is not estimated from this partial label set.",
            "One detector checkpoint, one repair seed, two held-out engines, no behavioral engine tests.",
            "Static validation uses collection flags and may include stub headers; it is not a native engine build.",
        ],
    }
    manifest_path = args.out / "manifest.json"
    inference_path = args.out / "inference.jsonl"
    if args.reuse_inference:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["design"] != design:
            raise ValueError("Stored pipeline inference provenance differs from this request")
        inferred = read_rows(inference_path)
        if Counter(row["id"] for row in inferred) != Counter(row["id"] for row in selected):
            raise ValueError("Stored inference has incomplete or duplicate row coverage")
    else:
        manifest_path.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "design": design}, indent=2) + "\n", encoding="utf-8")
        inferred = []
        inference_path.write_text("", encoding="utf-8")
        for index, row in enumerate(selected, 1):
            result = infer_row(row, detector, fixer, detector.vocab, support, args.max_new_tokens)
            inferred.append(result)
            with inference_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Inference {index}/{len(selected)} {row['id']} {result['pipeline_status']}", flush=True)
    # Release both models before the independent compiler work begins.
    detector.model = fixer.model = None
    del detector, fixer
    gc.collect()
    if not args.reuse_inference:
        import torch
        torch.cuda.empty_cache()
    from src.evaluation.repair_validation import RepairValidator
    validator = RepairValidator()
    row_by_id = {row["id"]: row for row in selected}

    def validate(result):
        row = row_by_id[result["id"]]
        # The original reference targets are used only in this independent scorer.
        validation = validator.validate(row, result["candidate"])
        return score_result(result, row, validation)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(validate, inferred))
    (args.out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored), encoding="utf-8")
    metrics = {"design": design, "manifest_sha256": sha256(manifest_path),
               "inference_sha256": sha256(inference_path), "results_sha256": sha256(args.out / "results.jsonl"),
               "validation_seconds": round(time.perf_counter() - started, 3), "metrics": summarize(scored)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["metrics"], indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    main()
