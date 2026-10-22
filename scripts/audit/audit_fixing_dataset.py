"""Audit local clang-tidy repair pairs and freeze a deterministic research pilot.

This samples existing conversations without rewriting targets or prompt templates.
The sample is balanced by requested check and engine, rather than prevalence
weighted, so report it as a stratified pilot rather than a population estimate.
No generated predictions or model scores are read when making the split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
FIX_SYSTEM = (
    "You are a C++ refactoring assistant for game-engine code. "
    "Rewrite the given function so that the listed clang-tidy findings are fixed and nothing else changes.\n"
    "Answer with the complete rewritten function only, with no explanation.\n"
)
FIX_USER_PREFIX = "Fix these clang-tidy findings: "


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_code(code):
    """Match the source builder's whitespace normalization, without AST claims."""
    return " ".join(code.split())


def code_pair(row):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError(f"Unexpected roles in {row['id']}")
    expected_header = FIX_USER_PREFIX + ", ".join(row["fixed_checks"]) + "\n\n"
    if messages[0]["content"] != FIX_SYSTEM or not messages[1]["content"].startswith(expected_header):
        raise ValueError(f"Prompt does not match collector template: {row['id']}")
    return messages[1]["content"][len(expected_header):], messages[2]["content"]


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    return digest_bytes(Path(path).read_bytes())


def percentiles(values):
    if not values:
        return {}
    ordered = sorted(values)
    result = {"min": ordered[0], "max": ordered[-1], "mean": round(sum(ordered) / len(ordered), 2)}
    for p in (25, 50, 75, 90, 95, 99):
        # Nearest-rank quantiles are explicit and need no scientific dependencies.
        result[f"p{p}"] = ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
    return result


def split_summary(rows, detection_vocab):
    pairs = [code_pair(row) for row in rows]
    checks = Counter(c for row in rows for c in row["fixed_checks"])
    inputs = [norm_code(before) for before, _ in pairs]
    targets = [norm_code(after) for _, after in pairs]
    return {
        "rows": len(rows),
        "engines": dict(sorted(Counter(r["engine"] for r in rows).items())),
        "file_error_free": sum(r["file_error_free"] for r in rows),
        "file_with_compile_errors_elsewhere": sum(not r["file_error_free"] for r in rows),
        "multi_check_rows": sum(len(r["fixed_checks"]) > 1 for r in rows),
        "checks_per_row": dict(sorted(Counter(len(r["fixed_checks"]) for r in rows).items())),
        "rows_per_check": dict(checks.most_common()),
        "single_check_rows_per_check": dict(Counter(r["fixed_checks"][0] for r in rows
                                                     if len(r["fixed_checks"]) == 1).most_common()),
        "exact_noop_rows": sum(before == after for before, after in pairs),
        "whitespace_normalized_noop_rows": sum(a == b for a, b in zip(inputs, targets)),
        "duplicate_ids": len(rows) - len({r["id"] for r in rows}),
        "duplicate_normalized_inputs": len(inputs) - len(set(inputs)),
        "duplicate_normalized_targets": len(targets) - len(set(targets)),
        "conversation_tokens_stored": percentiles([r["n_tokens"] for r in rows]),
        "input_chars": percentiles([len(a) for a, _ in pairs]),
        "target_chars": percentiles([len(b) for _, b in pairs]),
        "detection_vocabulary_overlap": {
            "checks": sorted(set(checks) & detection_vocab),
            "checks_outside_detector": sorted(set(checks) - detection_vocab),
            "rows_with_any_detector_check": sum(bool(set(r["fixed_checks"]) & detection_vocab) for r in rows),
            "rows_with_only_detector_checks": sum(set(r["fixed_checks"]) <= detection_vocab for r in rows),
        },
    }


def leakage_summary(parts):
    codes = {s: {norm_code(code_pair(r)[0]) for r in rows} for s, rows in parts.items()}
    targets = {s: {norm_code(code_pair(r)[1]) for r in rows} for s, rows in parts.items()}
    ids = {s: {r["id"] for r in rows} for s, rows in parts.items()}
    engines = {s: {r["engine"] for r in rows} for s, rows in parts.items()}
    results = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        results[f"{left}:{right}"] = {
            "shared_ids": len(ids[left] & ids[right]),
            "shared_engines": sorted(engines[left] & engines[right]),
            "shared_normalized_inputs": len(codes[left] & codes[right]),
            "shared_normalized_targets": len(targets[left] & targets[right]),
            f"{left}_input_equals_{right}_target": len(codes[left] & targets[right]),
            f"{right}_input_equals_{left}_target": len(codes[right] & targets[left]),
        }
    return results


def stable_key(seed, namespace, value):
    return digest_bytes(f"{seed}|{namespace}|{value}".encode("utf-8"))


def stratified_sample(rows, size, train_counts, seed, split):
    """Rarest requested training check, then engine, with stable hash ranking.

    Frequencies come from training candidates only. Test outcomes never influence
    stratum definitions or sampling. Every row is selected at most once.
    """
    groups = defaultdict(list)
    for row in rows:
        primary = min(row["fixed_checks"], key=lambda c: (train_counts.get(c, 0), c))
        groups[(primary, row["engine"])].append(row)
    pools = [deque(sorted(group, key=lambda r: stable_key(seed, split, r["id"])))
             for key, group in sorted(groups.items())]
    selected = []
    while pools and len(selected) < size:
        active = []
        for pool in pools:
            if len(selected) >= size:
                break
            selected.append(pool.popleft())
            if pool:
                active.append(pool)
        pools = active
    return sorted(selected, key=lambda r: r["id"])


def prepare_pilot(parts, sizes, max_tokens, seed):
    candidates = {split: [row for row in rows if row["file_error_free"] and row["n_tokens"] <= max_tokens]
                  for split, rows in parts.items()}
    train_counts = Counter(c for row in candidates["train"] for c in row["fixed_checks"])
    supported = set(train_counts)
    excluded = {}
    for split in ("val", "test"):
        eligible = [row for row in candidates[split] if set(row["fixed_checks"]) <= supported]
        excluded[split] = len(candidates[split]) - len(eligible)
        candidates[split] = eligible
    selected = {split: stratified_sample(rows, sizes[split], train_counts, seed, split)
                for split, rows in candidates.items()}
    return selected, {
        "seed": seed,
        "strict_source_rows_only": True,
        "max_stored_full_conversation_tokens": max_tokens,
        "requested_sizes": sizes,
        "candidate_sizes": {s: len(rows) for s, rows in candidates.items()},
        "excluded_eval_rows_with_checks_absent_from_train_candidates": excluded,
        "train_candidate_checks": dict(train_counts.most_common()),
        "sampling": "Round robin over (rarest requested check by train candidate frequency, engine); stable SHA256 row order",
        "statistical_population": "Purposive check-balanced short-function pilot; not prevalence-weighted full-engine performance",
        "messages_and_ids": "Preserved exactly from data/collected/fixing; source split membership retained",
        "evaluation_freeze": "Evaluation IDs and source file hashes are frozen before model predictions are read",
    }


def detection_overlap(parts, detection_dir):
    detection = {}
    for split in SPLITS:
        path = detection_dir / f"{split}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            detection[split] = {norm_code(r["messages"][1]["content"].split("\n\n", 1)[1]) for r in rows}
    result = {}
    for fixing_split, rows in parts.items():
        inputs = {norm_code(code_pair(row)[0]) for row in rows}
        result[fixing_split] = {s: len(inputs & codes) for s, codes in detection.items()}
    return result


def token_count_comparison(rows, tokenizer, max_tokens):
    """Recompute the trainer's separate prompt/answer encoding without truncation.

    The original collector used apply_chat_template(tokenize=True) with its local
    detection tokenizer. A newer base tokenizer or different encoding boundary
    can change counts. Report exclusions; never mutate the frozen sample.
    """
    counts, answer_counts, mismatches, skipped = [], [], [], []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        answer = row["messages"][-1]["content"] + "\n" + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        actual = len(prompt_ids) + len(answer_ids)
        counts.append(actual)
        answer_counts.append(len(answer_ids))
        if actual != row["n_tokens"]:
            mismatches.append({"id": row["id"], "stored": row["n_tokens"], "actual": actual})
        if actual > max_tokens:
            skipped.append(row["id"])
    return {
        "rows": len(rows), "max_training_tokens": max_tokens,
        "actual_conversation_tokens": percentiles(counts),
        "actual_answer_tokens": percentiles(answer_counts),
        "stored_count_mismatches": len(mismatches),
        "mismatch_examples_first_10": mismatches[:10],
        "retained_rows_at_limit": len(rows) - len(skipped),
        "excluded_rows_at_limit": len(skipped), "excluded_ids": skipped,
        "encoding": "Inference prompt and assistant answer tokenized separately, add_special_tokens=False; newline plus EOS appended to answer",
    }


def syntax_summary(rows):
    """Distinguish newly introduced parse errors from missing snippet context."""
    import tree_sitter_cpp
    from tree_sitter import Language, Parser
    parser = Parser(Language(tree_sitter_cpp.language()))
    states, newly_unparseable = Counter(), []
    for row in rows:
        before, after = code_pair(row)
        before_error = parser.parse(before.encode("utf-8")).root_node.has_error
        after_error = parser.parse(after.encode("utf-8")).root_node.has_error
        states[f"before_error_{before_error}_after_error_{after_error}"] += 1
        if after_error and not before_error:
            newly_unparseable.append(row["id"])
    return {"states": dict(states), "newly_unparseable_ids": newly_unparseable}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/collected/fixing")
    parser.add_argument("--output", type=Path, default=ROOT / "data/fixing_pilot")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/collected/detection/labels.json")
    parser.add_argument("--detection", type=Path, default=ROOT / "data/clean/detection")
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--compare-tokenizer", help="Only compare the frozen output files with this locally cached tokenizer; do not resample")
    parser.add_argument("--check-syntax-only", action="store_true", help="Only audit raw C++ syntax in the frozen output files; do not resample")
    args = parser.parse_args(argv)
    if min(args.train_size, args.val_size, args.test_size, args.max_tokens) < 1:
        parser.error("Sizes and token limit must be positive")
    if args.check_syntax_only:
        result = {
            "parser": "tree_sitter_cpp raw extracted function, no declaration wrapper",
            "manifest_sha256": file_sha256(args.output / "manifest.json"),
            "results": {s: syntax_summary(read_jsonl(args.output / f"{s}.jsonl")) for s in SPLITS},
        }
        (args.output / "syntax_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return result
    if args.compare_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.compare_tokenizer, local_files_only=True)
        result = {
            "tokenizer": args.compare_tokenizer,
            "manifest_sha256": file_sha256(args.output / "manifest.json"),
            "frozen_outputs_sha256": {s: file_sha256(args.output / f"{s}.jsonl") for s in SPLITS},
            "comparison": {s: token_count_comparison(read_jsonl(args.output / f"{s}.jsonl"), tokenizer, args.max_tokens)
                           for s in SPLITS},
        }
        path = args.output / "runtime_tokenization.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({s: {k: v for k, v in r.items() if k not in {"excluded_ids", "mismatch_examples_first_10"}}
                          for s, r in result["comparison"].items()}, indent=2))
        return result
    source = {s: args.source / f"{s}.jsonl" for s in SPLITS}
    parts = {s: read_jsonl(path) for s, path in source.items()}
    detection_vocab = set(json.loads(args.labels.read_text(encoding="utf-8"))["checks"])
    summaries = {s: split_summary(rows, detection_vocab) for s, rows in parts.items()}
    sizes = {s: getattr(args, f"{s}_size") for s in SPLITS}
    pilot, design = prepare_pilot(parts, sizes, args.max_tokens, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in pilot.items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {s: file_sha256(path) for s, path in source.items()},
        "output_sha256": {s: file_sha256(args.output / f"{s}.jsonl") for s in SPLITS},
        "design": design,
        "ids": {s: [r["id"] for r in rows] for s, rows in pilot.items()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "source": str(args.source.resolve()),
        "detection_vocabulary_size": len(detection_vocab),
        "full_dataset": summaries,
        "full_dataset_cross_split_overlap": leakage_summary(parts),
        "fixing_input_overlap_with_detection_inputs": detection_overlap(parts, args.detection),
        "pilot": {s: split_summary(rows, detection_vocab) for s, rows in pilot.items()},
        "pilot_cross_split_overlap": leakage_summary(pilot),
        "design": design,
        "limits": [
            "Targets are clang-tidy automatic replacements, not human-verified semantic fixes.",
            "A clang-tidy --fix baseline is essential: teacher imitation alone does not establish an advantage over the teacher.",
            "Strict means the collector recorded no compiler errors in the original translation unit; it is not evidence repaired outputs compile.",
            "Some original files use generated dependency stubs; strict collector eligibility is weaker than a successful native engine build.",
            "Whitespace-normalized exact match is a reproduction metric, not semantic correctness.",
            "Conversation token counts are stored builder counts, not recomputed by this lightweight audit.",
            "Multi-check targets can mix semantic and style changes; report per-check results and collateral edits separately.",
            "Only checks in both detector vocabulary and repair support can be claimed as detector-to-repair end-to-end coverage.",
            "No-op repairs and clean-function controls are absent from these positive-only repair pairs.",
            "Whitespace normalization detects exact copied functions but cannot rule out near-duplicate leakage or pretraining contamination.",
        ],
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_rows": {s: len(r) for s, r in parts.items()},
                      "pilot_rows": {s: len(r) for s, r in pilot.items()},
                      "pilot_engines": {s: audit["pilot"][s]["engines"] for s in SPLITS},
                      "output": str(args.output.resolve())}, indent=2))
    return audit


if __name__ == "__main__":
    main()
