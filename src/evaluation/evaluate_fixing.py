"""Compare a repair adapter, untuned base, no-op and recorded clang-tidy edits.

The supplied check names are reference/oracle checks. This is conditional repair,
not an end-to-end detector score. Stored tool targets are a teacher baseline;
they must pass the same static validation as model outputs.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.inference.fixing import candidate_code, syntax_check


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def original_code(row):
    return row["messages"][1]["content"].split("\n\n", 1)[1]


def generate_repairs(model, tok, prompts, batch_size, max_new_tokens):
    import torch
    from src.evaluation.evaluate_detection import pack
    lengths = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
    if any(length + max_new_tokens > 4096 for length in lengths):
        raise ValueError("A prompt plus reserved output exceeds 4096 tokens; evaluation never truncates.")
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    results = [None] * len(prompts)
    with torch.inference_mode():
        for batch_no, indices in enumerate(pack(order, lengths, batch_size, batch_size * 512), 1):
            encoded = tok([prompts[i] for i in indices], padding=True, add_special_tokens=False,
                          return_tensors="pt").to(model.device)
            outputs = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                                     eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            for index, ids in zip(indices, outputs[:, encoded["input_ids"].shape[1]:]):
                # Decoder batches pad after EOS. Count only the continuation up to its first EOS.
                tokens = ids.tolist()
                ended = tok.eos_token_id in tokens
                if ended:
                    tokens = tokens[:tokens.index(tok.eos_token_id) + 1]
                results[index] = {"raw": tok.decode(tokens, skip_special_tokens=True).strip(),
                                  "truncated": not ended and len(tokens) >= max_new_tokens,
                                  "generated_tokens": len(tokens)}
            print(f"  generated {sum(r is not None for r in results)}/{len(prompts)}", flush=True)
    return results


def summarize(rows):
    n = len(rows)
    def rate(key):
        return round(sum(bool(row.get(key)) for row in rows)/n, 4) if n else None
    static = [r for r in rows if r.get("validation")]
    counts = Counter(r["validation"]["status"] for r in static)
    def wilson(k, total):
        if not total:
            return None
        z = 1.959963984540054
        p, z2 = k / total, z*z
        center = (p + z2/(2*total))/(1+z2/total)
        radius = z*math.sqrt(p*(1-p)/total+z2/(4*total*total))/(1+z2/total)
        return [round(center-radius, 4), round(center+radius, 4)]
    def static_count(items):
        passed = sum(r.get("validation", {}).get("static_validated", False) for r in items)
        return {"rows": len(items), "passed": passed,
                "pass_rate": round(passed/len(items), 4), "wilson_95": wilson(passed, len(items))}
    return {"rows": n, "exact_reference_rate": rate("exact_reference"),
            "changed_rate": rate("changed"), "parse_rate": rate("parse_ok"),
            "truncation_rate": rate("truncated"), "fresh_snippet_parse_regression_rate": rate("parse_regression"),
            "static_validation_rows": len(static),
            "static_status_counts": dict(counts),
            "static_pass_rate_over_all_attempts": round(counts["passed_static_validation"]/n, 4) if static else None,
            "static_pass_wilson_95": wilson(counts["passed_static_validation"], n) if static else None,
            "per_engine": {engine: static_count([r for r in static if r["engine"] == engine])
                           for engine in sorted({r["engine"] for r in static})},
            "per_requested_check_full_patch_validation": {
                check: static_count([r for r in static if check in r["requested_checks"]])
                for check in sorted({c for r in static for c in r["requested_checks"]})},
            "behavioral_correctness": "not_tested_on_engine_functions"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fixing_pilot")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--adapter", default="models/collected_fixing_pilot_r16")
    ap.add_argument("--out", default="results/collected_fixing/pilot_test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--reuse-predictions", action="store_true")
    ap.add_argument("--reparse-predictions", action="store_true", help="re-extract saved raw generations with the current common code-block parser")
    ap.add_argument("--reference-validation", help="same-session validation of recorded tool edits/no-ops, keyed by row id")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.data / f"{args.split}.jsonl"
    rows = read_rows(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    predictions_path = out / "predictions.jsonl"
    metadata_path = out / "generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if args.reuse_predictions and metadata_path.exists() else {}
    if args.reuse_predictions:
        predictions = read_rows(predictions_path)
        if any(p["dataset_sha256"] != digest for p in predictions):
            raise ValueError("Stored predictions do not match this evaluation manifest.")
        current_adapter_hash = hashlib.sha256((ROOT / args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        if metadata.get("adapter_sha256") != current_adapter_hash:
            raise ValueError("Stored predictions belong to another adapter checkpoint.")
        if metadata.get("max_new_tokens") != args.max_new_tokens or metadata.get("batch_size") != args.batch_size:
            raise ValueError("Stored prediction generation settings differ from this request.")
        from transformers import AutoTokenizer
        cache_tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", local_files_only=True)
        tokenizer_hash = hashlib.sha256((cache_tok.backend_tokenizer.to_str() + str(cache_tok.chat_template)).encode()).hexdigest()
        if metadata.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("Stored predictions use another tokenizer/chat template.")
        expected_systems = {"no_op", "clang_tidy_recorded_edits", "adapter"} | (set() if args.skip_base else {"base"})
        expected = {(system, row["id"]) for system in expected_systems for row in rows}
        actual = Counter((p["system"], p["id"]) for p in predictions)
        if set(actual) != expected or any(count != 1 for count in actual.values()):
            raise ValueError("Stored predictions have incomplete or duplicate system/row coverage.")
        if args.reparse_predictions:
            row_by_id = {row["id"]: row for row in rows}
            for pred in predictions:
                row = row_by_id[pred["id"]]
                code = candidate_code(pred["raw"])
                syntax = syntax_check(code)
                if code != pred["candidate"]:
                    pred.pop("validation", None)
                pred.update(candidate=code, exact_reference=code.strip() == row["messages"][2]["content"].strip(),
                            changed=code.strip() != original_code(row).strip(),
                            parse_ok=syntax["parse_ok"] and syntax["single_function"],
                            parse_regression=syntax_check(original_code(row))["parse_ok"] and not syntax["parse_ok"],
                            plain_code_format=pred["raw"].strip() == code.strip())
            metadata["code_extraction"] = "single complete C++ fence allowed with surrounding prose; multiple blocks rejected"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import transformers
        import peft
        name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in rows]
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
        metadata = {"model": name, "model_commit": getattr(model.config, "_commit_hash", None),
                    "tokenizer_sha256": hashlib.sha256((tok.backend_tokenizer.to_str()+str(tok.chat_template)).encode()).hexdigest(),
                    "adapter_sha256": hashlib.sha256((ROOT / args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
                    "dataset_sha256": digest, "torch": torch.__version__, "transformers": transformers.__version__,
                    "peft": peft.__version__, "gpu": torch.cuda.get_device_name(0),
                    "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                    "generation_seconds": {}}
        systems = {"no_op": [{"raw": original_code(r), "truncated": False} for r in rows],
                   "clang_tidy_recorded_edits": [{"raw": r["messages"][2]["content"], "truncated": False} for r in rows]}
        if not args.skip_base:
            print(f"Generating untuned base outputs for {len(rows)} rows", flush=True)
            started = time.perf_counter()
            systems["base"] = generate_repairs(model, tok, prompts, args.batch_size, args.max_new_tokens)
            metadata["generation_seconds"]["base"] = round(time.perf_counter() - started, 3)
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter), local_files_only=True).merge_and_unload().eval()
        print(f"Generating repair adapter outputs for {len(rows)} rows", flush=True)
        started = time.perf_counter()
        systems["adapter"] = generate_repairs(model, tok, prompts, args.batch_size, args.max_new_tokens)
        metadata["generation_seconds"]["adapter"] = round(time.perf_counter() - started, 3)
        predictions = []
        for system, outputs in systems.items():
            for row, output in zip(rows, outputs):
                raw = output["raw"]
                code = candidate_code(raw)
                syntactic = syntax_check(code)
                before_syntax = syntax_check(original_code(row))
                predictions.append({"id": row["id"], "engine": row["engine"], "system": system,
                    "dataset_sha256": digest, "candidate": code, "raw": raw,
                    "requested_checks": row["fixed_checks"],
                    "plain_code_format": raw.strip() == code.strip(),
                    "exact_reference": code.strip() == row["messages"][2]["content"].strip(),
                    "changed": code.strip() != original_code(row).strip(),
                    "parse_ok": syntactic["parse_ok"] and syntactic["single_function"],
                    "parse_regression": before_syntax["parse_ok"] and not syntactic["parse_ok"],
                    "truncated": output["truncated"], "generated_tokens": output.get("generated_tokens")})
        del model
        torch.cuda.empty_cache()
        predictions_path.write_text("".join(json.dumps(p)+"\n" for p in predictions), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.validate:
        from src.evaluation.repair_validation import RepairValidator
        validator = RepairValidator()
        row_by_id = {r["id"]: r for r in rows}
        if args.reference_validation:
            reference_path = ROOT / args.reference_validation
            references = {r["id"]: r for r in read_rows(reference_path)}
            metadata["reference_validation_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            for pred in predictions:
                key = {"no_op": "no_op", "clang_tidy_recorded_edits": "reference"}.get(pred["system"])
                if key is None:
                    source_row = row_by_id[pred["id"]]
                    if pred["candidate"].strip() == source_row["messages"][2]["content"].strip():
                        key = "reference"
                    elif pred["candidate"].strip() == original_code(source_row).strip():
                        key = "no_op"
                if key and pred["id"] in references:
                    pred["validation"] = references[pred["id"]][key]
        # Validate duplicate source/candidate pairs once; no-op, reference and model often coincide.
        groups = {}
        for pred in predictions:
            if pred["truncated"]:
                pred["validation"] = {"status": "rejected", "static_validated": False,
                    "reasons": ["Generation reached the output limit without EOS."],
                    "validation_scope": "generation completeness gate", "behavioral_correctness": "not_tested"}
            if not pred.get("validation"):
                groups.setdefault((pred["id"], pred["candidate"]), []).append(pred)
        def validate(item):
            (row_id, candidate), duplicates = item
            result = validator.validate(row_by_id[row_id], candidate)
            return duplicates, result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, (duplicates, result) in enumerate(pool.map(validate, groups.items()), 1):
                for pred in duplicates:
                    pred["validation"] = result
                if index % 10 == 0:
                    print(f"Static validation {index}/{len(groups)}", flush=True)
        predictions_path.write_text("".join(json.dumps(p)+"\n" for p in predictions), encoding="utf-8")
    metrics = {"task": "oracle_check_conditional_repair", "split": args.split,
               "dataset_sha256": digest, "adapter": args.adapter,
               "generation": metadata,
               "systems": {system: summarize([p for p in predictions if p["system"] == system])
                           for system in sorted({p["system"] for p in predictions})},
               "limitations": ["One seed and a restricted pilot dataset.",
                               "Tool edits are teacher targets, not human-validated ground truth.",
                               "Static validation does not establish behavioral equivalence.",
                               "Recorded clang-tidy edits are compared, not a timed fresh --fix tool run."]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
