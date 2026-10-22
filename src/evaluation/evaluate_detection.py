"""Score a model on the collected detection dataset, next to simple baselines.

    python src/evaluation/evaluate_detection.py --name base                      # untuned base model
    python src/evaluation/evaluate_detection.py --name lora --adapter models/collected_detection_lora
    python src/evaluation/evaluate_detection.py --name lora --adapter ... --split val

Generates an answer for every row of the split, parses it, and reports the same numbers for the
model, for "always clean", and for the length rule. Results for the length-matched subset of the
split are computed from the same predictions. Writes metrics.json and predictions.jsonl to
results/collected_detection/<name>/<split>/.

Numbers reported:
  valid          share of answers that parse as {"checks": [...]}
  exact          share of rows whose predicted set of checks equals the reference set
  smelly P/R/F1  the binary question "does any check apply?"
  micro P/R/F1   over every (row, check) pair
  macro F1       mean F1 over the checks that occur in the split
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def parse_answer(text):
    """(set of checks, parsed ok). Accepts the JSON object anywhere in the text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict) and isinstance(obj.get("checks"), list):
                    return {c for c in obj["checks"] if isinstance(c, str)}, True
                break
        start = text.find("{", start + 1)
    return set(), False


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold, pred, valid=None):
    """Metrics for parallel lists of reference and predicted check sets."""
    n = len(gold)
    tp = fp = fn = 0
    per_check = Counter(), Counter(), Counter()
    b_tp = b_fp = b_fn = b_tn = 0
    for g, p in zip(gold, pred):
        for c in g & p:
            per_check[0][c] += 1
        for c in p - g:
            per_check[1][c] += 1
        for c in g - p:
            per_check[2][c] += 1
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        b_tp += bool(g) and bool(p)
        b_fp += (not g) and bool(p)
        b_fn += bool(g) and (not p)
        b_tn += (not g) and (not p)
    micro = prf(tp, fp, fn)
    binary = prf(b_tp, b_fp, b_fn)
    in_split = sorted({c for g in gold for c in g})
    f1s = {c: prf(per_check[0][c], per_check[1][c], per_check[2][c])[2] for c in in_split}
    out = {"rows": n, "exact": sum(g == p for g, p in zip(gold, pred)) / n,
           "smelly_accuracy": (b_tp + b_tn) / n, "smelly_precision": binary[0], "smelly_recall": binary[1],
           "smelly_f1": binary[2], "micro_precision": micro[0], "micro_recall": micro[1], "micro_f1": micro[2],
           "macro_f1": sum(f1s.values()) / len(f1s) if f1s else 0.0,
           "per_check": {c: {"support": per_check[0][c] + per_check[2][c], "f1": round(f1s[c], 3),
                             "predicted": per_check[0][c] + per_check[1][c]} for c in in_split}}
    if valid is not None:
        out["valid"] = sum(valid) / n
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def length_rule(train_rows):
    """Token threshold that best separates smelly from clean on the training split."""
    pairs = sorted((r["n_tokens"], bool(r["checks"])) for r in train_rows)
    total_smelly = sum(s for _, s in pairs)
    best, best_acc, smelly_below, clean_below = 0, total_smelly / len(pairs), 0, 0
    for i, (n, s) in enumerate(pairs):
        smelly_below += s
        clean_below += not s
        if i + 1 < len(pairs) and pairs[i + 1][0] == n:
            continue
        acc = (clean_below + (total_smelly - smelly_below)) / len(pairs)  # predict smelly above n
        if acc > best_acc:
            best, best_acc = n + 1, acc
    return best


def pack(order, lengths, batch_size, token_limit):
    """Batches of at most batch_size prompts and token_limit padded tokens; fewer rows when prompts are long."""
    batches, current = [], []
    for j in order:
        longest = max(lengths[j], lengths[current[0]]) if current else lengths[j]
        if current and (len(current) == batch_size or (len(current) + 1) * longest > token_limit):
            batches.append(current)
            current = []
        current.append(j)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def generate(model, tok, prompts, batch_size, max_new_tokens):
    lengths = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])
    outputs = [None] * len(prompts)
    started, finished = time.time(), 0
    for done, idx in enumerate(pack(order, lengths, batch_size, batch_size * 1024), 1):
        enc = tok([prompts[j] for j in idx], add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, t in zip(idx, texts):
            outputs[j] = t.strip()
        finished += len(idx)
        if done % 20 == 0:
            print(f"  generated {finished}/{len(order)}  ({finished / (time.time() - started):.1f} rows/s)", flush=True)
    return outputs


def show(name, m):
    print(f"{name:26s} rows {m['rows']:5d}  valid {m.get('valid', 1):6.1%}  exact {m['exact']:6.1%}  "
          f"smelly acc {m['smelly_accuracy']:6.1%} F1 {m['smelly_f1']:.3f}  "
          f"micro P {m['micro_precision']:.3f} R {m['micro_recall']:.3f} F1 {m['micro_f1']:.3f}  macro F1 {m['macro_f1']:.3f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="label for this run; results go to results/collected_detection/<name>")
    ap.add_argument("--adapter", help="LoRA adapter folder; omit to score the untuned base model")
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    ap.add_argument("--data", default="data/collected/detection")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--limit", type=int, help="score only the first N rows (for a trial run)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = ROOT / args.data
    rows = read_jsonl(data / f"{args.split}.jsonl", args.limit)
    matched_ids = {r["id"] for r in read_jsonl(data / f"{args.split}_length_matched.jsonl")}
    train_rows = read_jsonl(data / "train.jsonl")
    vocab = set(json.loads((data / "labels.json").read_text(encoding="utf-8"))["checks"])

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter)).merge_and_unload()
    model.eval()

    prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in rows]
    print(f"{args.name}: generating for {len(rows)} rows of {args.split}", flush=True)
    started = time.time()
    raw = generate(model, tok, prompts, args.batch_size, args.max_new_tokens)
    seconds = time.time() - started

    gold = [set(r["checks"]) for r in rows]
    parsed = [parse_answer(t) for t in raw]
    pred, valid = [p for p, _ in parsed], [ok for _, ok in parsed]
    unknown = Counter(c for p in pred for c in p if c not in vocab)

    threshold = length_rule(train_rows)
    top_label = Counter(c for r in train_rows for c in r["checks"]).most_common(1)[0][0]
    systems = {
        "model": (pred, valid),
        "baseline: always clean": ([set()] * len(rows), None),
        "baseline: length rule": ([{top_label} if r["n_tokens"] >= threshold else set() for r in rows], None),
    }
    subsets = {"all": list(range(len(rows))),
               "length_matched": [i for i, r in enumerate(rows) if r["id"] in matched_ids]}
    metrics = {"name": args.name, "split": args.split, "adapter": args.adapter, "model": args.model,
               "seconds": round(seconds, 1), "length_rule_threshold": threshold, "length_rule_label": top_label,
               "predicted_checks_outside_vocabulary": dict(unknown.most_common(15)), "results": {}}
    for subset, idx in subsets.items():
        print(f"\n--- {args.split} / {subset} ---")
        metrics["results"][subset] = {}
        for system, (p, v) in systems.items():
            m = score([gold[i] for i in idx], [p[i] for i in idx], [v[i] for i in idx] if v else None)
            metrics["results"][subset][system] = m
            show(system if system != "model" else args.name, m)

    out = ROOT / "results" / "collected_detection" / args.name / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    with open(out / "predictions.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r, t, p, ok in zip(rows, raw, pred, valid):
            f.write(json.dumps({"id": r["id"], "engine": r["engine"], "reference": r["checks"],
                                "predicted": sorted(p), "valid": ok, "raw": t[:400]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}  ({seconds:.0f} s of generation)")
    if unknown:
        print("checks predicted that are not in the vocabulary:", dict(unknown.most_common(5)))


if __name__ == "__main__":
    main()
