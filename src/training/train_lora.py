"""Fine-tune a chat model with LoRA on one of the collected datasets.

    python src/training/train_lora.py                                   # detection, default settings
    python src/training/train_lora.py --data data/collected/fixing --out models/collected_fixing_lora
    python src/training/train_lora.py --limit 200 --epochs 1 --out models/smoke   # quick trial

Reads <data>/train.jsonl and <data>/val.jsonl (chat-format `messages`), writes the LoRA adapter
with the lowest validation loss, a step log and the run settings to --out.

The default LoRA and epoch settings follow Zhang et al., "A Comprehensive Evaluation of
Parameter-Efficient Fine-Tuning on Code Smell Detection": rank 8, alpha 16, dropout 0.1 on the
query, key and value projections, learning rate 3e-4, up to 10 epochs with the epoch of lowest
validation loss kept. One addition: training stops once --patience epochs pass without a better
validation loss. The settings of the first run (docs/TRAINING.md) are:

    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --lora-targets all --lr 2e-4 --epochs 3

What it does differently from the 2025 notebook, and why (see docs/AUDIT.md):
  * The prompt is rendered with the model's own chat template, not an [INST] wrapper.
  * Only the assistant's answer is supervised. The prompt and the code carry no loss.
  * Nothing is truncated: a conversation longer than --max-length is skipped and counted.
  * The base model is loaded in bf16, not 4-bit; a 1.3B model needs no quantisation on a 24 GB card.

Needs torch, transformers and peft (requirements-train.txt).
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
ATTENTION_QKV = ["q_proj", "k_proj", "v_proj"]
ALL_PROJECTIONS = ATTENTION_QKV + ["o_proj", "gate_proj", "up_proj", "down_proj"]


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def split_conversation(tok, messages):
    """(prompt text, answer text) exactly as the chat template lays them out.

    The prompt is what the model is given at inference; the answer ends with the end-of-turn token.
    """
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    return prompt, messages[-1]["content"] + "\n" + tok.eos_token


def encode(tok, messages):
    """Token ids and labels. Prompt and answer are tokenised separately, as they are at inference,
    so the boundary between them is exact and only answer tokens get a label."""
    prompt, answer = split_conversation(tok, messages)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a


def check_encoding(tok, messages):
    """Fail early if the template or tokenizer does not behave as encode() assumes."""
    prompt, answer = split_conversation(tok, messages)
    full = tok.apply_chat_template(messages, tokenize=False)
    assert full.startswith(prompt) and full[len(prompt):].rstrip("\n") == answer, "chat template layout changed"
    ids, labels = encode(tok, messages)
    assert ids.count(tok.bos_token_id) == 1 and ids[0] == tok.bos_token_id, "expected exactly one BOS, at the start"
    supervised = [t for t in labels if t != -100]
    assert supervised[-1] == tok.eos_token_id, "the answer must end with the end-of-turn token"
    assert tok.decode(supervised) == answer, "supervised tokens do not decode to the answer"
    return len(ids), len(supervised)


def pack(indices, lengths, batch_size, token_limit):
    """Split length-sorted indices into batches of at most batch_size rows and token_limit padded tokens.

    Memory grows with the number of tokens in the batch (measured: about 3 GB per 1,000 tokens for
    this model with LoRA on every projection), so long conversations have to go in smaller batches.
    """
    batches, current = [], []
    for j in indices:
        longest = max(lengths[j], lengths[current[0]]) if current else lengths[j]
        if current and (len(current) == batch_size or (len(current) + 1) * longest > token_limit):
            batches.append(current)
            current = []
        current.append(j)
    if current:
        batches.append(current)
    return batches


def make_batches(lengths, batch_size, cost_limit, rng):
    """Batches of similar length (less padding), in random order."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    pool = batch_size * 50
    batches = []
    for i in range(0, len(order), pool):
        batches += pack(sorted(order[i:i + pool], key=lambda j: lengths[j]), lengths, batch_size, cost_limit)
    rng.shuffle(batches)
    return batches


def collate(examples, pad_id, device):
    width = max(len(ids) for ids, _ in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(lab)] = torch.tensor(lab)
        mask[i, :len(ids)] = 1
    return input_ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def validation_loss(model, data, batch_size, cost_limit, pad_id, device):
    """Mean loss per supervised token over the whole validation set."""
    model.eval()
    lengths = [len(ids) for ids, _ in data]
    order = sorted(range(len(data)), key=lambda j: lengths[j])
    total, count = 0.0, 0
    for idx in pack(order, lengths, batch_size, cost_limit):
        batch = [data[j] for j in idx]
        input_ids, mask, labels = collate(batch, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        n = int((labels != -100).sum())
        total += float(loss) * n
        count += n
    model.train()
    return total / max(count, 1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/collected/detection", help="folder with train.jsonl and val.jsonl")
    ap.add_argument("--out", default="models/collected_detection_lora")
    ap.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    ap.add_argument("--epochs", type=float, default=10, help="most epochs to train; the best one is kept")
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs in a row without a better validation loss; 0 trains every epoch")
    ap.add_argument("--batch-size", type=int, default=16, help="most rows in one batch")
    ap.add_argument("--batch-tokens", type=int, default=6144,
                    help="most padded tokens in one batch; 6144 peaks near 21 GB of GPU memory for the 1.3B model")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of steps spent warming up")
    ap.add_argument("--max-length", type=int, default=4096, help="longer conversations are skipped, never cut")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-targets", nargs="+", default=ATTENTION_QKV,
                    help="modules that get an adapter; 'all' adds the output and MLP projections")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only this many training rows (for a trial run)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    if args.lora_targets == ["all"]:
        args.lora_targets = ALL_PROJECTIONS

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    import transformers

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        sys.exit("No CUDA device found; this script expects a GPU.")
    device = "cuda"
    data_dir, out = ROOT / args.data, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_rows = read_jsonl(data_dir / "train.jsonl", args.limit)
    val_rows = read_jsonl(data_dir / "val.jsonl", args.limit and max(args.limit // 5, 20))
    n_tokens, n_answer = check_encoding(tok, train_rows[0]["messages"])
    print(f"encoding check passed on the first row: {n_tokens} tokens, {n_answer} of them supervised")

    def prepare(rows, name):
        data, skipped = [], 0
        for r in rows:
            ids, labels = encode(tok, r["messages"])
            if len(ids) > args.max_length:
                skipped += 1
                continue
            data.append((ids, labels))
        supervised = sum(sum(t != -100 for t in lab) for _, lab in data)
        total = sum(len(ids) for ids, _ in data)
        print(f"{name}: {len(data)} conversations ({skipped} skipped as longer than {args.max_length} tokens), "
              f"{total:,} tokens, {supervised / total:.1%} supervised")
        return data, skipped

    train, train_skipped = prepare(train_rows, "train")
    val, _ = prepare(val_rows, "val")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                              bias="none", task_type="CAUSAL_LM", target_modules=args.lora_targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,} of {sum(p.numel() for p in model.parameters()):,}")

    lengths = [len(ids) for ids, _ in train]
    cost_limit = args.batch_tokens
    rng = random.Random(args.seed)
    # Batch counts differ slightly from epoch to epoch (the packing depends on the shuffle); plan with the first.
    epochs = [make_batches(lengths, args.batch_size, cost_limit, rng) for _ in range(math.ceil(args.epochs))]
    if args.epochs % 1:
        epochs[-1] = epochs[-1][:max(1, int(len(epochs[-1]) * (args.epochs % 1)))]
    steps_per_epoch = len(epochs[0])
    total_steps = sum(len(e) for e in epochs)
    warmup_steps = max(1, int(total_steps * args.warmup))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup_steps if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))))

    settings = {**vars(args), "train_rows": len(train), "val_rows": len(val), "train_skipped": train_skipped,
                "steps": total_steps, "steps_first_epoch": steps_per_epoch, "trainable_parameters": trainable,
                "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                "gpu": torch.cuda.get_device_name(0)}
    (out / "run_settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    log = open(out / "train_log.jsonl", "w", encoding="utf-8")

    model.train()
    step, best, started, window, seen_tokens = 0, float("inf"), time.time(), [], 0
    history, stale = [], 0
    small = sum(len(b) < args.batch_size for b in epochs[0])
    print(f"training for {total_steps} steps ({steps_per_epoch} in the first epoch; {small} of those batches hold "
          f"fewer than {args.batch_size} rows because their rows are long)", flush=True)
    for epoch_no, batches in enumerate(epochs, 1):
        for batch_idx in batches:
            input_ids, mask, labels = collate([train[j] for j in batch_idx], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            # A short batch counts for less, so every row carries the same weight whatever batch it lands in.
            (loss * len(batch_idx) / args.batch_size).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window.append(loss.item())
            seen_tokens += int(mask.sum())
            if step % args.log_every == 0 or step == total_steps:
                elapsed = time.time() - started
                entry = {"step": step, "epoch": epoch_no, "loss": round(sum(window) / len(window), 4),
                         "lr": scheduler.get_last_lr()[0], "tokens_per_s": round(seen_tokens / elapsed),
                         "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                window = []
                log.write(json.dumps(entry) + "\n")
                log.flush()
                left = (total_steps - step) * elapsed / step
                print(f"step {step}/{total_steps}  epoch {epoch_no}  loss {entry['loss']:.4f}  lr {entry['lr']:.2e}  "
                      f"{entry['tokens_per_s']:,} tok/s  peak {entry['gpu_gb']} GB  about {left / 60:.0f} min left",
                      flush=True)
        # Same token budget as training: bigger batches pushed the GPU past its memory in the first run
        # and made each validation pass take about seven minutes.
        v = validation_loss(model, val, args.batch_size * 2, cost_limit, tok.pad_token_id, device)
        is_best = v < best
        history.append({"step": step, "epoch": epoch_no, "val_loss": round(v, 4)})
        log.write(json.dumps({**history[-1], "saved": is_best}) + "\n")
        log.flush()
        print(f"== epoch {epoch_no}: validation loss {v:.4f}{'  (best so far, adapter saved)' if is_best else ''}",
              flush=True)
        if is_best:
            best, stale = v, 0
            model.save_pretrained(out)
            tok.save_pretrained(out)
        else:
            stale += 1
            if args.patience and stale >= args.patience and epoch_no < len(epochs):
                print(f"stopping after epoch {epoch_no}: no better validation loss for {stale} epochs", flush=True)
                break
    log.close()
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    settings.update(validation=history, best_val_loss=round(best, 4), best_epoch=best_epoch,
                    epochs_run=len(history), steps_run=step, minutes=round((time.time() - started) / 60, 1),
                    peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    (out / "run_settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    print(f"done in {settings['minutes']} min; best validation loss {best:.4f} (epoch {best_epoch}); "
          f"peak GPU memory {settings['peak_gpu_gb']} GB; adapter in {out}")


if __name__ == "__main__":
    main()
