"""Single-function inference using the collected-data adapter, with a visible trace.

The 2025 scripts beside this file use older adapters and prompts. This module uses
the same input format, decoding and JSON parser as the collected-data evaluator.
"""
import json
import threading
import time
from pathlib import Path

from src.detection_prompt import detection_messages

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = ROOT / "models/collected_detection_lora"


def check_budget(prompt_tokens, max_new_tokens, limit=4096):
    if prompt_tokens + max_new_tokens > limit:
        raise ValueError(
            f"Input has {prompt_tokens:,} tokens; reserving {max_new_tokens} output tokens "
            f"exceeds the {limit:,}-token demo limit. Shorten the declarations or function. "
            "Nothing was truncated."
        )


class Detector:
    def __init__(self, adapter=DEFAULT_ADAPTER, device="auto"):
        self.adapter = Path(adapter).resolve()
        self.device = device
        self.model = self.tokenizer = None
        self.lock = threading.Lock()
        labels = ROOT / "data/collected/detection/labels.json"
        self.vocab = set(json.loads(labels.read_text(encoding="utf-8"))["checks"])

    def load(self):
        """Lazy load local artifacts. No model downloads or weight changes."""
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = json.loads((self.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        print(f"Loading detector from {self.adapter}", flush=True)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(str(self.adapter), local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model_name_or_path"], dtype=dtype, local_files_only=True
        ).to(self.device)
        print(f"Base model loaded on {self.device}; loading and merging adapter", flush=True)
        model = PeftModel.from_pretrained(base, str(self.adapter), local_files_only=True).merge_and_unload()
        model.eval()
        self.model, self.tokenizer = model, tokenizer
        print("Detector ready", flush=True)

    def stream(self, code, context="", use_context=False, max_new_tokens=192):
        """Yield prompt, live raw text and final parsed result for the UI or CLI checks."""
        import torch
        from transformers import TextIteratorStreamer
        from src.evaluation.evaluate_detection import parse_answer

        messages = detection_messages(code, context, use_context)
        max_new_tokens = int(max_new_tokens)
        if not 16 <= max_new_tokens <= 384:
            raise ValueError("Output token budget must be between 16 and 384.")
        with self.lock:
            self.load()
            tok = self.tokenizer
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            n_prompt = enc["input_ids"].shape[1]
            check_budget(n_prompt, max_new_tokens)
            trace = {"prompt": prompt, "raw": "", "parsed": None, "valid": False,
                     "prompt_tokens": n_prompt, "generated_tokens": 0, "seconds": 0,
                     "device": self.device, "context_experiment": bool(use_context), "finished": False}
            yield dict(trace)
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=120)
            errors, outputs = [], []

            def generate():
                try:
                    with torch.inference_mode():
                        outputs.append(self.model.generate(
                            **enc, max_new_tokens=max_new_tokens, do_sample=False,
                            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                            streamer=streamer,
                        ))
                except Exception as error:
                    errors.append(error)
                    streamer.end()

            started = time.perf_counter()
            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    trace["raw"] += piece
                    trace["seconds"] = round(time.perf_counter() - started, 3)
                    yield dict(trace)
            finally:
                worker.join()
            if errors:
                raise RuntimeError(f"Generation failed: {errors[0]}") from errors[0]
            output_ids = outputs[0][0, n_prompt:]
            raw = tok.decode(output_ids, skip_special_tokens=True).strip()
            checks, valid = parse_answer(raw)
            trace.update(raw=raw, valid=valid, parsed={"checks": sorted(checks)} if valid else None,
                         unknown_checks=sorted(checks - self.vocab),
                         generated_tokens=len(output_ids),
                         stopped_at_limit=len(output_ids) >= max_new_tokens and output_ids[-1].item() != tok.eos_token_id,
                         seconds=round(time.perf_counter() - started, 3), finished=True)
            yield dict(trace)
