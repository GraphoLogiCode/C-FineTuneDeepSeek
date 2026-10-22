"""Local learned repair inference. Generated code is a candidate, never an accepted fix by itself."""
import difflib
import json
import re
import threading
import time
from pathlib import Path

from src.fixing_prompt import fixing_messages

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = ROOT / "models/collected_fixing_pilot_r16"


def candidate_code(raw):
    """Extract one complete fenced block even with prose, fairly for all systems.

    Multiple blocks remain unparsed rather than selecting whichever looks best.
    Completeness/truncation and compilation are separate acceptance gates.
    """
    text = raw.strip()
    blocks = re.findall(r"^```(?:cpp|c\+\+|c)?[ \t]*\r?\n(.*?)^```[ \t]*$", text, re.S | re.I | re.M)
    fences = re.findall(r"^```", text, re.M)
    return blocks[0].strip() if len(blocks) == 1 and len(fences) == 2 else text


def syntax_check(code):
    import tree_sitter_cpp
    from tree_sitter import Language, Parser
    root = Parser(Language(tree_sitter_cpp.language())).parse(code.encode("utf-8")).root_node
    definitions = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            definitions.append(node)
        stack.extend(node.children)
    return {"parse_ok": not root.has_error, "function_definitions": len(definitions),
            "single_function": len(definitions) == 1,
            "note": "Syntax parsing does not establish compilation or behavioral correctness."}


def repair_diff(before, after):
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        fromfile="before.cpp", tofile="candidate.cpp"))


class Fixer:
    def __init__(self, adapter=DEFAULT_ADAPTER):
        self.adapter = Path(adapter).resolve()
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def supported_checks(self):
        path = self.adapter / "trained_checks.json"
        return set(json.loads(path.read_text(encoding="utf-8"))["checks"]) if path.exists() else None

    def load(self):
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        settings = json.loads((self.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter), local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(settings["base_model_name_or_path"],
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    local_files_only=True).to(self.device)
        self.model = PeftModel.from_pretrained(base, str(self.adapter), local_files_only=True).merge_and_unload().eval()

    def repair(self, code, checks, max_new_tokens=1536):
        import torch
        messages = fixing_messages(code, checks)
        supported = self.supported_checks()
        unsupported = sorted(set(checks) - supported) if supported is not None else []
        if unsupported:
            raise ValueError("No repair training support for these checks: " + ", ".join(unsupported))
        with self.lock:
            self.load()
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
            n_input = encoded["input_ids"].shape[1]
            if n_input + max_new_tokens > 4096:
                raise ValueError("Input plus reserved answer space exceeds 4096 tokens; nothing was truncated.")
            started = time.perf_counter()
            with torch.inference_mode():
                output = self.model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
            ids = output[0, n_input:]
            raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            candidate = candidate_code(raw)
            return {"candidate": candidate, "raw": raw, "prompt": prompt,
                    "requested_checks": sorted(set(checks)), "input_tokens": n_input,
                    "output_tokens": len(ids), "seconds": round(time.perf_counter()-started, 3),
                    "truncated": len(ids) >= max_new_tokens and ids[-1].item() != self.tokenizer.eos_token_id,
                    "changed": code.strip() != candidate.strip(), "syntax": syntax_check(candidate),
                    "diff": repair_diff(code, candidate), "behavioral_correctness": "not_tested"}
