"""Local interactive inference and study guide. Run from any working directory."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
from src.inference.detection_demo import DEFAULT_ADAPTER, Detector

TEACHING_CODE = "int Particle::lifetimeTicks()\n{\n    return lifetime;\n}"
TEACHING_CONTEXT = "class Particle {\npublic:\n    int lifetimeTicks();\nprivate:\n    double lifetime;\n};"

LESSONS = {
    "1. Label source files": """### 1. The teacher is clang-tidy
`scripts/collect/scan.py` runs clang-tidy on engine source files and records diagnostic names,
messages, byte offsets and analysis errors. These findings become reference labels.

The model is learning agreement with this tool. A diagnostic is not a human judgement of a game-specific smell.

**Think about it:** if a file failed to analyse, would an empty warning list prove the code is clean?
The pipeline records failures separately so it can exclude them.""",
    "2. Extract whole functions": """### 2. Keep the function and the findings inside it
`scripts/collect/extract.py` uses tree-sitter to find C++ function bodies. It attaches diagnostics
whose byte offsets lie within each function and filters macro-expansion findings.

Each unit contains `code`, `path`, function name, line range, findings and metrics.
**This is where context is currently lost:** a method body is retained, but its containing
class, member declarations, inherited types and declarations from headers are not included in the input.

Tree-sitter identifies syntax. Resolving a type alias or overloaded call reliably requires semantic analysis.""",
    "3. Build examples and splits": """### 3. Turn functions into input/answer conversations
`scripts/collect/build_dataset.py` filters labels, normalises aliases, removes duplicates and splits by engine.
Training uses seven engines; validation uses Irrlicht; testing uses OGRE and Spring.

Input: system instruction + one function. Target: `{"checks": ["..."]}`.
Clean and smelly examples are sampled to reduce the length shortcut.

The live demo never sends the reference checks or diagnostic messages to the model.
Those are displayed separately so you can inspect its prediction.""",
    "4. Train the adapter": """### 4. Learn the answer, not the instruction
`src/training/train_lora.py` renders the model's chat template and tokenises prompt and answer separately:

```python
input_ids = prompt_ids + answer_ids
labels = [-100] * len(prompt_ids) + answer_ids
```

`-100` masks prompt tokens out of the loss. LoRA trains small adapter matrices while the base weights stay frozen.
The saved rank-16 run trained for three epochs; the best validation checkpoint was from epoch two.
The demo loads that saved adapter and merges it into the base model in memory.""",
    "5. Run inference": """### 5. Generate an answer for an unseen input
Open **Run inference**, choose a held-out example and press **Run detection**.

1. Build the two input messages using `src/detection_prompt.py`.
2. Apply the tokenizer's chat template. Open **Exact model prompt** to inspect it.
3. Tokenise the prompt. No reference answer is included.
4. Call `model.generate(..., do_sample=False)` to choose tokens greedily.
5. Decode only the newly generated tokens and parse the check list.

These stages live in `src/inference/detection_demo.py`. The raw output is streamed below the input.
Inference computes predictions; it does not update the adapter or run clang-tidy on your pasted code.""",
    "6. Evaluate predictions": """### 6. Compare check sets on held-out engines
`src/evaluation/evaluate_detection.py` compares predicted checks with clang-tidy references.
Micro F1 aggregates all check decisions; macro F1 gives each check equal weight, exposing rare-label weaknesses.

The saved rank-16 test run reports 0.766 micro F1 and 0.489 macro F1 over 3,940 functions.
Its 85.2% binary accuracy asks only whether any check is present.

Invalid output is a parse failure, not reliable evidence that a function is clean. The demo keeps this distinction visible.
For edited or custom code, the demo cannot report accuracy because it has no matching reference.""",
}

CONTEXT_GUIDE = """### Why declarations can matter
For the teaching example, `return lifetime;` does not reveal the member's type. The declaration
`double lifetime;` reveals a floating-point value returned as `int`, making a narrowing conversion relevant.
The class also helps investigate whether the method could be `const`. These are teaching hypotheses;
the example has not been labelled by clang-tidy.

### Where to add context to the real pipeline
1. **Extraction:** extend each unit from `extract.py` with a context record: enclosing class declaration,
   relevant fields and base classes, resolved aliases, and referenced function signatures. Preserve file and line provenance.
2. **Resolution:** use tree-sitter for a first pass over local declarations. Use clangd/LSP or libclang for actual
   symbol/type resolution across headers. Supply the engine's real `compile_commands.json` so include paths,
   macros and language standards match its build. Mark unresolved symbols explicitly.
3. **Selection:** include declarations relevant to this function, not whole headers. Remove other method bodies.
   Use deterministic ordering and a context token budget; retain the entire target and answer.
4. **Dataset:** put context and target in separate labelled sections. Keep the target's check list unchanged.
   Never include clang-tidy diagnostics, reference checks, fixes or evaluation labels as context.
5. **Training and evaluation:** train a separate adapter on those augmented examples. Compare it with the original
   on the same target IDs and engine split, with several seeds and per-check scores, input token counts and latency.
   Audit duplicate/shared code and retrieved context across splits. Review a sample of tool labels manually.

### What you can try now
The context checkbox changes the prompt to include declarations you paste. The existing adapter was trained
without this format: an output change here is an exploratory observation, **not evidence that context improves accuracy**.
The baseline model, dataset and saved results are preserved. Automatic context extraction and retraining are future work.

Sources: [clangd architecture](https://clangd.llvm.org/design/code),
[Clang compilation database](https://clang.llvm.org/docs/JSONCompilationDatabase.html),
[C++ context-assisted repair study](https://arxiv.org/abs/2601.14936).
"""


def load_examples():
    path = ROOT / "data/collected/detection/test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = {}
    for title, check in [
        ("Held-out: clean function", None),
        ("Held-out: else after return", "readability-else-after-return"),
        ("Held-out: narrowing conversion", "bugprone-narrowing-conversions"),
        ("Held-out: method could be const", "readability-make-member-function-const"),
    ]:
        candidates = [r for r in rows if (not r["checks"] if check is None else check in r["checks"])]
        if candidates:
            selected[title] = min(candidates, key=lambda r: len(r["messages"][1]["content"]))
    selected["Teaching example: hidden member type"] = None
    return selected


def example_values(name, examples):
    row = examples[name]
    if row is None:
        return (TEACHING_CODE, TEACHING_CONTEXT, False,
                "Teaching example; no clang-tidy reference is available.",
                {"code": TEACHING_CODE, "checks": None})
    code = row["messages"][1]["content"].split("\n\n", 1)[1]
    info = (f"**{row['engine']} · {row['name']}** — `{row['path']}`:{row['start_line']}\n\n"
            f"clang-tidy reference: `{json.dumps({'checks': row['checks']})}`")
    return code, "", False, info, {"code": code, "checks": row["checks"]}


def reference_comparison(code, reference, parsed):
    if not reference or reference["checks"] is None or code != reference["code"]:
        return "No reference comparison is available for this input."
    if parsed is None:
        return "Output did not parse; it is not a confirmed clean result."
    gold, pred = set(reference["checks"]), set(parsed["checks"])
    return (f"Matches: {', '.join(sorted(gold & pred)) or 'none'}\n\n"
            f"Missed: {', '.join(sorted(gold - pred)) or 'none'}\n\n"
            f"Extra: {', '.join(sorted(pred - gold)) or 'none'}\n\n"
            f"Exact agreement with reference: {'yes' if gold == pred else 'no'}.")


def build_app(detector):
    examples = load_examples()
    first = "Held-out: else after return" if "Held-out: else after return" in examples else next(iter(examples))
    initial_code, _, _, initial_info, initial_ref = example_values(first, examples)
    with gr.Blocks(title="DeepSeek detection lab") as app:
        gr.Markdown("# DeepSeek detection lab\nInspect the current adapter, one C++ function at a time.")
        with gr.Tab("Run inference"):
            example = gr.Dropdown(list(examples), value=first, label="Example")
            source_info = gr.Markdown(initial_info)
            reference = gr.State(initial_ref)
            with gr.Row():
                code = gr.Code(value=initial_code, language="cpp", label="Target function", lines=13, interactive=True)
                with gr.Column():
                    context = gr.Code(language="cpp", label="Class and type declarations (optional)", lines=10, interactive=True)
                    use_context = gr.Checkbox(False, label="Include declarations — experimental")
                    gr.Markdown("The adapter was trained on functions alone. Context support has not been validated.")
            with gr.Row():
                run = gr.Button("Run detection", variant="primary")
                output_limit = gr.Slider(16, 384, value=192, step=16, label="Maximum output tokens")
            status = gr.Markdown("Ready. The first run loads the saved model into memory.")
            with gr.Row():
                raw = gr.Code(language="json", label="Raw generated response", lines=6, interactive=False)
                parsed = gr.JSON(label="Parsed checks")
            comparison = gr.Markdown()
            with gr.Accordion("Exact model prompt", open=False):
                prompt = gr.Textbox(label="Chat template after formatting", lines=14, interactive=False)
            result_outputs = [status, raw, parsed, prompt, comparison]

            def reset_result():
                return "Input changed. Run detection for this input.", "", None, "", ""

            example.change(lambda name: (*example_values(name, examples), *reset_result()), inputs=example,
                           outputs=[code, context, use_context, source_info, reference, *result_outputs])
            code.input(lambda: ("Edited input: the example's reference no longer necessarily applies.", *reset_result()),
                       outputs=[source_info, *result_outputs])
            context.input(reset_result, outputs=result_outputs)
            use_context.change(reset_result, outputs=result_outputs)

            def infer(code_text, declarations, include_context, limit, reference_value):
                yield "Loading model / preparing input…", "", None, "", ""
                try:
                    for trace in detector.stream(code_text, declarations or "", include_context, limit):
                        summary = (f"{'Complete' if trace['finished'] else 'Generating'} · {trace['device']} · "
                                   f"{trace['prompt_tokens']} input tokens · {trace['seconds']:.2f}s")
                        if trace["finished"]:
                            summary += f" · {trace['generated_tokens']} output tokens · JSON {'valid' if trace['valid'] else 'invalid'}"
                            if trace["unknown_checks"]:
                                summary += " · Unknown check names: " + ", ".join(trace["unknown_checks"])
                            if trace["stopped_at_limit"]:
                                summary += " · Output reached its token limit; it may be incomplete."
                        if include_context:
                            summary += " · Experimental context prompt"
                        match = reference_comparison(code_text, reference_value, trace["parsed"]) if trace["finished"] else ""
                        yield summary, trace["raw"], trace["parsed"], trace["prompt"], match
                except Exception as error:
                    yield f"Could not run detection: {error}", "", None, "", ""

            run.click(infer, [code, context, use_context, output_limit, reference],
                      [status, raw, parsed, prompt, comparison], concurrency_limit=1, api_name="detect")
        with gr.Tab("Study the pipeline"):
            gr.Markdown("Source → clang-tidy labels → extracted functions → conversations → LoRA training → inference → evaluation")
            lesson = gr.Dropdown(list(LESSONS), value=next(iter(LESSONS)), label="Study one step")
            explanation = gr.Markdown(next(iter(LESSONS.values())))
            lesson.change(lambda step: LESSONS[step], lesson, explanation)
        with gr.Tab("Add class and type context"):
            gr.Markdown(CONTEXT_GUIDE)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--warmup", action="store_true", help="load model before opening the app")
    args = parser.parse_args()
    detector = Detector(args.adapter, args.device)
    if args.warmup:
        detector.load()
    build_app(detector).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                                     inbrowser=False, show_error=True)


if __name__ == "__main__":
    main()
