# Detection lab: run inference and study the pipeline

This local Gradio app runs the October collected-data detection adapter. It shows
the formatted prompt, live generated text, parsed check names, token counts and
generation time. It does not use the 2025 adapters in the old inference scripts.

## Run

From the repository root, using the environment prepared for this demo:

```powershell
.venv-demo\Scripts\python.exe demo\detection_app.py --warmup
```

Open <http://127.0.0.1:7860>. The server binds to this PC only and does not create a
public share. Stop it with Ctrl+C in its terminal. The default adapter is
`models/collected_detection_lora`; use `--adapter PATH` for another compatible
adapter, `--port 7861` for another port, or `--device cpu` if CUDA is unavailable.

The prepared environment inherits the PC's installed CUDA PyTorch; its other demo
dependencies are local to `.venv-demo`. For a fresh independent environment:

```powershell
python -m venv .venv-demo
.venv-demo\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-demo\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Choose the PyTorch build for your GPU. The app loads only local model artifacts:
the adapter, the cached `deepseek-ai/deepseek-coder-1.3b-instruct` base model, and
`data/collected/detection/{test.jsonl,labels.json}` must be present. It does not
download weights. A fresh machine must obtain the base model before launching.

## First exercise: watch one prediction

1. Keep **Held-out: else after return** selected. Read the target function and its
   clang-tidy reference. The reference is displayed separately and never enters
   the model prompt.
2. Leave the context checkbox off and press **Run detection**. The raw JSON is
   generated token by token; the model does not execute or compile the C++.
3. Open **Exact model prompt**. Observe the system instruction, user function,
   and the empty response boundary. The reference answer is absent.
4. Compare the generated check set with the reference. An empty parsed list is
   a model prediction of no checks; invalid text is a parse failure. Neither is
   a guarantee that the code is correct.
5. Edit the function and run again. There is no reference score for edited code.

Use **Study the pipeline** to advance through six short lessons with code paths
and small understanding checks.

## Follow one example through the code

| Stage | Code | Input → output |
| --- | --- | --- |
| Scan | `scripts/collect/scan.py` | Engine files → clang-tidy diagnostics and analysis status |
| Extract | `scripts/collect/extract.py:extract_file` | Syntax tree and diagnostic offsets → function units with labels |
| Build | `scripts/collect/build_dataset.py:build` | Filtered, deduplicated functions → engine-separated chat examples |
| Train | `src/training/train_lora.py:encode` | Prompt and answer → input tokens and answer-only loss labels |
| Infer | `src/inference/detection_demo.py:Detector.stream` | Prompt tokens → generated tokens → parsed check set |
| Evaluate | `src/evaluation/evaluate_detection.py:score` | Predicted and reference check sets → precision, recall and F1 |

Training stores a system/user/assistant conversation. Inference receives only
system/user messages. The input format is shared in `src/detection_prompt.py`;
moving the existing constants there does not change the collected-data prompt.
Greedy decoding and the output parser match the existing evaluator. The
4,096-token demo budget includes room for the generated answer; oversized input
is rejected instead of silently truncated. No training weights or datasets are
changed by using the app.

## Second exercise: reveal a hidden type

Select **Teaching example: hidden member type**:

```cpp
int Particle::lifetimeTicks()
{
    return lifetime;
}
```

The function alone does not identify the member's type. Add its declaration:

```cpp
class Particle {
public:
    int lifetimeTicks();
private:
    double lifetime;
};
```

Now a floating-point value is returned as an integer, so a narrowing conversion
is relevant. Class information also helps investigate const-correctness. This
synthetic example has no measured clang-tidy reference. Run with the checkbox
off, then on, and inspect how the prompt and output change.

The current adapter was trained on function-only inputs. This toggle is an
exploratory prompt experiment; changed output is not evidence of improvement.

## Add context to a future training pipeline

1. Extend extracted units with an optional context record: enclosing class,
   relevant member declarations, base classes, resolved aliases and referenced
   function signatures. Record where each declaration came from and whether
   resolution succeeded. Never guess an unresolved type.
2. Start with tree-sitter for nearby syntax. For headers, aliases, templates and
   overloads, use clangd/LSP or libclang with the engine's actual compilation
   database. The present scan approximates the build environment; real compile
   commands would make semantic context more reliable.
3. Retrieve only declarations relevant to the target. Remove other method
   bodies, order context deterministically, and allocate a separate token budget
   to context. Keep the entire target and answer intact.
4. Build a separate contextual dataset with the same target IDs, reference
   labels and engine splits as the baseline. Keep context and target sections
   explicit, as the demo does. Keep diagnostics, labels and fixed code out of
   input context. Audit shared source and retrieved context across splits.
5. Train a separate adapter, then compare function-only input, class context,
   type/signature context and both. Use the same evaluation targets, multiple
   seeds and per-check F1, alongside token counts, latency and manual review of
   a sample of tool labels. Make tuning decisions on validation data; use a new
   untouched final test if repeated inspection of the current test influences
   those decisions.

The existing test results suggest focusing on narrowing conversions (F1 0.432),
methods that could be static (0.304), and methods that could be const (0.000).
Those observations motivate the experiment; they do not establish the cause of
the errors. Context retrieval, contextual training and improved accuracy have
not been implemented or established by this demo.

References: [clangd code architecture](https://clangd.llvm.org/design/code),
[Clang compilation database format](https://clang.llvm.org/docs/JSONCompilationDatabase.html),
[C++ repair using LSP context](https://arxiv.org/abs/2601.14936),
and [Transformers generation](https://huggingface.co/docs/transformers/v5.19.0/en/main_classes/text_generation).

## Check the input contracts

```powershell
.venv-demo\Scripts\python.exe -m unittest discover -s tests -v
```

These checks cover compatibility with saved training inputs, keeping the answer
out of inference, opt-in context, context limits, parse failures and edited-code
reference handling. Live inference and browser checks are separate verification.
