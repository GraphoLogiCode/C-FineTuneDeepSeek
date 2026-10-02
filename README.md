# C-FineTuneDeepSeek

Research question: can a small code model learn to recognise industry-specific code patterns? The test case is code smells in C++ game engines.

The project fine-tunes [`deepseek-ai/deepseek-coder-1.3b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) with QLoRA into two adapters:

- **Detection** — given a C++ snippet, output the smells found as JSON.
- **Fixing** — given a snippet (and detected smells), output refactored code and explanations as JSON.

## Status

The work was done in July 2025 and paused. It is **not in a working state**: an audit in October 2026 found that the adapters did not learn the intended task and that no evaluation produced a valid score. The causes are in the dataset and the training setup, and are documented with measurements in [docs/AUDIT.md](docs/AUDIT.md). Read that before reviewing the code.

## Repository layout

```
notebooks/
  DeepSeek_Coder_1.3B_FineTuned.ipynb   main notebook: data prep, training, export, evaluation
  scratch/                              early experiments (LoRA smoke test, NPU check, saved cells)
src/
  inference/                            three stand-alone inference scripts for the adapters
  evaluation/advanced_evaluation.py     metric code (detection P/R/F1, fixing BLEU/ROUGE)
  testing/simple_test.py                quick load-and-generate check
scripts/
  audit/                                scripts behind the numbers in docs/AUDIT.md
  plot_architecture.py                  draws the pipeline diagram
  test_npu.py                           OpenVINO device check
docs/
  AUDIT.md                              findings and recommended next steps
  DeepSeek_Coder_1.3B_FineTuned.html    static export of the main notebook
results/
  figures/                              loss curves and sample outputs
  evaluation_csvs/                      per-example outputs from early evaluation runs
  inference_results/                    metrics from the last evaluation run (all zero, see audit)
requirements.txt
```

## What is not in this repository

These are kept locally and ignored by git because of their size:

| Path | Size | Contents |
| --- | --- | --- |
| `data/processed/{detection,fixing}/{train,val}` | 0.5 GB | Tokenizer-ready datasets (16,720 train / 4,180 val rows per task) |
| `models/` | 14.7 GB | LoRA adapters, checkpoints, merged models and OpenVINO exports |

The raw JSONL files the datasets were built from are no longer available. They were generated with clang-tidy and tree-sitter from ten open-source engines: Cafu, cocos2d-x, Defold, Esoterica, Godot, Irrlicht, Nebula, OGRE, Spring and Torque.

Local model folders:

| Folder | Run |
| --- | --- |
| `models/detection`, `models/fixing` | Full dataset, r=32, 3 epochs, `max_length=1024` (19–20 July 2025) |
| `models/detection_qlora`, `models/fixing_qlora` | 1,000-sample subset, r=16, 5 epochs, `max_length=2048` (22 July 2025) |
| `models/*_merged`, `models/*_full_merged` | Adapters merged into the base model |
| `models/*_ov` | OpenVINO exports |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
# install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

`requirements.txt` was reconstructed from the notebook imports and the original environment; it has not been install-tested as a whole.

## Running

The scripts use paths relative to the repository root and expect the adapters under `models/`.

```bash
python src/inference/simple_inference.py
```

To reproduce the audit numbers (needs `data/processed/`):

```bash
python scripts/audit/audit_lengths_and_leakage.py
python scripts/audit/audit_labels.py
python scripts/audit/read_training_args.py
```

## Known issues

Summarised from [docs/AUDIT.md](docs/AUDIT.md):

- The answer was truncated out of a large share of training samples.
- The loss was computed over the whole prompt, of which the answer is a small fraction.
- Detection labels are six clang-tidy-derived buckets, not the 13 game smells the prompt describes.
- The fixing dataset contains no fixed code.
- The saved evaluation metrics were computed over zero samples.
- The inference scripts use different prompts from the ones used in training.
