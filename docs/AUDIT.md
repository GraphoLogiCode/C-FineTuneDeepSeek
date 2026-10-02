# Audit — October 2026

This is a review of the July 2025 work, made before resuming it. It covers the notebooks, the scripts, the processed datasets and the saved checkpoints.

**Conclusion.** The two adapters almost certainly did not learn the intended task, and no evaluation produced a valid score. The causes are in the data and the training setup, so the next step is to rebuild those, not to tune hyperparameters.

**Scope.** The numbers below were measured from `data/processed/` and from the saved checkpoints with the scripts in `scripts/audit/`. The models were not re-run, so statements about what they learned are inferred from how they were trained and from the outputs saved in `results/`.

## Findings

### 1. The answer was truncated out of training

Every sample begins with the same system prompt: 877 tokens for detection, 1,057 for fixing. The answer comes last, and `SFTTrainer` truncates from the right.

| `max_length` | Detection: answer fully cut | Fixing: answer fully cut |
| --- | --- | --- |
| 1024 | 84.6% | 100% |
| 2048 | 44.2% | 46.4% |
| 4096 | 0.1% | 0.0% |

- The subset runs (`models/*_qlora`) used 2048.
- The full-data runs (`models/detection`, `models/fixing`) used 1024. Their fixing eval loss of 0.0014 is consistent with training on the system prompt alone.

The 1024 row is computed on the datasets as they are on disk (written 22 July 2025). The full-data runs are two days older and may have used a slightly different prompt.

### 2. The loss covered the whole text

`completion_only_loss` was not set, so the prompt and the input code were supervised along with the answer. The answer is 1.6% of the tokens in a detection sample and 13.6% in a fixing sample. The loss curves in `results/figures/` therefore mostly measure how well the model reproduces the prompt and the code.

### 3. Detection labels do not match the task in the prompt

The system prompt lists 13 game-specific smells and asks for objects with `type`, `severity` and `explanation`. The data has neither.

- The label vocabulary is six strings: `Maintainability Smells`, `Medium-High Severity (60-79%)`, `Medium Severity (40-59%)`, `High Severity (80%+)`, `Performance Smells`, `Other`. None of the 13 smell names occurs.
- The `explanation` field holds clang-tidy check names. The most frequent are `readability-identifier-length`, `bugprone-reserved-identifier` and `cppcoreguidelines-macro-usage`.
- Two schemas are mixed: full-snippet samples use `{smells, severity, explanation}` and gap samples use `{smells, location}`.

A model trained perfectly on this data would predict a clang-tidy category, not a game-engine smell.

### 4. The fixing dataset contains no fixes

None of the 16,720 fixing targets has a `fixed_code` field.

| Target | Share |
| --- | --- |
| Gap sample → the original middle of the code (plain fill-in-the-middle) | 44.2% |
| Full snippet → raw code, usually a near copy of the input (median similarity 0.93 on a 3,000-row sample) | 43.8% |
| Detection-style label JSON | 11.9% |

### 5. No evaluation produced a valid score

- The three files in `results/inference_results/` report zero for every metric, over zero samples: each batch of the evaluation loop processed nothing.
- `src/evaluation/advanced_evaluation.py` scores predictions against the 13 smell names. Because those never occur in the references, it would report zero regardless of the model.
- The earlier CSVs in `results/evaluation_csvs/` show the schema mismatch directly: references such as `{"smells": ["High Severity (80%+)"], "location": "middle"}` against predictions of `{"smells": []}`.

### 6. Inference does not match training

- The scripts in `src/inference/` and `src/testing/` each use their own short system prompt, not the training prompt.
- Three of the four sample at temperature 0.3–0.7.
- Training used an `[INST] … [/INST]` wrapper. The base model's own template is `### Instruction:` / `### Response:`.

The saved sample outputs agree with this: detection returns "Single Responsibility Principle", which is in neither label set, and fixing returns the input code unchanged.

### 7. The validation split leaks and there is no test set

The split is random by row. About 8% of validation samples share most of their code with training samples (7.5% for detection, 8.6% for fixing), and vendored third-party code (asio, FreeType, Bullet) appears across engines. No engine was held out.

### 8. Smaller points

- `models/fixing_qlora/training_args.bin` records its output directory as `./models/detection_qlora`, so the files alone do not confirm which dataset that adapter was trained on.
- LoRA targeted `q/k/v/o_proj`, `up_proj` and `down_proj` but not `gate_proj`.
- Validation loss in the subset runs was lowest at epoch 2 of 5 and rose afterwards.
- The four inference scripts are near duplicates of each other.
- The notebook has hard-coded absolute paths (`C:\Users\...`, `D:\dev\...`).

## Recommended next steps

1. **Choose one task and drop fixing for now.** Either train a clang-tidy-category detector, which the existing labels support, or create real game-smell labels. Fixing needs genuine before/after pairs.
2. **Recover the raw pairs.** The raw JSONL is gone; the instruction/output pairs can be extracted from `data/processed/` by stripping the system prompt.
3. **Build the evaluation first.** Run it on the untuned base model and on a majority-class baseline, so fine-tuning has a number to beat.
4. **Fix the training setup.**
   - Cut the system prompt to about 100 tokens.
   - Use the model's own chat template.
   - Compute the loss on the answer only.
   - Truncate the code, never the answer.
   - Split by engine and hold one or two engines out as a test set.
5. **Reconsider the model size.** The 1.3B model, 4-bit loading and batch size 1 were chosen for an 8 GB laptop GPU. A 7B-class code model fits comfortably on a 24–32 GB card.

## Reproducing the numbers

```bash
python scripts/audit/audit_lengths_and_leakage.py   # findings 1, 2, 4, 7
python scripts/audit/audit_labels.py                # findings 3, 4
python scripts/audit/read_training_args.py          # findings 1, 8
```

The first two need `data/processed/`; the third needs `models/`. Neither folder is in git.
