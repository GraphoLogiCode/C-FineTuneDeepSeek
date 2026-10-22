# Fine-tuning on the collected data

How the detection adapter is trained and scored on the dataset from [COLLECTION.md](COLLECTION.md), and what the runs produced.

## Running it

```bash
.venv\Scripts\python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements-train.txt

.venv\Scripts\python src/training/train_lora.py --out models/collected_detection_lora_qkv_r8
.venv\Scripts\python src/evaluation/evaluate_detection.py --name lora_qkv_r8 --adapter models/collected_detection_lora_qkv_r8
.venv\Scripts\python src/evaluation/evaluate_detection.py --name base_zero_shot
```

The first command is the PyTorch build for an RTX 50-series card; pick the one for your GPU. The base model (2.6 GB) is downloaded from Hugging Face on first use.

The trainer's defaults are the settings of the second run below. The first run is reproduced with:

```bash
.venv\Scripts\python src/training/train_lora.py --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --lora-targets all --lr 2e-4 --epochs 3 --out models/collected_detection_lora
```

The same trainer supports the fixing task. A restricted fixing pilot and its new scoring/validation scripts are described in [FIXING.md](FIXING.md); it uses a separate adapter and frozen subset rather than the full collected dataset.

## Training setup

Both runs share:

| Setting | Value | Why |
| --- | --- | --- |
| Base model | `deepseek-ai/deepseek-coder-1.3b-instruct`, bf16 | No quantisation needed on a 32 GB card |
| Prompt | The model's own chat template | The 2025 runs used an `[INST]` wrapper the model was not trained with |
| Loss | On the assistant's answer only | The answer is about 4% of the tokens; the 2025 runs spread the loss over everything |
| Truncation | None; a conversation over 4,096 tokens would be skipped | The 2025 runs cut the answer off |
| Batches | Up to 16 rows and 6,144 padded tokens | Memory grows with tokens per batch |
| Optimiser | AdamW, 3% warm-up, cosine decay, gradient clipping at 1.0 | |
| Model kept | The adapter from the epoch with the lowest validation loss | |

Prompt and answer are tokenised separately and joined, exactly as at inference, and the script checks on start that the supervised tokens decode to the answer followed by the end-of-turn token.

They differ in the LoRA and epoch settings:

| Setting | First run | Second run (settings of Zhang et al.) |
| --- | --- | --- |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 8 / 16 / 0.1 |
| Adapted modules | All seven projections (attention and MLP) | Query, key and value projections |
| Trainable parameters | 15.0M (1.1%) | 2.36M (0.17%) |
| Learning rate | 2e-4 | 3e-4 |
| Epochs | 3 | Up to 10; stops after 3 epochs without a better validation loss |

### Where the second run's settings come from

Zhang et al., *A Comprehensive Evaluation of Parameter-Efficient Fine-Tuning on Code Smell Detection* ([RELATED_WORK.md](RELATED_WORK.md)), fine-tune the same base model, among eight others, with four parameter-efficient methods. Their LoRA setup is rank 8, alpha 16, dropout 0.1 on the query, key and value projections, learning rate 3e-4 with AdamW, 10 epochs with the epoch of lowest validation loss used for testing, batch size 1 and seed 42. On this model that is 2.36M trainable parameters, the same count as here.

What they report that bears on these settings:

- **Rank.** They vary the rank over 8, 32, 64, 640, 1,280 and 2,560 for DeepSeek Coder 1.3B. The best rank differs from smell to smell and a higher rank does not reliably help.
- **Method.** For DeepSeek Coder 1.3B, LoRA is not their best method: (IA)³, with 0.28M parameters, scores higher on all four smells (MCC 45.0–72.6% against 40.5–67.7% for LoRA).
- **Batch size and seed** have little effect in their appendix.

Their task is not the same as this one, so the settings are a starting point and not a guarantee:

| | Zhang et al. | This project |
| --- | --- | --- |
| Output | One yes/no per smell, from a classification head | A JSON list of check names, generated as text |
| Labels | 4 smells, manually validated | 40 clang-tidy checks, not reviewed by a person |
| Language | Java | C++ |
| Training rows | 490 to 11,076 per smell, half of them positive | 28,050, 53% with at least one check |
| Input limit | 1,024 tokens, longer rows excluded | 4,096 tokens |
| Test data | A tenth of the same pool of samples | Two engines absent from training |

One addition to their recipe: training stops early when three epochs in a row fail to improve the validation loss. They always train all 10 epochs. The adapter kept is the same either way unless the loss recovers after three worse epochs.

## How it is scored

`evaluate_detection.py` generates an answer for every row (greedy decoding), parses the JSON, and reports:

- **valid**: the answer parses as `{"checks": [...]}`.
- **exact**: the predicted set of checks equals the reference set.
- **smelly accuracy / F1**: the binary question "does any check apply?".
- **micro precision / recall / F1**: over every (function, check) pair.
- **macro F1**: mean F1 over the checks that occur in the split.

Each number is also given for two baselines, and for the length-matched subset, where length carries no information:

- **Always clean**: answer `{"checks": []}` every time.
- **Length rule**: call a function smelly if it is longer than the threshold that works best on the training split, and name the most common check.

## Results of the first run

Run on 8 October 2026 on one RTX 5090, with rank 16 on all seven projections. Training took 73 minutes for 7,673 steps and peaked at 21.9 GB of allocated GPU memory.

**Training**

| Epoch | Training loss (end of epoch) | Validation loss |
| --- | --- | --- |
| 1 | 0.071 | 0.0801 |
| 2 | 0.039 | **0.0690** |
| 3 | 0.017 | 0.0711 |

The adapter from epoch 2 was kept. In epoch 3 the training loss keeps falling while the validation loss rises, so two epochs are enough at this learning rate.

**Test split** (3,940 functions from OGRE and Spring, two engines the model never saw in training)

| | Valid JSON | Exact match | Smelly-or-clean accuracy | Micro precision | Micro recall | Micro F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuned adapter | 100% | 72.6% | 85.2% | 0.827 | 0.714 | **0.766** | 0.489 |
| Untuned base model | 5.6% | 43.5% | 46.7% | 0.002 | 0.001 | 0.001 | 0.000 |
| Always clean | – | 45.8% | 45.8% | 0 | 0 | 0 | 0 |
| Length rule | – | 17.7% | 56.7% | 0.138 | 0.128 | 0.133 | 0.006 |

- **On the length-matched subset** (3,340 functions) the adapter scores 85.1% accuracy and 0.756 micro F1, while the length rule drops to 49.9%. The model is not using length.
- **Validation split** (Irrlicht, 2,443 functions): 85.5% accuracy, 0.766 micro F1.
- **By engine:** OGRE 0.775 micro F1, Spring 0.727.
- **Kind of error:** 17.0% of rows miss a check and add none, 6.9% add one and miss none, 3.5% do both. The model errs towards saying too little.
- **It never invents a check name**; every predicted check is in the 40-check vocabulary.
- **The untuned model** answers in prose and names checks outside the vocabulary, so it has no chance on this output format. It shows that the format is learned, not how hard the task is.

**By check** (test split; the 16 most frequent)

| Check | Functions | F1 |
| --- | --- | --- |
| `cppcoreguidelines-init-variables` | 448 | 0.938 |
| `readability-function-size` | 400 | 0.932 |
| `bugprone-narrowing-conversions` | 386 | 0.432 |
| `cppcoreguidelines-owning-memory` | 297 | 0.852 |
| `readability-else-after-return` | 257 | 0.981 |
| `misc-unused-parameters` | 194 | 0.935 |
| `readability-convert-member-functions-to-static` | 194 | 0.304 |
| `readability-function-cognitive-complexity` | 167 | 0.840 |
| `misc-no-recursion` | 128 | 0.633 |
| `cppcoreguidelines-prefer-member-initializer` | 120 | 0.845 |
| `readability-simplify-boolean-expr` | 66 | 0.862 |
| `bugprone-implicit-widening-of-multiplication-result` | 63 | 0.313 |
| `bugprone-switch-missing-default-case` | 50 | 0.448 |
| `readability-make-member-function-const` | 43 | 0.000 |
| `bugprone-assignment-in-if-condition` | 39 | 0.933 |
| `performance-unnecessary-value-param` | 34 | 0.625 |

The split is clear. Checks whose evidence is in the function text are learned well: an uninitialised variable, an `else` after a `return`, an unused parameter, a long or deeply nested body. Checks that depend on declarations elsewhere are not: whether a conversion narrows depends on types declared in a header, and whether a method could be `static` or `const` depends on the class. The model only sees the function.

Of the 39 checks that occur in the test split, 17 reach an F1 of 0.7 or more and 15 stay below 0.3; most of the latter have fewer than ten test examples, which is what pulls the macro F1 down to 0.49.

**What this does and does not show**

- It shows that a 1.3B model can learn to reproduce clang-tidy's findings for locally visible patterns, and that this carries over to engines it was not trained on.
- The reference is clang-tidy's output, not a human judgement (see [COLLECTION.md](COLLECTION.md#limits)). A high score means agreement with the tool.
- It is one run with one seed. Differences of a point or two are not meaningful.
- Training rows include the less certain "relaxed" level; test rows do not.

Full numbers and every prediction are in `results/collected_detection/<name>/<split>/`.

## Second run: interrupted after epoch 4

The run with the settings of Zhang et al. was started on 8 October 2026 and cut off in its fifth epoch when the session that launched it ended. It has not been repeated. The adapter saved after epoch 4 was scored as `lora_qkv_r8_epoch4_interrupted`, so the numbers below describe an unfinished run, not the settings at their best.

| Epoch | Validation loss |
| --- | --- |
| 1 | 0.1008 |
| 2 | 0.0836 |
| 3 | 0.0776 |
| 4 | 0.0721 |

The validation loss was still falling when the run stopped, and had not yet reached the first run's best (0.0690).

| | Exact match | Smelly-or-clean accuracy | Micro F1 | Macro F1 |
| --- | --- | --- | --- | --- |
| First run, test | 72.6% | 85.2% | 0.766 | 0.489 |
| Second run after epoch 4, test | 69.8% | 84.0% | 0.744 | 0.420 |
| First run, validation | 66.6% | 85.5% | 0.766 | 0.431 |
| Second run after epoch 4, validation | 66.8% | 86.0% | 0.761 | 0.420 |

- **On the test split** the unfinished adapter is behind by 0.022 micro F1 (95% interval from resampling the test rows: 0.011 to 0.033).
- **On the validation split** the two are level (difference 0.005, interval includes zero).
- **The gap is in the less frequent checks.** The eight most frequent ones score within 0.04 of the first run; `prefer-member-initializer`, `assignment-in-if-condition` and `unnecessary-value-param` drop by 0.15 to 0.28.
- **It used less memory and ran faster:** 16.9 GB at peak against 21.9 GB, and about 15,000 tokens per second against 12,000.

Whether the remaining epochs close the gap is not known. Training does not resume from a saved adapter, so finding out means running all epochs again (about two hours).
