# Clean detection dataset

`data/clean/detection/` is the fine-tuning set rebuilt from the July 2025 processed data by [`scripts/build_clean_dataset.py`](../scripts/build_clean_dataset.py). The folder is not in git; run the script to recreate it.

**What it is good for.** The files are structurally sound: one schema, no duplicates, no overlap between splits, and every conversation fits the context window with its answer intact.

**What it is not.** The labels are weak. They are clang-tidy categories rather than game-specific smells, about half of the named checks point at code that is not in the snippet, and snippet length alone predicts the label 90% of the time. See [Label quality](#label-quality) before training on it.

There is no clean fixing dataset. The source data contains no real fixes; see [What was dropped](#what-was-dropped).

## Files

| File | Rows | Smelly | Clean |
| --- | --- | --- | --- |
| `train.jsonl` | 9,302 | 4,642 | 4,660 |
| `val.jsonl` | 1,163 | 536 | 627 |
| `test.jsonl` | 1,163 | 591 | 572 |
| `val_length_matched.jsonl` | 268 | 134 | 134 |
| `test_length_matched.jsonl` | 302 | 151 | 151 |

`stats.json` holds the full counts from the last build.

The two `length_matched` files are subsets of `val` and `test` with equal numbers of smelly and clean rows inside each length bin. Report results on them as well as on the full splits, because they cannot be solved by length.

## Record format

One JSON object per line:

| Field | Meaning |
| --- | --- |
| `id` | Stable id derived from the code |
| `messages` | `system` / `user` / `assistant` conversation, ready for a chat template |
| `smells` | Categories, most severe first; empty for clean code |
| `severity` | 0, 48, 53, 63 or 83, as in the source labels |
| `checks` | Sorted clang-tidy check names; empty for clean code |
| `n_tokens` | Length of the whole conversation under the DeepSeek Coder chat template |
| `group` | Rows that share code have the same group and are always in the same split |
| `source_file` | Which legacy file the row came from (`detection` or `fixing`) |

The assistant message is the training target:

```json
{"smells": ["Medium-High Severity (60-79%)"], "severity": 63, "checks": ["cppcoreguidelines-macro-to-enum"]}
```

Categories are `High Severity (80%+)`, `Medium-High Severity (60-79%)`, `Medium Severity (40-59%)`, `Maintainability Smells` and `Performance Smells`. There are 122 distinct check names.

## Using it for fine-tuning

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": "data/clean/detection/train.jsonl",
    "validation": "data/clean/detection/val.jsonl",
}).select_columns(["messages"])
```

Settings that matter, each tied to a problem found in the original runs:

- **`max_length=4096`.** The longest conversation is 3,269 tokens. At 2,048 only 91.8% of smelly rows fit (99.9% of clean rows), and at 1,024 only 10.1%. To train with a smaller window, rebuild with `--max-tokens` so long rows are dropped instead of truncated.
- **Answer-only loss.** Set `assistant_only_loss=True` in `SFTConfig` (present in trl 0.19.1). The answer is about 3% of the tokens in a conversation.
- **Use the model's chat template.** Pass the `messages` column as is; do not wrap it in `[INST]`.
- **Use the same system prompt at inference.** It is the `system` message of every row, and `SYSTEM_PROMPT` in the build script.

This has been checked at the data level only: the files load, the chat template renders them, and token counts are verified. No training run has been made on this dataset.

## How it was built

| Step | Rows |
| --- | --- |
| Full-snippet detection rows in the legacy files (10,467 + 1,276 found in the fixing file) | 11,743 |
| Dropped: category `Other` | −11 |
| Dropped: exact duplicates, ignoring whitespace | −102 |
| Dropped: duplicates with conflicting labels | −2 |
| **Kept** | **11,628** |

Changes made to the kept rows:

- **System prompt replaced.** Every row had an 877- or 1,057-token prompt baked in, describing 13 game smells that the labels do not contain. It was replaced by a 166-token prompt that describes the labels the data actually has.
- **Labels normalised.** `smells` was a single comma-joined string; it is now a list of categories. The check names, previously a comma-joined string under `explanation`, are now a list under `checks`.
- **Glued preprocessor directives repaired** in 964 rows. The legacy builder dropped a newline where it joined chunks, producing lines such as `#pragma once#include "x.h"`. A newline was restored only where a directive was attached directly to the previous token.
- **Split by shared code.** Rows that share any six consecutive lines of code form a group (9,974 groups, largest 335 rows), and a group never spans two splits. Splits are 80/10/10.

## What was dropped

| Legacy rows | Count | Why |
| --- | --- | --- |
| Detection, gap | 11,657 | The input is the code before and after a gap, and the label describes the missing middle. The model is asked about code it cannot see. |
| Fixing, full snippet | 9,218 | No fixes. In 50.4% the target is the input unchanged plus the comment "No fixes needed, code is clean". In 49.3% the target is unrelated code from a different file plus "Fixed … issue". |
| Fixing, gap | 9,182 | Fill-in-the-middle targets with no fix. About half have an empty suffix (plain completion). Where a context could be matched to a smelly detection row (735 rows), the target shares no more identifiers with it than a random pairing does. |

## Label quality

Measured on the clean set with [`scripts/audit/audit_clean_dataset.py`](../scripts/audit/audit_clean_dataset.py).

**Length predicts the label.** The rule "smelly if the conversation has at least 1,031 tokens" scores 89.6% accuracy on `test`. On `test_length_matched` it scores 61.9%.

| Conversation length (tokens) | Rows | Smelly |
| --- | --- | --- |
| up to 393 | 2,316 | 1.6% |
| 393–636 | 2,328 | 7.8% |
| 636–1,429 | 2,326 | 46.1% |
| 1,429–1,802 | 2,320 | 94.0% |
| over 1,802 | 2,338 | 98.3% |

**Named checks often refer to code outside the snippet.** For checks that report a visible construct, the construct is frequently absent:

| Check | Rows | Construct in the snippet |
| --- | --- | --- |
| `cppcoreguidelines-macro-usage` | 669 | 46.8% contain a `#define` |
| `cppcoreguidelines-macro-to-enum` | 202 | 46.5% contain a `#define` |
| `performance-enum-size` | 238 | 21.8% contain an `enum` |
| `bugprone-reserved-identifier` | 1,169 | 63.1% contain a reserved-style identifier |

The labels are still related to the code: the same constructs appear in about 1% of clean rows (9% for reserved identifiers). The likely cause is that diagnostics were attached at file level or with shifted line numbers.

**Categories are not a fixed function of the check.** Among rows with one check and one category, 78.4% carry that check's usual category.

**Other limits.**

- Source files are unknown, so two non-overlapping chunks of the same file can sit in different splits.
- Snippets include vendored third-party code (zstd, FreeType, Bullet, wxWidgets and others), not only engine code.
- Three in four smelly rows name a single check, which suggests the check list is incomplete.

## Rebuilding

```bash
python scripts/build_clean_dataset.py
```

The script rebuilds from `data/processed/`, prints the statistics, and then verifies the output; it exits non-zero if any check fails. The checks cover the schema, duplicate snippets, code shared between splits, token counts and the context limit.
