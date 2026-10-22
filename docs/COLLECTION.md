# Collecting labelled functions from engine source

`scripts/collect/` is a command-line pipeline that turns engine source code into two datasets:

- **Detection:** a C++ function, and the clang-tidy checks it triggers.
- **Fixing:** a C++ function, and the same function with clang-tidy's own fixes applied.

It replaces the GUI labelling tool used in 2025, whose output had the problems listed in [AUDIT.md](AUDIT.md). The section [What changed from the old tool](#what-changed-from-the-old-tool) maps each problem to what the pipeline does instead.

The labels are clang-tidy checks. The pipeline does not map them to game-specific smells; see [Limits](#limits).

## Running it

One-time setup, in the repository folder:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-collect.txt
```

This installs clang-tidy itself, so no LLVM installation or engine build is needed. The engine sources are expected under the `engines_root` folder named in [`collect_config.json`](../scripts/collect/collect_config.json).

Then:

```bash
.venv\Scripts\python scripts/collect/run_pipeline.py
```

The three steps can also be run one at a time, for all engines or for named ones:

```bash
.venv\Scripts\python scripts/collect/scan.py godot ogre
.venv\Scripts\python scripts/collect/extract.py godot ogre
.venv\Scripts\python scripts/collect/build_dataset.py
```

The scan is the slow step. It can be stopped and restarted: files already scanned with the same settings are skipped. Changing which checks count as labels needs only the last step.

## What each step does

| Step | Script | Reads | Writes |
| --- | --- | --- | --- |
| 1. Scan | `scan.py` | engine source | `data/collect/<engine>/scan.jsonl`: per file, a status and every diagnostic |
| 2. Extract | `extract.py` | `scan.jsonl`, engine source | `data/collect/<engine>/units.jsonl`: per function, its code and findings |
| 3. Build | `build_dataset.py` | `units.jsonl` | `data/collected/detection/`, `data/collected/fixing/`, `stats.json` |

**Scan.** Runs clang-tidy on every engine file, one process per file, in parallel. Vendored libraries, tests, samples and tools are skipped (see `exclude_dir_names` and each engine's `exclude` list in the config).

**Extract.** Parses each file with tree-sitter, takes every function definition that has a body, and attaches the diagnostics located inside it.

**Build.** Applies the label policy, removes duplicates, splits by engine, samples, writes the chat-format files and verifies them.

## What changed from the old tool

| Problem in the old tool | What the pipeline does |
| --- | --- |
| Files with the same name were mixed up | Every record is keyed by its path relative to the engine root |
| Warnings from headers were recorded against the file being scanned | Each file is analysed as its own main file, and only diagnostics located in that file are kept |
| Console output was scraped with a regular expression | Diagnostics come from clang-tidy's structured export, with byte offsets |
| A timeout or a failed compile looked the same as "no warnings" | Every file has a status; only files with status `ok` are used |
| Anything without a recorded warning counted as clean | A function is used only if clang-tidy demonstrably analysed it (see below) and reported no compile error inside it |
| Smelly code was a padded window, clean code a whole function | Both are whole functions, cut by the same code |
| One warning per snippet, though the snippet held many | A function's label is every finding inside it |
| The warning line was located by text search | Findings are located by byte offset |
| Check-to-smell mapping with no basis, and alias-dependent | Labels are the check names; known aliases are folded into one name |
| Random train/validation split by row | Split by engine |
| "Fixed" targets were unrelated code | Fixing targets are the input with clang-tidy's own edits applied |
| Results could not be reproduced | Fixed seed, pinned tool versions, engine commits recorded in `stats.json` |

### How a function is known to be analysed

clang-tidy skips code in inactive `#if` branches and can silently stop analysing after certain errors. To tell "no findings" from "not analysed", the scan sets the thresholds of `readability-function-size` to zero. clang-tidy then reports that check for every function it actually processed, with its line, statement, branch and parameter counts. A function without that report is dropped.

### Findings that come from macros

Engines wrap error handling in macros. Godot's `ERR_FAIL_COND(x)` expands to an `if … else`, so clang-tidy reports "do not use else after return" at every use, although the developer wrote neither. On the first full run, 83% of that check's findings in four engines were of this kind.

- A finding reported at the invocation of a function-like macro defined in the engine is ignored. A finding inside a macro's arguments, which the developer did write, is kept.
- Cognitive complexity is measured with macro-expanded code left out.
- The size label uses only what is visible in the source: lines, parameters and nesting depth. Statement and branch counts are recorded in `metrics` but include expanded code.

### Analysing without building the engine

No engine is compiled. Instead:

- **Include paths** are derived from the engine's own `#include` lines: for each include that does not resolve next to the including file, the folder that would make it resolve gets a vote. Folders voted for by engine code come first; folders that only vendored libraries ask for come after them. Folders that contain a file named like a system header (`math.h`, `signal.h`) are searched after the system headers so they cannot shadow them.
- **Platform macros** `WIN32` and `_WINDOWS` are defined, as CMake and Visual Studio projects do by default on Windows.
- **Missing headers** (platform SDKs, libraries not in the repository) are replaced by empty stub files, searched last. Without a stub, a missing header is a fatal error and clang stops reporting; with one, whatever the header should have declared shows up as ordinary errors where it is used.
- **Export macros** such as `_OgreExport` in `class _OgreExport Root` normally come from a header the build generates. Any macro used in that position at least three times is defined as nothing.
- **The language standard** is chosen per engine by trying each candidate on a sample of files and keeping the one under which the most functions analyse cleanly. The same sample is used to find annotation macros (for example `OGRE_DEPRECATED`) that show up as unknown types; they are defined as nothing if that does not lower the yield.
- **Anything else** an engine's build would supply can be given by hand in the config, per engine, as `defines`, `include_dirs` and `force_include` (a precompiled header the build includes in every file). Three engines need one such entry each: OGRE a define, Torque an include folder, Nebula its precompiled header.

The cost is that some files still have compile errors. Two levels of trust result:

| Level | Meaning | Used for |
| --- | --- | --- |
| Strict | The function's file, including everything it includes, compiled with no error | Validation and test rows |
| Relaxed | The function was analysed and has no error inside it, but its file has errors elsewhere | Training rows as well |

Relaxed rows may be missing findings from checks that depend on types declared in a header that failed. Each row records which level it is (`file_error_free`). `--eligibility strict` restricts every split to the strict level.

## Label policy

The scan records a broad set of checks. Which of them count as labels is decided in the build step by `labels` in the config, so it can be changed without scanning again.

- **Default labels** are checks that describe design, complexity, performance or bug-risk problems: function size, cognitive complexity, `performance-*`, `bugprone-*`, and selected `readability-`, `misc-` and `cppcoreguidelines-` checks.
- **Left out by default** are formatting and modernisation checks (braces, `nullptr`, C-style casts, `auto`) and checks that fire on almost every function (identifier length, magic numbers). In a trial on Irrlicht with those checks included, the braces check alone labelled 56% of the functions.
- **Size labels** come from the measured counts, with the thresholds in `labels.thresholds`: more than 80 lines, more than 6 parameters, nesting deeper than 4, or cognitive complexity above 25.
- **Rare checks** (fewer than `min_label_count` training rows) are removed from the vocabulary. A function whose only findings are rare checks is dropped, not called clean.

The fixing dataset has its own policy (`fixes`), because the checks worth fixing automatically are mostly the formatting and modernisation ones.

## Output

`data/collected/detection/{train,val,test}.jsonl`, one JSON object per line:

| Field | Meaning |
| --- | --- |
| `id` | Stable id derived from the code |
| `engine`, `path`, `name`, `start_line`, `end_line` | Where the function comes from |
| `checks` | Sorted label checks; empty for a clean function |
| `findings` | Each finding behind the labels: check, line within the function, message |
| `metrics` | Lines, statements, branches, parameters, variables, cognitive complexity |
| `file_error_free` | `true` for the strict trust level |
| `n_tokens` | Length of the whole conversation under the DeepSeek Coder chat template |
| `messages` | `system` / `user` / `assistant` conversation |

The assistant message is the target, for example `{"checks": ["cppcoreguidelines-init-variables", "readability-function-size"]}`.

Also written:

- `detection/{val,test}_length_matched.jsonl`: subsets with equal numbers of smelly and clean rows in every length bin.
- `detection/labels.json`: the label vocabulary with training counts.
- `fixing/{train,val,test}.jsonl`: same provenance fields, `fixed_checks`, and a conversation whose target is the fixed function.
- `stats.json`: counts at every stage, per engine and per check, and the accuracy of the rule "long means smelly".

Clean functions are sampled to follow the length distribution of the smelly ones within each engine, so length is a weak signal by construction.

To load for training:

```python
from datasets import load_dataset

ds = load_dataset("json", data_files={
    "train": "data/collected/detection/train.jsonl",
    "validation": "data/collected/detection/val.jsonl",
}).select_columns(["messages"])
```

The training settings in [DATASET.md](DATASET.md#using-it-for-fine-tuning) apply unchanged.

## Results of the first full run

Run on 8 October 2026 with clang-tidy 22.1.8 on the ten engines in `collect_config.json`, at the commits recorded in `data/collected/stats.json`.

**Scan and extraction**

| Engine | Standard chosen | Files | Files with no compile error | Functions kept | of which strict | Split |
| --- | --- | --- | --- | --- | --- | --- |
| cafu | C++17 | 882 | 475 | 3,265 | 2,150 | train |
| cocos2d-x | C++17 | 1,365 | 880 | 6,474 | 3,423 | train |
| Defold | C++17 | 952 | 369 | 6,689 | 1,472 | train |
| Esoterica | C++20 | 1,177 | 319 | 6,018 | 1,441 | train |
| godot | C++17 | 3,492 | 649 | 21,634 | 1,582 | train |
| nebula | C++17 | 1,659 | 0 | 8,231 | 0 | train |
| Torque | C++23 | 2,620 | 2,266 | 18,683 | 16,326 | train |
| irrlicht | C++17 | 635 | 604 | 3,795 | 3,637 | validation |
| ogre | C++17 | 1,323 | 1,019 | 6,946 | 5,573 | test |
| spring | C++17 | 1,276 | 734 | 6,692 | 1,566 | test |
| **Total** | | **15,381** | **7,315** | **88,427** | **37,170** | |

- 15,374 files were analysed; 6 failed and 1 still had a missing header. None of the 7 is used.
- clang-tidy analysed 263,372 functions. The 88,427 kept are those of 5 to 400 lines with no compile error inside them and a clean parse.
- 40,509 findings were ignored because they were reported at a macro invocation; 29,042 of them in Godot.
- The scan took 33 CPU-hours, about 66 minutes with 30 parallel processes. Extraction and the build take a few minutes each.

**Detection dataset** (40 checks in the vocabulary)

| File | Rows | Smelly | Clean | Engines |
| --- | --- | --- | --- | --- |
| `train.jsonl` | 28,050 | 14,828 | 13,222 | cafu, cocos2d-x, Defold, Esoterica, godot, nebula, Torque |
| `val.jsonl` | 2,443 | 1,425 | 1,018 | irrlicht |
| `test.jsonl` | 3,940 | 2,137 | 1,803 | ogre, spring |
| `val_length_matched.jsonl` | 1,866 | 933 | 933 | irrlicht |
| `test_length_matched.jsonl` | 3,340 | 1,670 | 1,670 | ogre, spring |

The most frequent labels, as a share of all 34,433 rows: `readability-function-size` 10.6%, `bugprone-narrowing-conversions` 10.0%, `cppcoreguidelines-owning-memory` 9.7%, `cppcoreguidelines-init-variables` 8.6%, `misc-unused-parameters` 6.5%, `readability-else-after-return` 5.6%, `readability-function-cognitive-complexity` 4.0%. The full list is in `detection/labels.json`.

Conversations are short: median 264 tokens, 90th percentile 733, longest 2,207.

**Fixing dataset**

| File | Pairs |
| --- | --- |
| `train.jsonl` | 18,849 |
| `val.jsonl` | 2,569 |
| `test.jsonl` | 3,827 |

83 different checks are fixed at least once. The most frequent are braces around statements (8,691 pairs), C-style casts (4,952), `nullptr` (4,308) and `auto` (4,253). 400 candidate pairs were dropped because the fixed code no longer parsed.

**Checks on the result**

- **The build's own verification** passes all 17 checks: schema, duplicates, split by engine, strict-only evaluation rows, findings inside their functions, token counts, and that every fixing target differs from its input.
- **Findings point at the right line.** [`scripts/audit/audit_collected_units.py`](../scripts/audit/audit_collected_units.py) tests 14 checks whose trigger is a visible token (an `else`, a `goto`, a cast, a named variable). Of 233,884 such findings across all engines, 99.1% have the token on the reported line. Thirteen of the checks are at 99.2% or above; the fourteenth, `modernize-use-auto`, is at 83.8% because the test pattern does not cover every initialiser form it reports. On the 2025 data a more lenient test (is the construct anywhere in the snippet) passed for only 22% to 63% of findings, depending on the check.
- **Length is a weak signal.** The rule "longer than 193 tokens means smelly" scores 56.7% on the test split, against 54.2% for always answering "smelly". On the 2025 data the same kind of rule scored 89.6%.
- **Strict and relaxed functions look alike overall.** Findings from type-dependent checks occur at 4.26 per 100 lines in strict functions and 4.29 in relaxed ones. Within single engines the relaxed rate is lower in 7 of 9 (Torque 3.2 against 4.7, Esoterica 1.1 against 2.2), which is consistent with some under-reporting there.

## Limits

- **The labels are lint findings, not game smells.** They say what clang-tidy reports. No check knows what an update loop or a game object is. Mapping checks to the game-smell catalogues would need a justified mapping or human labels.
- **Relaxed rows can under-report.** See the two trust levels above.
- **Windows view of the code.** The scan runs with the Windows toolchain's headers, so code for other platforms is inactive and never labelled.
- **Engines are analysed at whatever commit is checked out.** `stats.json` records the commit.
- **Fixes are not re-analysed.** A fixing target is checked to differ from its input and to parse, but clang-tidy is not run again on it to confirm the finding is gone or that it still compiles.
  The later [fixing pilot](FIXING.md) independently re-analyses a frozen sample; that validation does not retroactively certify every collected target.
- **Functions only.** Findings outside function bodies (class declarations, macros, globals) are recorded by the scan but not used.
- **Licences.** The rows contain engine source code under each engine's own licence (Spring, for one, is GPL). Check them before publishing a dataset built with this pipeline.
