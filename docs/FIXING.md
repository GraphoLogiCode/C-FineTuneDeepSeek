# Learned repair pilot and validation

The new fixer uses `models/collected_fixing_pilot_r16`, trained from the cached
DeepSeek-Coder-1.3B-Instruct base. It is independent of the invalid July 2025
fixing adapters. Model output is a **candidate function**, not an accepted fix.

## Run the repair demo

```powershell
.venv-demo\Scripts\python.exe demo\repair_app.py
```

Open <http://127.0.0.1:7861>. Choose a held-out function, generate a candidate,
inspect its diff, and read the validation evidence. Reference-check mode isolates
repair quality. Detector mode supplies the current detector's predictions; it
abstains when the requested checks have no support in repair training. Edited
input without matching source context is explicitly inconclusive. No patch is
applied to the engine sources.

The app requires the local base/adapter, collected data, original engine sources,
and cached collection flags/scans. Missing or changed context cannot produce a
static pass. See `requirements-demo.txt` and the detection-demo setup instructions
for the isolated environment; clang-tidy must match the recorded collection
version, 22.1.8. The compiled behavior suite also requires a C++ compiler.

## What was trained

`scripts/audit/audit_fixing_dataset.py` froze 4,096 training, 128 validation and
128 test rows before model evaluation, with seed 20261015. Sampling uses strict
original-source eligibility, check/engine strata and a stored conversation length
cap. It preserves original engine splits, IDs, prompts and targets. The training
set contains six engines; Nebula has no strict candidates. Validation is Irrlicht;
the test has 70 OGRE and 58 Spring functions.

Stored token counts use an older locally saved tokenizer. Actual base-tokenizer
encoding excludes 138 training and four validation conversations above 2,048
tokens, so the model trained on **3,958 rows**, using **124 validation rows**.
The frozen test still contains all 128 rows. Training never truncates input or
answer text; generation-limit failures are counted separately and cannot pass.
Audit evidence and exact dataset hashes are in `data/fixing_pilot/`.
The collection builder now loads the same base tokenizer as the trainer instead
of preferring the legacy merged model's tokenizer. Existing frozen files were not
rebuilt; new collections will use the corrected token counts.

Training used rank-16 LoRA on all seven projections, alpha 32, dropout 0.05,
learning rate 2e-4, seed 42 and two epochs. Answer-only supervision uses the
model's chat template. The epoch with lower validation loss was saved. Measured
training took 6.5 minutes and peaked at 15.5 GB allocated GPU memory on one RTX
5090. Final settings and logs are in the adapter folder.

The actual retained training rows support 66 check names. Only 14 of the current
detector's 40 labels occur in this repair training. The two tasks are not
interchangeable. The primary 128-row benchmark supplies **reference check names**
to every repair model; describe it as conditional/oracle-check repair.

## Validation contract

`src/evaluation/repair_validation.py` independently:

1. Verifies the original file SHA1, engine commit, clang-tidy version, collection
   configuration and compiler flags against recorded provenance.
2. Locates the exact original function by syntax, text and line range; checks
   the candidate retains the function identity and introduces no extra function.
3. Replaces it through a temporary Clang virtual filesystem overlay. Relative
   includes still resolve at the original source path; original files stay intact.
4. Re-analyses the original and candidate translation units with the collection
   flags. The original must compile/analyse and contain the requested findings.
5. Requires a changed, complete candidate, no candidate compiler errors, removal
   of every requested target diagnostic, and no new enabled main-file warning.
   Diagnostic suppression is rejected. Missing analysis/context is inconclusive.

A pass is **static validation under collection flags**. Those inferred flags can
include stub headers. It is not a native engine build, an engine integration test,
a proof of equivalence, or a developer's judgement that the patch is desirable.
Standalone tree-sitter parsing is reported separately: macro-dependent methods
can fail snippet parsing while the full source compiles.

## Evaluation artifacts

See [FIXING_RESULTS.md](FIXING_RESULTS.md) for tables recomputed from the saved
predictions and validation logs, with uncertainty and workshop-paper limits.

- `results/collected_fixing/pilot_test/`: trained/base/no-op/recorded-target
  candidates, generation provenance, per-engine/check metrics and static reports.
- `results/collected_fixing/tool_test/`: fresh direct-tool candidates and checks.
- `results/collected_fixing/pipeline_test/`: restricted detector-to-fixer run.
- `results/collected_fixing/behavior/`: illustrative compiled regression fixtures.
- `results/collected_fixing/engine_component/`: a preselected actual Spring component,
  compiled with real headers and no stubs, plus semantic negative controls.
- `data/fixing_pilot/reference_static_validation.jsonl`: independently rechecked
  recorded tool targets and no-op controls.

The independent reference audit found **100/128** recorded targets pass static
validation and **0/128** no-ops pass. Rejection categories overlap: 21 reference
patches introduce enabled warnings, 10 leave requested warnings, and two introduce
compiler errors. Applying one automatic edit does not guarantee all findings of
that check disappear. The collection builder did not previously recheck this.

The behavior suite contains six hand-written functions and 36 explicit input
assertions per system. Original and reference controls must pass; deliberately
incorrect but compilable functions must fail. These fixtures test the validator's
ability to catch behavior changes; they are not held-out engine behavior evidence.

The separate native component test exercises one frozen Spring equality method.
Both neural candidates compile and pass its 31,468 assertions. A deliberately
incorrect candidate that ignores the version passes static validation but fails
the native tests. This establishes the static gate's limit on one real component;
it does not estimate neural semantic failures across the benchmark.

## Reproduce

Do not rebuild the frozen dataset while reproducing its saved evaluation. A new
sampling run creates a new manifest and should use a separate output directory.

```powershell
$env:HF_HUB_OFFLINE='1'
.venv-demo\Scripts\python.exe scripts\audit\audit_fixing_dataset.py --output data/fixing_pilot_rebuilt
.venv-demo\Scripts\python.exe src\training\train_lora.py --data data/fixing_pilot --out models/collected_fixing_reproduction --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --lora-targets all --lr 2e-4 --epochs 2 --patience 0 --batch-tokens 4096 --max-length 2048 --seed 42 --log-every 25
.venv-demo\Scripts\python.exe src\evaluation\evaluate_fixing.py --adapter models/collected_fixing_pilot_r16 --batch-size 4 --max-new-tokens 1536 --validate --workers 6
.venv-demo\Scripts\python.exe src\evaluation\repair_behavior.py --adapter models/collected_fixing_pilot_r16
.venv-demo\Scripts\python.exe -m src.evaluation.repair_engine_component
.venv-demo\Scripts\python.exe -m unittest discover -s tests -v
```

The main evaluator compares the same target IDs across systems and verifies
dataset, adapter, tokenizer and decoding provenance before reusing predictions.
Exact reference match is textual agreement, not correctness. The static pass
denominator is every attempted test function, including rejection/inconclusive
outcomes. Confidence intervals describe function-level uncertainty in this
purposively balanced pilot; only two test engines and one training seed remain.
Both neural systems share the same code extractor, accepting a single complete
fenced C++ block with surrounding prose and rejecting multiple blocks. Raw
generations and the earlier strict-format evaluation are preserved.

## Workshop/short-paper assessment

See [REPAIR_RESEARCH_ASSESSMENT.md](REPAIR_RESEARCH_ASSESSMENT.md) for primary
literature, research questions and evidence requirements. Prior work already
covers learned lint repair and C++ repair with type context. Training a model to
copy available clang-tidy edits is not sufficient novelty. A defensible direction
is a bounded empirical study of cross-engine repair failures and validation,
followed by relevant context or resource-cost ablations. Stronger claims need
independent engine tests, developer-reviewed patches, repeated seeds, an untouched
final evaluation and a demonstrated advantage or useful failure analysis against
the direct tool. No paper acceptance or behavioral correctness is established by
this pilot.
