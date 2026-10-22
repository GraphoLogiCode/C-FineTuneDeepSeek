"""Build the project fundamentals report from prose and verified local artifacts.

Requires reportlab, pypdf and Pillow (available in the Codex document runtime).
Run from any directory. No training, generation, source scanning or engine edits occur.
"""
from __future__ import annotations

import ast
from collections import Counter
from html import escape
import hashlib
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "output/pdf/Project_Fundamentals_and_Techniques.pdf"
HTML = ROOT / "output/reports/Project_Fundamentals_and_Techniques.html"
CHECKS = ROOT / "output/reports/fundamentals_report_verification.json"
REPORT_DATE = "October 14, 2026"
NAVY = colors.HexColor("#132B44")
TEAL = colors.HexColor("#087F8C")
INK = colors.HexColor("#243548")
GRAY = colors.HexColor("#536475")
LIGHT = colors.HexColor("#F0F5F8")
LINE = colors.HexColor("#D7E2EA")
WIDTH = 504


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding="utf-8").splitlines() if x]


def verify_evidence():
    # Use the actual scoring functions, without importing torch or loading a model.
    tree = ast.parse((ROOT / "src/evaluation/evaluate_detection.py").read_text(encoding="utf-8"))
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in {"prf", "score", "parse_answer", "length_rule"}]
    ns = {"Counter": Counter, "json": json}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "evaluate_detection.py", "exec"), ns)
    evidence = {"snapshot_date": REPORT_DATE, "method": "Recompute saved outputs; no fresh model inference."}
    runs = {}
    for run in ("lora_r16_e3", "lora_qkv_r8_epoch4_interrupted", "base_zero_shot"):
        base = f"results/collected_detection/{run}/test"
        pred = rows(base + "/predictions.jsonl")
        stored = load(base + "/metrics.json")
        gold, predicted = [set(x["reference"]) for x in pred], [set(x["predicted"]) for x in pred]
        calc = ns["score"](gold, predicted, [x["valid"] for x in pred])
        assert calc == stored["results"]["all"]["model"], f"Stored metrics disagree: {run}"
        runs[run] = {"rows": len(pred), "all_metrics_match": True, "metrics": calc,
                     "raw_reparse_mismatches": sum(ns["parse_answer"](x["raw"]) != (set(x["predicted"]), x["valid"]) for x in pred)}
    # Verify the two simple baselines and the length-matched subset as well.
    data = rows("data/collected/detection/test.jsonl")
    train = rows("data/collected/detection/train.jsonl")
    by_id = {x["id"]: x for x in data}
    p = rows("results/collected_detection/lora_r16_e3/test/predictions.jsonl")
    assert len(by_id) == len(p) == 3940
    assert all(by_id[x["id"]]["checks"] == x["reference"] for x in p)
    stored = load("results/collected_detection/lora_r16_e3/test/metrics.json")
    top = Counter(c for x in train for c in x["checks"]).most_common(1)[0][0]
    assert top == stored["length_rule_label"]
    threshold = stored["length_rule_threshold"]
    assert ns["length_rule"](train) == threshold
    matched = {x["id"] for x in rows("data/collected/detection/test_length_matched.jsonl")}
    for subset, pp in (("all", p), ("length_matched", [x for x in p if x["id"] in matched])):
        gold = [set(x["reference"]) for x in pp]
        systems = {
            "model": ([set(x["predicted"]) for x in pp], [x["valid"] for x in pp]),
            "baseline: always clean": ([set() for _ in pp], None),
            "baseline: length rule": ([{top} if by_id[x["id"]]["n_tokens"] >= threshold else set() for x in pp], None),
        }
        for system, (predicted, valid) in systems.items():
            assert ns["score"](gold, predicted, valid) == stored["results"][subset][system]
    reference = rows("data/fixing_pilot/reference_static_validation.jsonl")
    ref_pass = sum(x["reference"]["static_validated"] for x in reference)
    noop_pass = sum(x["no_op"]["static_validated"] for x in reference)
    assert (len(reference), ref_pass, noop_pass) == (128, 100, 0)
    behavior = load("results/collected_fixing/behavior/metrics.json")
    assert behavior["adapter"] is None and behavior["controls_ok"] is True
    stats = load("data/collected/stats.json")
    for task in ("detection", "fixing"):
        for split in ("train", "val", "test"):
            assert len(rows(f"data/collected/{task}/{split}.jsonl")) == stats[task][split]["rows"]
    settings = load("models/collected_detection_lora/run_settings.json")
    repair = load("models/collected_fixing_pilot_r16/run_settings.json")
    assert (repair["train_rows"], repair["val_rows"]) == (3958, 124)
    runtime = load("data/fixing_pilot/runtime_tokenization.json")
    assert hashlib.sha256((ROOT / "data/fixing_pilot/manifest.json").read_bytes()).hexdigest() == runtime["manifest_sha256"]
    for split in ("train", "val", "test"):
        assert hashlib.sha256((ROOT / f"data/fixing_pilot/{split}.jsonl").read_bytes()).hexdigest() == runtime["frozen_outputs_sha256"][split]
    excluded = set(runtime["comparison"]["train"]["excluded_ids"])
    retained = [x for x in rows("data/fixing_pilot/train.jsonl") if x["id"] not in excluded]
    repair_checks = {c for x in retained for c in x["fixed_checks"]}
    overlap = repair_checks & set(load("data/collected/detection/labels.json")["checks"])
    assert (len(retained), len(repair_checks), len(overlap)) == (3958, 66, 14)
    evidence.update(detection_runs=runs, detection_evaluation=stored, baselines_and_matched_subset_match=True,
                    dataset_counts_match=True, dataset_stats=stats,
                    detection_settings=settings, repair_settings=repair,
                    reference_static={"rows": len(reference), "reference_pass": ref_pass, "noop_pass": noop_pass},
                    behavior=behavior,
                    repair_dataset_hashes_match=True,
                    repair_training_checks=len(repair_checks), detector_repair_check_overlap=len(overlap),
                    missing_repair_aggregate_directories=[str(x.relative_to(ROOT)).replace("\\", "/") for x in
                        [ROOT / "results/collected_fixing/pilot_test", ROOT / "results/collected_fixing/pipeline_test"] if not x.exists()])
    return evidence


def p(text): return ("p", text)
def h(text): return ("h", text)
def small(text): return ("small", text)
def code(text): return ("code", text)
def box(title, text): return ("box", title, text)
def table(head, data, widths): return ("table", head, data, widths)
def source(text): return ("source", text)


def sections(e):
    m = e["detection_runs"]["lora_r16_e3"]["metrics"]
    baselines = e["detection_evaluation"]["results"]["all"]
    return [
    ("PROJECT REPORT", "Fundamental Knowledge\nand Techniques", [
        ("subtitle", "DeepSeek Fine-Tuning for C++ Code Analysis and Repair"),
        small(f"C-FineTuneDeepSeek  |  Repository-based technical study  |  {REPORT_DATE}"),
        h("Purpose and scope"),
        p("This project adapts a pretrained code language model to analyse C++ functions from game engines. It combines software engineering, compiler-based static analysis, data preparation, supervised machine learning, efficient neural-network training, and empirical evaluation. This report explains the knowledge behind each part and shows where the techniques appear in the implementation."),
        table(["Task", "Current input", "Current output"], [
            ["Detection", "One C++ function", 'JSON such as {"checks": ["misc-unused-parameters"]}'],
            ["Repair", "One C++ function plus requested clang-tidy check names", "A complete rewritten function, as source text"],
        ], [70, 185, 249]),
        small("The examples describe output contracts, not verified findings for an unseen function. The current repair contract differs from the older README's JSON-and-explanations description."),
        h("How the parts connect"),
        ("pipeline",),
        box("Central idea", "The model learns to reproduce selected tool findings and tool-generated edits. Detection scores measure agreement with these references. A generated repair still requires independent validation."),
        h("How to read this report"),
        p("Pages 2-4 cover prerequisites and data. Pages 5-8 explain the model, adapters, learning objective and optimisation. Pages 9-11 cover inference, measured results and repair validation. Page 12 provides the reading order and primary references."),
        source("Evidence: README.md; src/detection_prompt.py; src/fixing_prompt.py; scripts/collect/; src/training/train_lora.py. The report concerns the local snapshot reviewed for this request."),
    ]),
    ("01 / FOUNDATIONS", "Knowledge needed to understand the project", [
        p("The project uses several fields together. Programming and data handling explain how the pipeline works; linear algebra and probability explain the model; experimental design explains what the results support."),
        table(["Foundation", "Knowledge used here", "Why it matters"], [
            ["Python software engineering", "Functions, classes, dictionaries, JSONL, pathlib, command-line arguments and subprocesses", "Connect collection, training, inference and validation into reproducible scripts."],
            ["C++ programming", "Functions, scope, pointers, ownership, const, casts, classes, headers and macros", "Interpret warnings and determine whether a proposed code change is appropriate."],
            ["Compiler concepts", "Parsing, syntax trees, translation units, includes and compilation flags", "Extract the right function and analyse it with the context C++ requires."],
            ["Linear algebra", "Vectors, matrix multiplication, tensor dimensions and low-rank matrices", "Understand attention, linear projections and LoRA updates."],
            ["Probability and calculus", "Conditional probability, softmax, logarithms, gradients and the chain rule", "Understand next-token prediction, cross-entropy and backpropagation."],
            ["Machine learning and statistics", "Training/validation/test splits, overfitting, baselines, precision, recall and F1", "Separate memorisation and formatting success from useful generalisation."],
        ], [99, 207, 198]),
        h("Syntax, static evidence and program behaviour"),
        code("int choose(int value) {\n    if (value > 0) return value;\n    else return 0;\n}"),
        p("Removing the <b>else</b> and leaving <b>return 0;</b> after the conditional is a useful illustration of readability-else-after-return. Parsing checks grammatical structure. Static analysis checks configured source properties. Tests check behaviour on selected inputs. These are different forms of evidence; passing one does not establish the others."),
        p("Refactoring aims to preserve well-defined observable behaviour. Bug repair may intentionally change behaviour to meet a stated requirement. This distinction matters when a warning concerns an uninitialised variable or a potentially unsafe conversion."),
        box("Suggested starting point", "First follow one dataset row from a C++ function to its target answer. Then inspect how the trainer converts that conversation into token IDs and labels. This makes the later mathematics concrete."),
        source("Code anchors: scripts/collect/extract.py::extract_file; src/training/train_lora.py::encode; src/evaluation/repair_validation.py::RepairValidator; src/evaluation/repair_behavior.py::validate_behavior."),
    ]),
    ("02 / DATA ENGINEERING", "Static analysis, parsing and automatic labels", [
        h("A three-stage collection pipeline"),
        table(["Stage", "Technique", "Implementation role"], [
            ["Scan", "clang-tidy static analysis", "Runs configured checks over engine source and records diagnostics, replacement edits, errors and source provenance."],
            ["Extract", "Tree-sitter syntax-tree traversal", "Locates function definitions and maps diagnostic byte offsets into the correct function span."],
            ["Build", "Filtering, normalisation and chat-format data construction", "Creates function-to-check examples and before/after repair pairs with stable IDs and engine splits."],
        ], [65, 161, 278]),
        p("clang-tidy supplies tool findings and some automatic edits [R1]. Tree-sitter supplies concrete syntax trees [R2]; its role here is function boundaries and structural checks. C++ semantic analysis still depends on the compiler, headers, macros and declarations. A function that parses alone may still fail to compile in its project."),
        h("Labels are an operational definition of the task"),
        p("The detection builder selects checks for bug risk, readability, complexity and performance, canonicalises aliases, and retains checks with at least 30 occurrences in training candidates. The final vocabulary contains 40 checks. Size-related labels use configured thresholds: more than 80 lines, more than 6 parameters, nesting deeper than 4, or cognitive complexity above 25."),
        p("These are automatically produced reference labels, a form of weak supervision. They are not human judgements about every design problem in a game engine. An empty label set means none of the retained checks was reported under the collection policy. It does not establish that the code is free of defects."),
        h("Trust levels and source provenance"),
        p("The collector infers include paths, macros and language settings and can create stub headers for unavailable dependencies. <b>Strict</b> rows have no compiler errors in the analysed translation unit; <b>relaxed</b> rows can have errors elsewhere but none inside the extracted function. Training permits relaxed rows; validation and test require strict rows. Both remain specific to the inferred collection environment."),
        p("Source hashes, engine commits, tool versions and configuration fingerprints connect outputs to the source that produced them. Extraction rejects files changed since scanning. Repair validation later checks the original context again."),
        source("Implementation: scripts/collect/scan.py::prepare, calibrate, scan_engine; extract.py::extract_file; build_dataset.py::label_unit, apply_fixes, build; collect_config.json. Background: [R1-R2]."),
    ]),
    ("03 / EXPERIMENTAL DESIGN", "Data splits, leakage and shortcut controls", [
        p("A model should be tested on code it did not train on. The current pipeline assigns entire engines to splits, which makes the test harder and more informative than randomly dividing related functions from the same engines."),
        table(["Split", "Detection functions", "Engine allocation"], [
            ["Training", "28,050", "Cafu, cocos2d-x, Defold, Esoterica, Godot, Nebula and Torque"],
            ["Validation", "2,443", "Irrlicht"],
            ["Test", "3,940", "OGRE and Spring"],
            ["Total", "34,433", "Ten engines; each belongs to one split"],
        ], [90, 120, 294]),
        h("Three techniques that reduce misleading results"),
        p("<b>1. Deduplication.</b> Normalised source text groups duplicate functions across files and engines. The builder retains one copy and drops groups with conflicting labels. This controls exact or normalisation-equivalent copies; it does not prove the absence of near duplicates or related implementations."),
        p("<b>2. Shortcut controls.</b> Smelly and clean functions are sampled using code-length bins. Separate evaluation subsets balance positive and negative examples within conversation-length bins. A baseline also predicts from length alone. These checks test whether apparent success could be explained by a simple length shortcut; they cannot prove which internal features the model uses."),
        p("<b>3. Train/validation/test separation.</b> Training adjusts adapter parameters. Validation selects the checkpoint. Test measures the selected system on held-out engines. Tuning prompts, labels or hyperparameters repeatedly against test scores would weaken the test's independence."),
        h("Repair data has a separate scope"),
        p("The collected repair dataset has 18,849 training, 2,569 validation and 3,827 test pairs, or 25,245 pairs overall. Targets are constructed from non-conflicting clang-tidy replacements inside each function. The pilot freezes a smaller 4,096/128/128 sample; actual tokenizer checks retain 3,958 training and 124 validation conversations within its 2,048-token cap. Its 128 test identities remain fixed."),
        box("Reproducibility", "Stable row IDs, frozen manifests, source and dataset hashes, recorded settings, random seeds and per-example outputs make results auditable. A fixed seed aids reproduction; multiple seeds and more held-out engines are still needed to assess variability."),
        source("Evidence: data/collected/stats.json; build_dataset.py::build, sample_engine, length_matched; scripts/audit/audit_fixing_dataset.py; data/fixing_pilot/manifest.json; models/collected_fixing_pilot_r16/run_settings.json."),
    ]),
    ("04 / LANGUAGE MODELS", "Transformers, tokens and transfer learning", [
        h("What the pretrained model contributes"),
        p("The system starts with deepseek-ai/deepseek-coder-1.3b-instruct, an instruction-tuned code model [R3]. Pretraining supplies general code representations and continuation ability. Task-specific fine-tuning adapts that starting point to the project's input/output examples. The project does not train a language model from scratch."),
        p("A tokenizer converts source and instructions into integer token IDs. A token can represent part of an identifier, punctuation or a longer text fragment; tokens are not the same as source lines or words. Embeddings turn IDs into vectors. Transformer blocks then contextualise those vectors and produce scores for the next token."),
        h("Self-attention: relating one token to earlier tokens"),
        p("For one attention head, let X be the sequence of hidden vectors, with shape T by d, where T is the token count and d the hidden dimension. Learned projections form query, key and value matrices. The following is the standard attention abstraction, not a complete specification of this checkpoint [R4]."),
        code("Q = X Wq       K = X Wk       V = X Wv\nAttention = softmax(Q K^T / sqrt(dk) + M) V"),
        p("For head width dk, Wq and Wk have shape d by dk; Wv has shape d by dv. Thus QK<super>T</super> is T by T and the result is T by dv. M is a causal mask: positions after the current token receive negative infinity before softmax. Each row of softmax is a normalised distribution over allowed positions. Multiple heads combine different learned relationships."),
        h("The task is classification expressed through generation"),
        code("P(y | x) = product over j=1..m of P(yj | x, y1, ..., y(j-1))"),
        p("Here x is the rendered prompt and code; y is the m-token answer. Detection generates text containing a JSON check list, which is parsed into a label set. Repair generates function source. The current trainer uses AutoModelForCausalLM and a causal-language-model LoRA configuration, rather than a separate classification head."),
        p("This explains why both content and output formatting must be learned. Correct-looking prose can fail the detection contract. Valid JSON can still name the wrong checks. Longer input also cannot supply missing type declarations unless those declarations actually appear in the prompt."),
        source("Implementation: train_lora.py::main; src/detection_prompt.py; src/fixing_prompt.py; evaluate_detection.py::parse_answer. Model and theory: [R3-R4]."),
    ]),
    ("05 / EFFICIENT ADAPTATION", "LoRA and the earlier QLoRA experiments", [
        h("Low-rank adaptation"),
        p("LoRA freezes a pretrained weight matrix and learns a small factorised update [R5]. For a column-vector input x, take W0 with shape dout by din, A with shape r by din and B with shape dout by r. The rank r is much smaller than either full dimension."),
        code("W = W0 + (alpha / r) B A\nh = W0 x + (alpha / r) B (A x)\nFull matrix: dout * din parameters\nAdapter:    r * (din + dout) parameters"),
        p("For an illustrative 2,048 by 2,048 matrix and r = 8, a full matrix has 4,194,304 entries; the two adapter matrices have 32,768, or 0.78% as many. This is a worked example, not the aggregate count for the model. The formula shows why adapter training can reduce parameter and optimiser storage."),
        h("How this repository applies it"),
        table(["Setting", "Current trainer defaults", "Verified first detection run"], [
            ["Rank / alpha / dropout", "8 / 16 / 0.1", "16 / 32 / 0.05"],
            ["Adapted projections", "q_proj, k_proj, v_proj", "q/k/v/o plus gate/up/down projections"],
            ["Trainable parameters", "2,359,296 in the saved qkv run", "14,991,360"],
            ["Base weight precision", "bfloat16", "bfloat16"],
        ], [137, 176, 191]),
        p("get_peft_model attaches the adapters. The base parameters stay frozen; gradients update the adapter parameters. Alpha scales the update and dropout regularises its training branch. Choosing more projections or a larger rank changes capacity and cost, but does not guarantee a better score. The saved first run and the current defaults are different configurations."),
        h("QLoRA is historical in this repository"),
        p("The older notebook combines LoRA with frozen 4-bit base weights, NF4 quantisation and double quantisation, following the general QLoRA approach [R6]. It also includes gradient checkpointing, which trades extra recomputation for lower activation storage, and device offloading. These memory techniques belong to the earlier experimental path."),
        box("Precise terminology", "The current src/training/train_lora.py loads bfloat16 weights without a 4-bit quantisation configuration. Describe the current training as LoRA in bfloat16; describe the older notebook path as QLoRA. The old runs' validity issues are separate from the general usefulness of QLoRA."),
        source("Code: train_lora.py::ATTENTION_QKV, ALL_PROJECTIONS, main; notebooks/DeepSeek_Coder_1.3B_FineTuned.ipynb::fine_tune_variant; saved run_settings.json files. Background: [R5-R6]."),
    ]),
    ("06 / SUPERVISED LEARNING", "Chat formatting and answer-only loss", [
        h("Turn a labelled example into a learning signal"),
        p("Each JSONL row stores a conversation: a system instruction, a user message containing code, and an assistant target. The trainer renders the base model's own chat template. Special role and end-of-turn tokens are part of the model's expected format [R7]. Replacing them with a wrapper from another model changes the input distribution."),
        code("prompt_ids = tokenizer(render(system, user))\nanswer_ids = tokenizer(target + end_of_turn)\ninput_ids  = prompt_ids + answer_ids\nlabels     = [-100] * len(prompt_ids) + answer_ids"),
        p("The value -100 marks an ignored target for the loss [R8]. Prompt tokens remain visible as context. Masking their loss does not remove their influence on the answer. During training, teacher forcing supplies preceding reference answer tokens so the model learns the next token in the correct sequence."),
        h("The mathematical objective"),
        p("Let z be the concatenated prompt and answer, S the positions belonging to the answer (including its end token), and theta the trainable adapter parameters. The per-example objective averages the negative log probability of each supervised token:"),
        code("L(theta) = -(1 / |S|) * sum over t in S of\n           log P_theta(z_t | z_1, ..., z_(t-1))"),
        p("The causal model shifts predictions internally: the logits before a target position predict that position's token. The trainer therefore supplies unshifted labels matching input_ids. Input IDs, attention masks and labels have shape B by T for batch size B and padded length T; model logits have shape B by T by V for vocabulary size V."),
        h("Three concrete safeguards in the current code"),
        p("<b>Exact boundary.</b> Prompt and answer are tokenised separately, matching generation at inference. check_encoding verifies one beginning token, the final end token, and that the supervised IDs decode to the intended answer."),
        p("<b>No answer truncation.</b> Conversations longer than the configured limit are skipped and counted, rather than cut. The current default is 4,096 total tokens; the repair pilot uses 2,048. The maximum is a pipeline choice, not a claim about the base model's largest supported context."),
        p("<b>Padding is ignored.</b> Padding receives attention-mask value 0 and label -100. Validation loss is weighted by the number of supervised tokens. Low loss reflects answer-token prediction; generated-output evaluation is still necessary."),
        source("Code: train_lora.py::split_conversation (line 51), encode (60), check_encoding (69), collate (112), validation_loss (125). Formatting/loss references: [R7-R8]."),
    ]),
    ("07 / OPTIMISATION", "Backpropagation and training control", [
        h("How the model learns"),
        p("A forward pass computes answer loss. Backpropagation applies the chain rule to compute gradients for trainable adapter parameters. AdamW maintains moving statistics of those gradients to determine an update [R9]. The repository explicitly sets weight_decay=0.0, so this configuration does not apply a nonzero weight-decay penalty."),
        table(["Technique", "Current implementation and purpose"], [
            ["Learning-rate schedule", "3% warm-up, then cosine decay over planned updates; avoids starting at the full rate immediately."],
            ["Gradient clipping", "Caps the overall gradient norm at 1.0 before each optimiser step to limit unusually large updates."],
            ["bfloat16 and autocast", "Loads the base model in bfloat16 and uses CUDA autocast for forward passes to manage memory and compute cost."],
            ["Length-aware batches", "Groups similar lengths, with at most 16 rows and a nominal limit of 6,144 padded tokens: B times the longest sequence."],
            ["Validation selection", "Saves the adapter with lowest validation loss. Current defaults allow up to 10 epochs and stop after 3 unimproved epochs."],
        ], [131, 373]),
        p("The token budget is a batching heuristic, not a full memory model. Attention cost can grow with sequence length squared. Short batches have their mean loss scaled by batch row count divided by the configured batch size. Because losses average answer tokens and sequence lengths vary, this is not an exact derivation of equal weighting for every example."),
        h("Training algorithm used by the current script"),
        code("Inputs: train/validation conversations, tokenizer, base, LoRA settings\n1. Encode conversations; check answer boundaries; skip overlong rows.\n2. Freeze the base; attach LoRA; initialise optimiser and schedule.\n3. For each epoch, shuffle rows and form similar-length batches.\n4. For each batch:\n     pad IDs, attention masks and labels\n     compute causal answer loss; scale for short batch\n     backpropagate; clip gradients; update; advance schedule\n     clear gradients\n5. Compute validation answer-token loss.\n6. Save if it improves; otherwise update the patience counter.\n7. Stop at the epoch limit or patience limit; retain the best adapter."),
        box("Overfitting in the saved first run", "Validation loss was 0.0801, 0.0690 and 0.0711 across three epochs. The minimum is at epoch 2, so that checkpoint was retained. More training steps did not continue to improve held-out loss."),
        source("Implementation: train_lora.py::pack, make_batches, validation_loss, main; models/collected_detection_lora/run_settings.json. Optimiser reference: [R9]. This script steps once per batch; it does not accumulate several batches before stepping."),
    ]),
    ("08 / INFERENCE AND METRICS", "Generate, parse and measure the answer", [
        h("Inference uses learned parameters without updating them"),
        p("The evaluator loads the base and adapter, merges the adapter for evaluation, switches to evaluation mode and renders prompts without assistant targets. Greedy decoding uses do_sample=False. Each step chooses a highest-scoring next token until the end token or output limit. This reduces sampling variability; it does not guarantee correct output or bitwise reproducibility across every hardware setup."),
        code("Input: function -> shared chat template -> token IDs\n       -> base + detection adapter -> generated suffix\n       -> parse {\"checks\": [...]} -> predicted check set"),
        p("Only newly generated IDs are decoded. Batched generation uses left padding. The detection demo reserves room for output within a 4,096-token working budget and exposes the prompt, raw response and parsed result. Gradio supplies the local browser interface; queues and model locks control concurrent requests. The demo does not train the model when a user submits code."),
        h("Detection is a multi-label problem"),
        p("A function can trigger several checks. Let Gi be the reference set and Pi the predicted set for function i. Across all functions, true positives TP count shared checks, false positives FP count predicted-only checks, and false negatives FN count missed checks."),
        code("Precision = TP / (TP + FP)\nRecall    = TP / (TP + FN)\nF1        = 2 TP / (2 TP + FP + FN)"),
        p("For reference {A, B} and prediction {A, C}, TP=1, FP=1 and FN=1. Precision, recall and F1 are each 0.5; exact set match is false, while the binary smelly/clean decision is correct. This is why a single accuracy score cannot describe the whole task."),
        table(["Metric", "Question it answers"], [
            ["Parsed-output rate", "Can the response be parsed into an object containing a checks list?"],
            ["Exact set match", "Did the model recover the entire check set for this function?"],
            ["Smelly/clean accuracy", "Did it correctly predict whether the selected check set is empty?"],
            ["Micro F1", "How well are all individual check decisions recovered when pooled?"],
            ["Macro F1", "What is the mean per-check F1 over checks appearing in this split?"],
        ], [140, 364]),
        small("The parser accepts an embedded JSON object and filters non-string list entries; parsed-output rate is not strict whole-response schema compliance. In the scorer an invalid parse becomes an empty set, so it can still coincide with a clean reference. Inspect validity alongside content metrics."),
        source("Code: evaluate_detection.py::parse_answer (line 41), score (67), generate (131); src/inference/detection_demo.py::Detector; demo/detection_app.py. Metric definitions follow the actual scorer."),
    ]),
    ("09 / VERIFIED FINDINGS", "What the saved experiments establish", [
        p("For this report, metrics were recomputed from saved reference and predicted sets, using the repository's scoring functions. All stored test metrics match for the rank-16 run, the interrupted qkv rank-8 run and the untuned base. The following table focuses on the rank-16 system and the simple baselines on 3,940 held-out functions."),
        table(["System", "Exact match", "Binary accuracy", "Micro F1"], [
            ["LoRA rank 16", f"{m['exact']:.2%}", f"{m['smelly_accuracy']:.2%}", f"{m['micro_f1']:.4f}"],
            ["Always clean", f"{baselines['baseline: always clean']['exact']:.2%}", f"{baselines['baseline: always clean']['smelly_accuracy']:.2%}", f"{baselines['baseline: always clean']['micro_f1']:.4f}"],
            ["Length rule", f"{baselines['baseline: length rule']['exact']:.2%}", f"{baselines['baseline: length rule']['smelly_accuracy']:.2%}", f"{baselines['baseline: length rule']['micro_f1']:.4f}"],
        ], [167, 107, 123, 107]),
        p("The adapter's parsed-output rate is 100%, micro precision is 0.8271, micro recall is 0.7136, and macro F1 is 0.4893. The lower macro F1 shows uneven performance across checks, including rare checks and checks requiring context absent from the function. Detection is strongest for patterns visible in the supplied text."),
        h("Check the alternative explanation"),
        p("On the 3,340-function length-matched test subset, the adapter reaches 85.06% binary accuracy and 0.7565 micro F1, while the length rule reaches 49.88% accuracy. This is evidence against length alone explaining the detector's performance. It does not establish that length is unused inside the model."),
        p("The first detection run trains 14,991,360 parameters, takes 72.7 recorded minutes, and peaks at 21.9 GB of allocated GPU memory on the recorded RTX 5090. The interrupted rank-8 qkv test run has micro F1 0.7444. Its different target modules, learning rate and training duration prevent isolating rank as the cause of the difference."),
        h("Why the historical audit matters"),
        p("The repository's audit of the 2025 path documents answer truncation, loss on repeated prompts, inconsistent label schemas, unrelated repair targets, cross-split overlap and zero-sample evaluation. These are recorded historical findings; their audit scripts were not rerun for this report. They explain why decreasing training loss alone could not establish success and why the rebuilt pipeline uses explicit contracts and saved per-example outputs."),
        box("Practical interpretation", "The current evidence supports learning to reproduce selected clang-tidy findings on the two held-out engines. It does not establish complete game-specific smell detection, behaviourally correct repair, reliability on all C++ projects, or a causal explanation of the model's decisions."),
        small("Verification scope: saved parsed predictions were rescored; no model was retrained or rerun. Tuned raw-output reparsing also matched. Untuned raw strings were stored with a 400-character cap, so some full parses cannot be reconstructed from those raw strings alone."),
        source("Primary artifacts: results/collected_detection/lora_r16_e3/test/{metrics.json,predictions.jsonl}; qkv/base test artifacts; data/collected/detection/test_length_matched.jsonl; model run_settings.json. Historical context: docs/AUDIT.md."),
    ]),
    ("10 / PROGRAM REPAIR", "Generate and validate candidate repairs", [
        h("Conditional repair is a separate learning task"),
        p("The current repair adapter receives a function and requested check names, then returns one complete rewritten function. The collected targets imitate available clang-tidy edits. Reference-check evaluation supplies the known checks and isolates repair performance; a detector-to-fixer pipeline adds detection errors and incomplete label coverage. Only 14 of the detector's 40 checks occur in the pilot's retained repair training, which supports 66 check names overall."),
        p("The pilot's saved settings confirm rank 16 on all seven projections, two epochs, learning rate 2e-4, 3,958 training rows, 124 validation rows, 6.5 minutes and 15.5 GB peak allocated GPU memory. These confirm a training run, not the correctness of generated repairs."),
        h("A layered validation contract"),
        table(["Layer", "Evidence required"], [
            ["Output and identity", "A complete, changed candidate with the expected function identity and permitted edit scope."],
            ["Original context", "Matching source hash, engine commit, tool version, configuration and compiler flags."],
            ["Compiler and diagnostics", "Reinsert through a temporary Clang virtual filesystem overlay; analyse original and candidate translation units."],
            ["Static acceptance", "Original requested findings confirmed; candidate removes all requested findings, with no candidate compiler errors or new enabled main-file warnings. Reject suppressions."],
            ["Behaviour and intent", "Independent tests, native engine builds and review are additional evidence beyond a static pass."],
        ], [131, 373]),
        p("Missing or changed context produces an inconclusive result. The virtual overlay leaves engine source files intact and preserves include resolution at the original path. Standalone syntax parsing is reported separately, because macros can affect snippet parsing."),
        h("What is verified in this snapshot"),
        p("The saved independent reference audit contains 128 cases: 100 recorded tool targets pass static validation and 0 unchanged-function controls pass. This shows that cached automatic edits are not uniformly valid under the later validator. It is a measurement of reference targets, not a learned-adapter success rate."),
        p("The saved behaviour result contains six hand-written fixtures with 36 assertions per system. Original and reference controls pass 6/6 cases; deliberately incorrect but compilable controls pass 0/6. Its adapter field is null, so it supplies no measured learned-repair behaviour score. The documented pilot_test and pipeline_test result directories are absent in this reviewed snapshot; no aggregate learned-repair success rate is asserted here."),
        source("Code/evidence: src/fixing_prompt.py; src/inference/fixing.py; repair_validation.py::RepairValidator; repair_behavior.py; data/fixing_pilot/reference_static_validation.jsonl; results/collected_fixing/behavior/metrics.json; docs/FIXING.md."),
    ]),
    ("11 / STUDY GUIDE AND SOURCES", "Connect each concept to the code", [
        table(["Read in this order", "Concrete learning target"], [
            ["1. detection_prompt.py and fixing_prompt.py", "Identify exactly what each task receives and returns."],
            ["2. collect/scan.py, extract.py, build_dataset.py", "Trace source code, findings, function spans, labels and splits."],
            ["3. training/train_lora.py::encode and collate", "Check token IDs, answer labels, masks and tensor shapes."],
            ["4. training/train_lora.py::main", "Locate LoRA setup, the loss, backpropagation and checkpoint selection."],
            ["5. evaluation/evaluate_detection.py", "Follow generation, parsing, check-set comparison and baselines."],
            ["6. evaluation/repair_validation.py and repair_behavior.py", "Distinguish static passes, behaviour tests and missing evidence."],
        ], [257, 247]),
        small("Paths above are relative to src/ except collect/, which is under scripts/. The local report builder records its evidence checks in output/reports/fundamentals_report_verification.json. Its JSON contains dataset counts, reproduced scores and run settings."),
        h("Primary technical references"),
        small('[R1] LLVM. <a href="https://clang.llvm.org/extra/clang-tidy/" color="#087F8C">Clang-Tidy documentation</a>. Static diagnostics, compile settings and automatic edits.'),
        small('[R2] Tree-sitter. <a href="https://tree-sitter.github.io/tree-sitter/" color="#087F8C">Introduction</a>. Concrete syntax trees and parsing.'),
        small('[R3] DeepSeek AI. <a href="https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct" color="#087F8C">DeepSeek-Coder-1.3B-Instruct model card</a>. Base checkpoint and intended use.'),
        small('[R4] Vaswani et al. (2017). <a href="https://arxiv.org/abs/1706.03762" color="#087F8C">Attention Is All You Need</a>. Attention and Transformer foundations.'),
        small('[R5] Hu et al. (2021). <a href="https://arxiv.org/abs/2106.09685" color="#087F8C">LoRA: Low-Rank Adaptation of Large Language Models</a>. Frozen base weights and low-rank updates.'),
        small('[R6] Dettmers et al. (2023). <a href="https://arxiv.org/abs/2305.14314" color="#087F8C">QLoRA: Efficient Finetuning of Quantized LLMs</a>. Quantised base models with LoRA.'),
        small('[R7] Hugging Face. <a href="https://huggingface.co/docs/transformers/chat_templating" color="#087F8C">Chat templates</a>. Consistent conversation formatting.'),
        small('[R8] PyTorch. <a href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html" color="#087F8C">CrossEntropyLoss documentation</a>. Ignored targets and cross-entropy.'),
        small('[R9] PyTorch. <a href="https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html" color="#087F8C">AdamW documentation</a>. Adaptive optimisation and weight decay.'),
        small("External references were checked for background concepts. Live documentation may describe versions newer than the recorded training environment; repository code and saved settings govern claims about this implementation."),
        box("The fundamental techniques in one sentence", "This project combines compiler-assisted weak supervision, syntax-tree data extraction, controlled dataset construction, pretrained Transformer adaptation with LoRA, answer-only supervised learning, and evaluation that separates label agreement from repair correctness."),
    ]),
    ]


class Pipeline(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 103

    def draw(self):
        c = self.canv
        labels = [("Engine source", "clang-tidy + parser"), ("Labelled data", "engine-held-out splits"),
                  ("LoRA training", "answer-only loss"), ("Local inference", "detect / repair")]
        bw, gap = 117, 12
        for i, (a, b) in enumerate(labels):
            x = i * (bw + gap)
            c.setFillColor(LIGHT); c.setStrokeColor(LINE)
            c.roundRect(x, 35, bw, 58, 7, fill=1, stroke=1)
            c.setFillColor(NAVY); c.setFont("Body-Bold", 10)
            c.drawCentredString(x + bw / 2, 70, a)
            c.setFillColor(GRAY); c.setFont("Body", 8)
            c.drawCentredString(x + bw / 2, 53, b)
            if i < 3:
                c.setStrokeColor(TEAL); c.setLineWidth(1.3)
                c.line(x + bw + 2, 64, x + bw + gap - 2, 64)
                c.line(x + bw + gap - 5, 67, x + bw + gap - 2, 64)
                c.line(x + bw + gap - 5, 61, x + bw + gap - 2, 64)
        c.setFont("Body", 8.5); c.setFillColor(TEAL)
        c.drawCentredString(WIDTH / 2, 13, "Evaluate saved predictions  |  Validate candidate repairs independently")


def styles():
    fonts = Path("C:/Windows/Fonts")
    for name, path in [("Body", "arial.ttf"), ("Body-Bold", "arialbd.ttf"),
                       ("Body-Italic", "ariali.ttf"), ("Body-BoldItalic", "arialbi.ttf"), ("Code", "consola.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / path)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
    return {
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.1, leading=14.2, textColor=INK, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8.65, leading=12.2, textColor=GRAY, spaceAfter=7),
        "source": ParagraphStyle("source", fontName="Body", fontSize=8.0, leading=11.2, textColor=GRAY, spaceBefore=6),
        "kicker": ParagraphStyle("kicker", fontName="Body-Bold", fontSize=9, leading=12, textColor=TEAL, spaceAfter=8),
        "title": ParagraphStyle("title", fontName="Body-Bold", fontSize=23, leading=27, textColor=NAVY, spaceAfter=17),
        "cover": ParagraphStyle("cover", fontName="Body-Bold", fontSize=31, leading=34.5, textColor=NAVY, spaceAfter=13),
        "subtitle": ParagraphStyle("subtitle", fontName="Body", fontSize=14, leading=19, textColor=TEAL, spaceAfter=13),
        "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.1, leading=16, textColor=NAVY, spaceBefore=7, spaceAfter=7, keepWithNext=True),
        "code": ParagraphStyle("code", fontName="Code", fontSize=8.5, leading=11.5, textColor=NAVY, backColor=LIGHT,
                               borderPadding=9, spaceBefore=3, spaceAfter=12),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=9.0, leading=12.0, textColor=INK),
        "th": ParagraphStyle("th", fontName="Body-Bold", fontSize=9, leading=12.5, textColor=colors.white),
        "box": ParagraphStyle("box", fontName="Body", fontSize=9.3, leading=13.1, textColor=NAVY),
    }


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, total):
        super().__init__(str(filename), pagesize=letter, leftMargin=54, rightMargin=54,
                         topMargin=59, bottomMargin=48, title="Fundamental Knowledge and Techniques - DeepSeek Fine-Tuning",
                         author="", subject="Repository-grounded technical report", allowSplitting=1)
        self.total = total
        self.addPageTemplates(PageTemplate(id="report", frames=[Frame(54, 48, WIDTH, 698, leftPadding=0,
                              bottomPadding=0, rightPadding=0, topPadding=0)], onPage=self.decorate))

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"title", "cover"}:
            key = f"section-{self.page}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0)

    def decorate(self, c, doc):
        c.saveState()
        c.setStrokeColor(LINE); c.setLineWidth(0.6)
        c.line(54, 753, 558, 753); c.line(54, 37, 558, 37)
        c.setFillColor(GRAY); c.setFont("Body", 8)
        c.drawString(54, 764, "C-FineTuneDeepSeek / Technical Report")
        c.drawRightString(558, 764, REPORT_DATE)
        c.drawString(54, 24, "Fundamental knowledge and techniques")
        c.drawRightString(558, 24, f"{doc.page} / {self.total}")
        c.restoreState()


def render_pdf(pages):
    s = styles()
    story = []
    for index, (kicker, title, blocks) in enumerate(pages):
        if index: story.append(PageBreak())
        story += [Paragraph(kicker, s["kicker"]), Paragraph(title.replace("\n", "<br/>"), s["cover" if index == 0 else "title"])]
        for b in blocks:
            kind = b[0]
            if kind in ("p", "h", "small", "source", "subtitle"):
                story.append(Paragraph(b[1], s["body" if kind == "p" else kind]))
            elif kind == "code":
                story.append(Preformatted(b[1], s["code"]))
            elif kind == "pipeline":
                story.append(Pipeline())
            elif kind == "box":
                inner = Paragraph(f"<b>{b[1]}</b><br/>{b[2]}", s["box"])
                t = Table([[inner]], colWidths=[WIDTH])
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), .5, LINE),
                                      ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL), ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                      ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 10),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
                story.extend([Spacer(1, 3), t, Spacer(1, 10)])
            elif kind == "table":
                cells = [[Paragraph(escape(v), s["th"]) for v in b[1]]]
                cells += [[Paragraph(escape(str(v)), s["cell"]) for v in row] for row in b[2]]
                t = Table(cells, colWidths=b[3], repeatRows=1, hAlign="LEFT")
                commands = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                            ("LINEBELOW", (0, 0), (-1, 0), .5, NAVY)]
                for row in range(1, len(cells)):
                    commands += [("BACKGROUND", (0, row), (-1, row), LIGHT if row % 2 else colors.white),
                                 ("LINEBELOW", (0, row), (-1, row), .4, LINE)]
                t.setStyle(TableStyle(commands))
                story.extend([t, Spacer(1, 10)])
    ReportDoc(PDF, len(pages)).build(story)


def render_html(pages):
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Project Fundamentals and Techniques</title>',
            '<style>body{font:16px/1.6 Arial,sans-serif;color:#243548;background:#eef3f6;margin:0}main{max-width:900px;margin:auto}section{background:white;padding:48px;margin:24px 0;box-shadow:0 2px 12px #142b4410}h1{font-size:34px;line-height:1.15;color:#132b44}h2{font-size:21px;color:#132b44}.kicker,a{color:#087f8c}.kicker{font-size:13px;font-weight:bold}.small,.source{font-size:13px;color:#536475}.source{border-top:1px solid #d7e2ea;padding-top:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:12px;vertical-align:top;border-bottom:1px solid #d7e2ea}th{background:#132b44;color:white}tr:nth-child(even){background:#f0f5f8}pre{background:#f0f5f8;padding:16px;overflow:auto;font:13px/1.6 Consolas,monospace}.box{background:#f0f5f8;border-left:4px solid #087f8c;padding:16px;margin:18px 0}.pipeline{padding:18px;background:#f0f5f8;color:#087f8c;text-align:center}.subtitle{font-size:21px;color:#087f8c}@media print{body{background:white}section{margin:0;padding:0;box-shadow:none;break-after:page}main{max-width:none}}</style><main>']
    for kicker, title, blocks in pages:
        html.append(f'<section><div class="kicker">{escape(kicker)}</div><h1>{escape(title).replace(chr(10), "<br>")}</h1>')
        for b in blocks:
            if b[0] in ("p", "h", "small", "source", "subtitle"):
                tag = "h2" if b[0] == "h" else "p"
                content = b[1].replace("<super>", "<sup>").replace("</super>", "</sup>")
                content = content.replace(' color="#087F8C"', '')
                html.append(f'<{tag} class="{b[0]}">{content}</{tag}>')
            elif b[0] == "code": html.append("<pre>" + escape(b[1]) + "</pre>")
            elif b[0] == "box": html.append(f'<div class="box"><b>{escape(b[1])}</b><br>{b[2]}</div>')
            elif b[0] == "pipeline": html.append('<div class="pipeline">Engine source &rarr; Labelled data &rarr; LoRA training &rarr; Local inference<br>Evaluate predictions and independently validate repairs</div>')
            elif b[0] == "table":
                html.append('<table><thead><tr>' + ''.join('<th>'+escape(x)+'</th>' for x in b[1]) + '</tr></thead><tbody>')
                html.extend('<tr>' + ''.join('<td>'+escape(str(x))+'</td>' for x in row) + '</tr>' for row in b[2])
                html.append('</tbody></table>')
        html.append('</section>')
    html.append('</main></html>')
    HTML.write_text("\n".join(html), encoding="utf-8")


def main():
    evidence = verify_evidence()
    for path in (PDF, HTML, CHECKS): path.parent.mkdir(parents=True, exist_ok=True)
    pages = sections(evidence)
    # Baseline values in the prose must be the verified stored values.
    stored = load("results/collected_detection/lora_r16_e3/test/metrics.json")["results"]["all"]
    print("Verified baselines:", {n:{k:v for k,v in m.items() if k in ('exact','smelly_accuracy','micro_f1')} for n,m in stored.items()})
    render_pdf(pages)
    render_html(pages)
    from pypdf import PdfReader
    reader = PdfReader(str(PDF))
    assert len(reader.pages) == len(pages), f"Expected {len(pages)} pages; got {len(reader.pages)}"
    for page, (kicker, title, _) in zip(reader.pages, pages):
        text = page.extract_text()
        assert kicker in text, f"Page overflow: expected {kicker}"
        assert '\ufffd' not in text
    evidence.update(pdf_pages=len(reader.pages), pdf_bytes=PDF.stat().st_size,
                    outputs=[str(PDF),str(HTML)])
    CHECKS.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({"pdf": str(PDF), "html": str(HTML), "pages":len(reader.pages), "verification":str(CHECKS)}, indent=2))


if __name__ == "__main__":
    main()
