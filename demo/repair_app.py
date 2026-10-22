"""Local repair lab: candidate patch, diff, and independent static validation."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
from src.inference.fixing import DEFAULT_ADAPTER, Fixer, repair_diff
from src.inference.detection_demo import Detector


def examples():
    rows = [json.loads(line) for line in (ROOT / "data/fixing_pilot/test.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: ("readability-else-after-return" not in r["fixed_checks"], r["n_tokens"]))
    return {f"{r['engine']} · {r['name']} · {r['id']}": r for r in rows}


def values(row):
    return (row["messages"][1]["content"].split("\n\n", 1)[1], ", ".join(row["fixed_checks"]), row,
            f"`{row['path']}`:{row['start_line']} · Held-out engine · Reference checks from clang-tidy")


def build_app(fixer):
    presets = examples()
    first = next(iter(presets))
    initial_code, initial_checks, initial_row, initial_info = values(presets[first])
    detector = Detector()
    with gr.Blocks(title="DeepSeek repair lab") as app:
        gr.Markdown("# DeepSeek repair lab\nGenerate a candidate fix, inspect its diff, and re-run clang-tidy in the original file context.")
        gr.Markdown("**Pilot model.** A static pass means the requested warnings disappeared without new enabled warnings or compiler errors. "
                    "It does not establish behavioral correctness. No engine source file is modified.")
        preset = gr.Dropdown(list(presets), value=first, label="Held-out function")
        provenance = gr.Markdown(initial_info)
        row = gr.State(initial_row)
        mode = gr.Radio(["Reference checks (conditional repair)", "Current detector predictions (end-to-end experiment)"],
                        value="Reference checks (conditional repair)", label="Where the requested checks come from")
        with gr.Row():
            before = gr.Code(initial_code, language="cpp", label="Original function", lines=15, interactive=True)
            after = gr.Code(language="cpp", label="Generated candidate", lines=15, interactive=False)
        checks = gr.Textbox(initial_checks, label="Requested checks (comma separated; used in reference mode)")
        run = gr.Button("Generate and validate repair", variant="primary")
        status = gr.Markdown("Ready. First use loads the repair model.")
        patch = gr.Code(label="Candidate diff", lines=12, interactive=False)
        with gr.Accordion("Validation evidence", open=False):
            validation = gr.JSON(label="Full static-analysis report")
        with gr.Accordion("Inference details", open=False):
            trace = gr.JSON(label="Generation and requested checks")
        with gr.Accordion("Exact repair prompt", open=False):
            prompt = gr.Textbox(lines=12, interactive=False)
        output_components = [after, status, patch, validation, trace, prompt]

        def reset():
            return "", "Input changed. Generate a new candidate.", "", None, None, ""

        preset.change(lambda name: (*values(presets[name]), *reset()), preset,
                      [before, checks, row, provenance, *output_components])
        before.input(reset, outputs=output_components)
        checks.input(reset, outputs=output_components)
        mode.change(reset, outputs=output_components)

        def repair(code, check_text, source_mode, source_row):
            yield "", "Loading / generating candidate…", "", None, None, ""
            try:
                requested = sorted({c.strip() for c in check_text.split(",") if c.strip()})
                detection = None
                if source_mode.startswith("Current detector"):
                    detection = list(detector.stream(code))[-1]
                    if not detection["valid"]:
                        raise ValueError("Detector returned invalid JSON; no repair was attempted.")
                    if detection.get("stopped_at_limit"):
                        raise ValueError("Detector output reached its token limit; no repair was attempted.")
                    requested = detection["parsed"]["checks"]
                    if not requested:
                        yield code, "Detector requested no fixes. This does not establish that the function is clean.", "", \
                            {"status": "not_validated", "behavioral_correctness": "not_tested"}, {"detected_checks": []}, ""
                        return
                result = fixer.repair(code, requested)
                metadata = {k: v for k, v in result.items() if k not in {"candidate", "raw", "prompt", "diff"}}
                if detection:
                    metadata["detected_checks"] = detection["parsed"]["checks"]
                yield result["candidate"], "Candidate generated; validating in source context…", result["diff"], None, metadata, result["prompt"]
                if result["truncated"]:
                    evidence = {"status": "rejected", "reason": "Generation hit its output limit.", "behavioral_correctness": "not_tested"}
                elif code != values(source_row)[0]:
                    evidence = {"status": "inconclusive", "reason": "Edited input has no matching full-source context.",
                                "syntax": result["syntax"], "behavioral_correctness": "not_tested"}
                else:
                    from src.evaluation.repair_validation import RepairValidator
                    requested_row = {**source_row, "fixed_checks": requested}
                    evidence = RepairValidator().validate(requested_row, result["candidate"])
                yield result["candidate"], f"Validation: **{evidence['status']}**. Behavioral correctness is not established.", \
                    result["diff"], evidence, metadata, result["prompt"]
            except Exception as error:
                yield "", f"Could not complete repair: {error}", "", {"status": "error"}, None, ""

        controls = [preset, before, checks, mode, run]
        disable = run.click(lambda: [gr.update(interactive=False) for _ in controls],
                            outputs=controls, queue=False, api_name=False)
        job = disable.then(repair, [before, checks, mode, row], output_components,
                           concurrency_limit=1, api_name="repair")
        job.then(lambda: [gr.update(interactive=True) for _ in controls],
                 outputs=controls, queue=False, api_name=False)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    build_app(Fixer(args.adapter)).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False,
                                                inbrowser=False, show_error=True)
