"""Print the settings stored in `training_args.bin` files without unpickling them.

The files are pickles; this only disassembles them, so nothing is executed.

    python scripts/audit/read_training_args.py models/detection/training_args.bin ...

With no arguments it reads every `models/*/training_args.bin`.
"""
import io
import pickletools
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WANT = ["output_dir", "max_length", "max_seq_length", "per_device_train_batch_size",
        "gradient_accumulation_steps", "learning_rate", "num_train_epochs", "completion_only_loss"]
VALUE = re.compile(r"\d+: \S\s+(BININT\d?|BINFLOAT|LONG1|NONE|NEWTRUE|NEWFALSE|BINUNICODE|SHORT_BINUNICODE)\s*(.*)")


def read(path):
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    out = io.StringIO()
    pickletools.dis(z.read(name), out=out)
    lines = out.getvalue().splitlines()
    res = {}
    for i, line in enumerate(lines):
        m = re.search(r"UNICODE\s+'(\w+)'", line)
        if not (m and m.group(1) in WANT and m.group(1) not in res):
            continue
        # the value is pushed right after the attribute name
        for nxt in lines[i + 1:i + 4]:
            v = VALUE.search(nxt)
            if v:
                res[m.group(1)] = v.group(2).strip().strip("'") or v.group(1)
                break
    return res


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted((ROOT / "models").glob("*/training_args.bin"))
    for p in paths:
        try:
            print(p, "\n   ", read(p))
        except Exception as e:
            print(p, "ERR", e)
