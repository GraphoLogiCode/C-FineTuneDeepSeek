"""Shared helpers for the collection pipeline (scan -> extract -> build)."""
import hashlib
import json
import os
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("collect_config.json")

SIZE_CHECK = "readability-function-size"
COGNITIVE_CHECK = "readability-function-cognitive-complexity"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def engine_root(cfg, engine):
    return Path(cfg["engines_root"]) / engine


def work_dir(cfg, engine):
    d = ROOT / cfg["work_dir"] / engine
    d.mkdir(parents=True, exist_ok=True)
    return d


def selected_engines(cfg, names):
    """The engines named on the command line, or every configured engine that is on disk."""
    if names:
        unknown = [n for n in names if n not in cfg["engines"]]
        if unknown:
            sys.exit(f"unknown engine(s): {', '.join(unknown)}; configured: {', '.join(cfg['engines'])}")
        return names
    return [e for e in cfg["engines"] if engine_root(cfg, e).is_dir()]


def is_excluded(rel_posix, cfg, engine):
    """True for vendored, test, sample and tool code that should not be labelled."""
    lowered = rel_posix.lower()
    names = set(cfg["exclude_dir_names"])
    if any(part in names for part in lowered.split("/")[:-1]):
        return True
    return any(lowered.startswith(p.lower()) for p in cfg["engines"][engine].get("exclude", []))


def walk_files(root, extensions):
    """Every file under root with one of the extensions, as (absolute path, posix path relative to root)."""
    exts = {e.lower() for e in extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                full = Path(dirpath) / name
                yield full, full.relative_to(root).as_posix()


def source_files(cfg, engine):
    """The files to analyse: engine code only, sorted for reproducibility."""
    root = engine_root(cfg, engine)
    limit = cfg["max_file_kb"] * 1024
    out = []
    for full, rel in walk_files(root, cfg["source_extensions"]):
        if not is_excluded(rel, cfg, engine) and full.stat().st_size <= limit:
            out.append((full, rel))
    return sorted(out, key=lambda x: x[1])


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def md5_text(s):
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()


def norm_code(s):
    return " ".join(s.split())


def matches_any(name, patterns):
    return any(fnmatchcase(name, p) for p in patterns)


def utf8_stdout():
    """Windows consoles default to a legacy code page; source code needs UTF-8."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
