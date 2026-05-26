#!/usr/bin/env python3
"""
preprocess_data.py — LexiLingo training data cleaner
=====================================================
Fixes 4 data quality issues found in train.jsonl / val.jsonl:

  1. [VOCABULARY] 71.2% samples missing `key_words` → add `"key_words": []`
  2. [VOCABULARY] Some samples have `key_words` as string (metadata bleed) → replace with `[]`
  3. [FLUENCY]    8.4% samples have inconsistent `reasoning` → remove it
  4. [DEDUP]      ~3.7% near-duplicate user inputs per task → deduplicate

Input : training_data/train.jsonl, training_data/val.jsonl
Output: training_data_clean/train_clean.jsonl, val_clean.jsonl
        training_data_clean/clean_report.json

Original files are NOT modified.
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ─── Config ────────────────────────────────────────────────────────────────
INPUT_DIR  = Path(__file__).parent / "training_data"
OUTPUT_DIR = Path(__file__).parent / "training_data_clean"

SPLITS = [
    ("train.jsonl", "train_clean.jsonl"),
    ("val.jsonl",   "val_clean.jsonl"),
]


# ─── Helpers ───────────────────────────────────────────────────────────────
def get_messages(ex: dict) -> list:
    return [m for m in ex.get("messages", []) if isinstance(m, dict)]


def get_assistant(ex: dict) -> dict | None:
    for m in get_messages(ex):
        if m.get("role") == "assistant":
            return m
    return None


def get_user_content(ex: dict) -> str:
    for m in get_messages(ex):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def dedup_key(ex: dict) -> tuple:
    """Deduplication key: (task, first 200 chars of user content)."""
    return (ex.get("task", ""), get_user_content(ex)[:200])


# ─── Fix functions ─────────────────────────────────────────────────────────
def fix_vocabulary(ex: dict, stats: dict) -> dict:
    """Add key_words: [] when missing, or fix to [] when value is not a list."""
    asst = get_assistant(ex)
    if asst is None:
        return ex
    try:
        obj = json.loads(asst["content"])
    except (json.JSONDecodeError, KeyError):
        return ex

    changed = False
    if "key_words" not in obj:
        obj["key_words"] = []
        stats["vocab_key_words_added"] += 1
        changed = True
    elif not isinstance(obj["key_words"], list):
        obj["key_words"] = []
        stats["vocab_key_words_type_fixed"] += 1
        changed = True

    if changed:
        asst = dict(asst, content=json.dumps(obj, ensure_ascii=False))
        msgs = [asst if m.get("role") == "assistant" else m for m in get_messages(ex)]
        ex = dict(ex, messages=msgs)

    return ex


def fix_fluency(ex: dict, stats: dict) -> dict:
    """Remove reasoning field to enforce consistent schema."""
    asst = get_assistant(ex)
    if asst is None:
        return ex
    try:
        obj = json.loads(asst["content"])
    except (json.JSONDecodeError, KeyError):
        return ex

    if "reasoning" in obj:
        obj.pop("reasoning")
        asst = dict(asst, content=json.dumps(obj, ensure_ascii=False))
        msgs = [asst if m.get("role") == "assistant" else m for m in get_messages(ex)]
        ex = dict(ex, messages=msgs)
        stats["fluency_reasoning_removed"] += 1

    return ex


# ─── Main pipeline ─────────────────────────────────────────────────────────
def clean_split(in_path: Path, out_path: Path, seen_keys: set, is_train: bool) -> dict:
    """
    Clean one split file.
    seen_keys is shared across calls so val never contains keys from train.
    Returns per-split stats dict.
    """
    stats = defaultdict(int)
    out_lines = []

    for raw_line in in_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            ex = json.loads(raw_line)
        except json.JSONDecodeError:
            stats["json_parse_error"] += 1
            continue

        stats["total_in"] += 1
        task = ex.get("task", "unknown")
        stats[f"task_{task}_in"] += 1

        # 1. Dedup (train only — val is too small to matter, but we still check
        #    against train keys to avoid leakage)
        key = dedup_key(ex)
        if key in seen_keys:
            stats["dedup_removed"] += 1
            continue
        seen_keys.add(key)

        # 2. Task-specific fixes
        if task == "vocabulary":
            ex = fix_vocabulary(ex, stats)
        elif task == "fluency":
            ex = fix_fluency(ex, stats)

        out_lines.append(json.dumps(ex, ensure_ascii=False))
        stats["total_out"] += 1
        stats[f"task_{task}_out"] += 1

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dict(stats)


def run(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    # seen_keys shared: train first, then val (prevents leakage into val)
    seen_keys: set = set()

    for in_name, out_name in SPLITS:
        in_path  = input_dir  / in_name
        out_path = output_dir / out_name

        if not in_path.exists():
            print(f"[SKIP] {in_path} not found")
            continue

        is_train = "train" in in_name
        print(f"\n{'─'*60}")
        print(f"Processing: {in_name} → {out_name}")
        stats = clean_split(in_path, out_path, seen_keys, is_train)

        report["splits"][in_name] = stats

        # Pretty-print per-split summary
        removed = stats["total_in"] - stats["total_out"]
        print(f"  Input       : {stats['total_in']:,}")
        print(f"  Output      : {stats['total_out']:,}  (removed {removed:,})")
        print(f"  Dedup rm'd  : {stats['dedup_removed']:,}")
        print(f"  vocab fix   : {stats['vocab_key_words_added']:,} key_words added  |  {stats['vocab_key_words_type_fixed']:,} type fixed (str→list)")
        print(f"  fluency fix : {stats['fluency_reasoning_removed']:,} reasoning removed")
        task_keys = [k for k in stats if k.startswith("task_") and k.endswith("_out")]
        print(f"  Tasks out   : { {k.replace('task_','').replace('_out',''): stats[k] for k in task_keys} }")

    report_path = output_dir / "clean_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'─'*60}")
    print(f"Report saved → {report_path}")
    print(f"Clean files  → {output_dir}")


# ─── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LexiLingo data preprocessor")
    parser.add_argument("--input-dir",  default=str(INPUT_DIR),  help="Folder with train.jsonl/val.jsonl")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Folder to write clean files")
    args = parser.parse_args()

    run(Path(args.input_dir), Path(args.output_dir))
