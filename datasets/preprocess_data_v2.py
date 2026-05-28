#!/usr/bin/env python3
"""
preprocess_data_v2.py - LexiLingo clean-v2 dataset builder.

Clean-v2 is intentionally less aggressive than the first cleaner:

* keep the original train/val/test split shape;
* remove exact/near duplicate prompts across splits to avoid leakage;
* keep richer assistant targets instead of collapsing outputs to tiny labels;
* repair vocabulary key_words into real keyword lists;
* keep fluency feedback/reasoning as a supervised target;
* add high-quality Vietnamese tutor explanations with train/val/test split;
* cap only the most repetitive train-side fluency scores;
* audit downloaded_datasets as a candidate pool without merging by default.

Original raw files are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "training_data"
DEFAULT_DOWNLOADED_DIR = ROOT_DIR.parent / "scripts" / "downloaded_datasets"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "training_data_clean_v2"
DEFAULT_EXPLANATION_SOURCE = DEFAULT_INPUT_DIR / "vietnamese_explanations.jsonl"

SPLITS = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}

COMPAT_OUTPUT_NAMES = {
    "train": "train_clean.jsonl",
    "validation": "val_clean.jsonl",
    "test": "test_clean.jsonl",
}

LEGACY_UNUSED_DATA_FILES = [
    "train_clean_v2.jsonl",
    "val_clean_v2.jsonl",
    "test_clean_v2.jsonl",
    "downloaded_candidates_clean_v2.jsonl",
]

ALLOWED_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}

STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "before",
    "being", "below", "between", "could", "does", "doing", "during", "each",
    "from", "further", "have", "having", "here", "hers", "himself", "into",
    "itself", "just", "more", "most", "other", "over", "same", "should",
    "some", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "until", "very", "were",
    "what", "when", "where", "which", "while", "with", "would", "your",
    "will", "shall", "been", "only", "many", "much", "make", "made", "used",
    "using", "text", "level", "real", "world", "heuristic", "source", "wiki",
    "wikitext", "unknown", "sentence", "answer", "question", "options",
}


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("@-@", "-").replace("<unk>", " ")
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def get_messages(ex: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in ex.get("messages", []) if isinstance(m, dict)]


def get_user_content(ex: dict[str, Any]) -> str:
    if "input" in ex and isinstance(ex.get("input"), str):
        return ex["input"]
    for msg in get_messages(ex):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def get_output_obj(ex: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(ex.get("output"), dict):
        return dict(ex["output"])
    for msg in get_messages(ex):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                obj = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                return None
            return obj if isinstance(obj, dict) else None
    return None


def get_assistant_content(ex: dict[str, Any]) -> str:
    for msg in get_messages(ex):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def get_output_payload(ex: dict[str, Any]) -> Any:
    if "output" in ex:
        return ex.get("output")

    content = get_assistant_content(ex)
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def dedup_keys(task: str, user_text: str) -> tuple[str, ...]:
    normalized = normalize_text(user_text)
    keys = [f"{task}:full:{short_hash(normalized)}"]

    # Long dialogue/instruction prompts often share the same opening task
    # definition while the actual instance appears later. Prefix-only keys
    # create false positives there, so only use this extra key for short text.
    if len(normalized) <= 320:
        keys.append(f"{task}:short:{normalized}")
    return tuple(keys)


def first_dedup_key(task: str, user_text: str) -> str:
    return dedup_keys(task, user_text)[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")


def clean_keyword_token(token: str) -> str:
    token = token.replace("@-@", "-").strip(" \t\r\n.,;:!?()[]{}\"'")
    token = re.sub(r"[^A-Za-z0-9'/-]+", "", token)
    return token


def parse_keyword_string(value: str) -> list[str]:
    lower = value.lower()
    metadata_markers = ("heuristic level", "real-world text", "wikitext", "source=")
    if any(marker in lower for marker in metadata_markers):
        return []

    pieces = re.split(r"[,;|/]+", value)
    keywords = []
    seen = set()
    for piece in pieces:
        token = clean_keyword_token(piece)
        norm = normalize_text(token)
        if not token or len(norm) < 3 or norm in STOPWORDS or norm in seen:
            continue
        seen.add(norm)
        keywords.append(token[:40])
    return keywords[:8]


def extract_keywords_from_text(text: str, limit: int = 6) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'/-]{2,}", text.replace("@-@", "-"))
    candidates = []
    seen = set()
    for token in raw_tokens:
        token = clean_keyword_token(token)
        norm = normalize_text(token)
        if len(norm) < 4 or norm in STOPWORDS or norm in seen:
            continue
        if norm.isdigit() or norm == "unk":
            continue
        seen.add(norm)
        candidates.append(token)

    def score(tok: str) -> tuple[int, int]:
        norm = normalize_text(tok)
        rare_bonus = 1 if len(norm) >= 8 else 0
        proper_bonus = 1 if tok[:1].isupper() else 0
        return (rare_bonus + proper_bonus, len(norm))

    ranked = sorted(candidates, key=score, reverse=True)
    selected = ranked[:limit]
    selected_set = {normalize_text(tok) for tok in selected}
    ordered = [tok for tok in candidates if normalize_text(tok) in selected_set]
    return ordered[:limit]


def normalize_level(value: Any, metadata: dict[str, Any]) -> str | None:
    candidates = [
        value,
        metadata.get("estimated_level"),
        metadata.get("level"),
        metadata.get("cefr"),
    ]
    for item in candidates:
        if item is None:
            continue
        match = re.search(r"\b(A1|A2|B1|B2|C1|C2)\b", str(item).upper())
        if match and match.group(1) in ALLOWED_LEVELS:
            return match.group(1)
    return None


def normalize_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    score = max(0.0, min(1.0, score))
    return round(score, 2)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return stable_json(value)


def safe_int(value: Any, default: int = -1) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_metadata(metadata: dict[str, Any], split_name: str, dedup_key: str) -> dict[str, Any]:
    """Return a fixed, flat metadata schema so HuggingFace JSON loading can infer safely."""
    return {
        "source": safe_str(metadata.get("source")),
        "index": safe_int(metadata.get("index")),
        "raw_text": safe_str(metadata.get("raw_text")),
        "raw_response": safe_str(metadata.get("raw_response")),
        "file": safe_str(metadata.get("file")),
        "type": safe_str(metadata.get("type")),
        "estimated_level": safe_str(metadata.get("estimated_level")),
        "context_sentences": safe_int(metadata.get("context_sentences")),
        "word_count": safe_int(metadata.get("word_count")),
        "grammatical": bool(metadata.get("grammatical")) if metadata.get("grammatical") is not None else False,
        "has_grammatical": metadata.get("grammatical") is not None,
        "error_count": safe_int(metadata.get("error_count")),
        "error_type": safe_str(metadata.get("error_type")),
        "quality_score": safe_int(metadata.get("quality_score")),
        "clean_v2_split": split_name,
        "clean_v2_dedup_key": dedup_key,
    }


def default_fluency_feedback(score: float, metadata: dict[str, Any], user_text: str) -> str:
    words = re.findall(r"[A-Za-z]+", user_text)
    word_count = len(words)
    if word_count < 12:
        length = "Short"
    elif word_count < 60:
        length = "Medium-length"
    else:
        length = "Long-form"

    source = str(metadata.get("source", "")).lower()
    text_type = str(metadata.get("type", "")).lower()
    if metadata.get("grammatical") is False:
        style = "learner-like"
    elif "cnn" in source or "news" in text_type:
        style = "news-style"
    elif "wiki" in source:
        style = "encyclopedic"
    elif "cola" in source:
        style = "sentence-level"
    elif "openorca" in source or "anthropic" in source:
        style = "instructional"
    else:
        style = "general"

    if score < 0.7:
        quality = "needs clearer grammar and more natural phrasing"
    elif score < 0.85:
        quality = "is understandable but somewhat uneven"
    elif score < 0.93:
        quality = "is fluent overall with minor wording issues"
    else:
        quality = "reads very naturally with clear structure"

    return f"{length} {style} text; {quality}."


def clean_output(
    task: str,
    user_text: str,
    obj: Any,
    metadata: dict[str, Any],
    stats: Counter,
) -> dict[str, Any] | None:
    if task == "explanation":
        if isinstance(obj, dict):
            explanation = (
                obj.get("explanation")
                or obj.get("output")
                or obj.get("response")
                or obj.get("answer")
            )
            error_type = obj.get("error_type") or metadata.get("error_type")
        else:
            explanation = obj
            error_type = metadata.get("error_type")

        explanation_text = str(explanation or "").strip()
        if not explanation_text:
            stats["drop_explanation_missing_text"] += 1
            return None

        output = {"explanation": explanation_text}
        if error_type:
            output["error_type"] = str(error_type)
        return output

    if not isinstance(obj, dict):
        stats[f"drop_{task}_non_json_output"] += 1
        return None

    if task == "dialogue":
        response = obj.get("response")
        if response is None:
            response = obj.get("answer")
        if response is None:
            stats["drop_dialogue_missing_response"] += 1
            return None
        return {"response": str(response).strip()}

    if task == "grammar":
        corrected = obj.get("corrected")
        if corrected is None:
            corrected = obj.get("correction")
        if corrected is None:
            stats["drop_grammar_missing_corrected"] += 1
            return None
        explanation = obj.get("explanation") or obj.get("feedback")
        if not explanation:
            explanation = "Learner sentence corrected."
            stats["grammar_explanation_added"] += 1
        return {
            "corrected": str(corrected).strip(),
            "explanation": str(explanation).strip(),
        }

    if task == "vocabulary":
        level = normalize_level(obj.get("level"), metadata)
        if level is None:
            stats["drop_vocabulary_missing_level"] += 1
            return None

        key_words = obj.get("key_words")
        keywords: list[str] = []
        if isinstance(key_words, list):
            for item in key_words:
                token = clean_keyword_token(str(item))
                norm = normalize_text(token)
                if token and len(norm) >= 3 and norm not in STOPWORDS:
                    keywords.append(token[:40])
        elif isinstance(key_words, str):
            keywords = parse_keyword_string(key_words)
            if not keywords:
                stats["vocabulary_metadata_keywords_replaced"] += 1
        else:
            stats["vocabulary_keywords_added"] += 1

        if not keywords:
            keywords = extract_keywords_from_text(user_text)
            stats["vocabulary_keywords_extracted"] += 1
        if not keywords:
            keywords = [level]
            stats["vocabulary_keyword_fallback_level"] += 1

        deduped = []
        seen = set()
        for token in keywords:
            norm = normalize_text(token)
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(token)
        return {"level": level, "key_words": deduped[:8]}

    if task == "fluency":
        score = normalize_score(obj.get("fluency_score") or obj.get("score"))
        if score is None:
            stats["drop_fluency_missing_score"] += 1
            return None
        feedback = obj.get("feedback") or obj.get("reasoning") or obj.get("explanation")
        if not feedback:
            feedback = default_fluency_feedback(score, metadata, user_text)
            stats["fluency_feedback_added"] += 1
        else:
            stats["fluency_reasoning_kept_as_feedback"] += 1
        return {"fluency_score": score, "feedback": str(feedback).strip()}

    stats["drop_unknown_task"] += 1
    return None


def make_chat_record(
    task: str,
    user_text: str,
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "messages": [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": stable_json(output)},
        ],
        "metadata": metadata,
    }


def clean_records(
    split_name: str,
    raw_rows: list[dict[str, Any]],
    seen_dedup: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats: Counter = Counter()
    cleaned = []

    for raw in raw_rows:
        stats["total_in"] += 1
        task = str(raw.get("task", "")).strip()
        user_text = get_user_content(raw).strip()
        metadata = dict(raw.get("metadata") or {})
        obj = get_output_payload(raw)

        if not task or not user_text:
            stats["drop_missing_task_or_input"] += 1
            continue
        if obj is None:
            stats["drop_missing_output"] += 1
            continue

        stats[f"task_{task}_in"] += 1
        keys = dedup_keys(task, user_text)
        if any(key in seen_dedup for key in keys):
            stats["dedup_removed"] += 1
            stats[f"task_{task}_dedup_removed"] += 1
            continue
        seen_dedup.update(keys)

        output = clean_output(task, user_text, obj, metadata, stats)
        if output is None:
            continue

        metadata = normalize_metadata(metadata, split_name, keys[0])
        cleaned.append(make_chat_record(task, user_text, output, metadata))
        stats["total_out"] += 1
        stats[f"task_{task}_out"] += 1

    return cleaned, dict(stats)


def cap_train_fluency_scores(
    rows: list[dict[str, Any]],
    max_per_score: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if max_per_score <= 0:
        return rows, {}

    score_to_indices: dict[float, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if row.get("task") != "fluency":
            continue
        obj = get_output_obj(row) or {}
        score = normalize_score(obj.get("fluency_score"))
        if score is not None:
            score_to_indices[score].append(idx)

    keep_indices = set(range(len(rows)))
    removed_by_score: Counter = Counter()
    for score, indices in score_to_indices.items():
        if len(indices) <= max_per_score:
            continue
        ranked = sorted(
            indices,
            key=lambda idx: short_hash(get_user_content(rows[idx])),
        )
        drop = set(ranked[max_per_score:])
        keep_indices.difference_update(drop)
        removed_by_score[str(score)] += len(drop)

    capped = [row for idx, row in enumerate(rows) if idx in keep_indices]
    stats = {"fluency_score_cap_removed": len(rows) - len(capped)}
    for score, count in sorted(removed_by_score.items(), key=lambda x: float(x[0])):
        stats[f"fluency_score_{score}_cap_removed"] = count
    return capped, stats


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(row.get("task", "unknown") for row in rows)
    output_counters: dict[str, Counter] = defaultdict(Counter)
    vocab_levels: Counter = Counter()
    vocab_keyword_lengths: Counter = Counter()
    fluency_scores: Counter = Counter()

    for row in rows:
        task = row.get("task", "unknown")
        obj = get_output_obj(row) or {}
        output_counters[task][stable_json(obj)] += 1
        if task == "vocabulary":
            vocab_levels[obj.get("level", "missing")] += 1
            kw = obj.get("key_words", [])
            vocab_keyword_lengths[len(kw) if isinstance(kw, list) else -1] += 1
        elif task == "fluency":
            fluency_scores[str(obj.get("fluency_score", "missing"))] += 1

    output_diversity = {}
    for task, counter in output_counters.items():
        total = sum(counter.values())
        top_count = counter.most_common(1)[0][1] if total else 0
        output_diversity[task] = {
            "rows": total,
            "unique_outputs": len(counter),
            "top_output_rate": round(top_count / total, 4) if total else 0,
        }

    total = len(rows)
    return {
        "samples": total,
        "task_counts": dict(sorted(task_counts.items())),
        "task_percent": {
            task: round(count / total, 4) for task, count in sorted(task_counts.items())
        } if total else {},
        "output_diversity": output_diversity,
        "vocabulary_level_counts": dict(sorted(vocab_levels.items())),
        "vocabulary_keyword_count_distribution": dict(sorted(vocab_keyword_lengths.items())),
        "fluency_score_top_counts": dict(fluency_scores.most_common(12)),
    }


def cross_split_overlap(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    split_keys = {
        name: {first_dedup_key(row.get("task", ""), get_user_content(row)) for row in rows}
        for name, rows in split_rows.items()
    }
    names = list(split_keys)
    overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap[f"{left}_vs_{right}"] = len(split_keys[left] & split_keys[right])
    return overlap


def load_downloaded_rows(downloaded_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {"available": downloaded_dir.exists(), "files": {}}
    if not downloaded_dir.exists():
        return [], report

    rows_for_candidates: list[dict[str, Any]] = []
    for path in sorted(downloaded_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["files"][path.name] = {"error": "invalid_json"}
            continue
        rows = data if isinstance(data, list) else data.get("data", [])
        if not isinstance(rows, list):
            rows = []
        task_counts = Counter(str(row.get("task", "unknown")) for row in rows if isinstance(row, dict))
        report["files"][path.name] = {
            "rows": len(rows),
            "task_counts": dict(sorted(task_counts.items())),
        }
        if path.name == "unified_training_data.json":
            rows_for_candidates = [row for row in rows if isinstance(row, dict)]

    if not rows_for_candidates:
        for path in sorted(downloaded_dir.glob("*_data.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, list):
                rows_for_candidates.extend(row for row in data if isinstance(row, dict))

    return rows_for_candidates, report


def build_downloaded_candidates(
    downloaded_rows: list[dict[str, Any]],
    primary_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen = set(primary_keys)
    rows, stats = clean_records("downloaded_candidate", downloaded_rows, seen)
    stats = dict(stats)
    stats["overlap_or_duplicate_removed"] = stats.get("dedup_removed", 0)
    return rows, stats


def load_explanation_rows(
    explanation_source: Path,
    quality_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "source": str(explanation_source),
        "available": explanation_source.exists(),
        "quality_threshold": quality_threshold,
    }
    if not explanation_source.exists():
        return [], report

    raw_text = explanation_source.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
        source_rows = loaded if isinstance(loaded, list) else loaded.get("data", [])
    except json.JSONDecodeError:
        source_rows = []
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                source_rows.append(json.loads(line))
            except json.JSONDecodeError:
                report.setdefault("jsonl_parse_errors", []).append(line_no)

    output_rows: list[dict[str, Any]] = []
    stats: Counter = Counter()
    seen_inputs: set[str] = set()
    error_types: Counter = Counter()
    qualities: Counter = Counter()

    for idx, item in enumerate(source_rows):
        if not isinstance(item, dict):
            stats["drop_non_object"] += 1
            continue
        user_text = str(item.get("input", "")).strip()
        explanation = str(item.get("output", "")).strip()
        try:
            quality_score = int(item.get("quality_score", 0))
        except (TypeError, ValueError):
            quality_score = 0
        error_type = str(item.get("error_type", "unknown") or "unknown")
        qualities[quality_score] += 1

        if quality_score < quality_threshold:
            stats["drop_low_quality"] += 1
            continue
        if not user_text or not explanation:
            stats["drop_missing_input_or_output"] += 1
            continue

        key = normalize_text(user_text)
        if key in seen_inputs:
            stats["drop_duplicate_input"] += 1
            continue
        seen_inputs.add(key)
        error_types[error_type] += 1

        output_rows.append({
            "task": "explanation",
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": explanation},
            ],
            "metadata": {
                "source": "vietnamese_explanations",
                "index": idx,
                "error_type": error_type,
                "quality_score": quality_score,
            },
        })

    report["source_total"] = len(source_rows)
    report["usable_rows"] = len(output_rows)
    report["filter_stats"] = dict(stats)
    report["quality_distribution"] = dict(sorted(qualities.items()))
    report["error_type_counts"] = dict(error_types.most_common())
    return output_rows, report


def split_explanation_rows(
    rows: list[dict[str, Any]],
    validation_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        error_type = str((row.get("metadata") or {}).get("error_type", "unknown"))
        grouped[error_type].append(row)

    split_rows = {"train": [], "validation": [], "test": []}
    per_error_type: dict[str, dict[str, int]] = {}

    for error_type, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: short_hash(
                get_user_content(row)
                + "\n"
                + get_assistant_content(row)
                + "\n"
                + error_type
            ),
        )
        total = len(ordered)
        if total >= 10:
            val_count = max(1, round(total * validation_ratio))
            test_count = max(1, round(total * test_ratio))
        elif total >= 3:
            val_count = 1
            test_count = 1
        else:
            val_count = 0
            test_count = 0
        if val_count + test_count >= total:
            test_count = max(0, total - val_count - 1)

        validation_part = ordered[:val_count]
        test_part = ordered[val_count:val_count + test_count]
        train_part = ordered[val_count + test_count:]

        split_rows["validation"].extend(validation_part)
        split_rows["test"].extend(test_part)
        split_rows["train"].extend(train_part)
        per_error_type[error_type] = {
            "train": len(train_part),
            "validation": len(validation_part),
            "test": len(test_part),
        }

    for split_name in split_rows:
        split_rows[split_name] = sorted(
            split_rows[split_name],
            key=lambda row: short_hash(get_user_content(row) + get_assistant_content(row)),
        )

    report = {
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "split_counts": {name: len(items) for name, items in split_rows.items()},
        "per_error_type": per_error_type,
    }
    return split_rows, report


def write_readme(path: Path, report_name: str) -> None:
    content = f"""# LexiLingo Training Data Clean V2

Generated by `preprocess_data_v2.py`.

Clean-v2 keeps train/validation/test separate, removes duplicate prompt leakage,
repairs vocabulary keywords into real lists, keeps fluency feedback as a target,
adds Vietnamese tutor explanations as a supervised task, and caps only the most
repetitive train fluency scores.
Metadata is normalized to a fixed flat schema for HuggingFace `load_dataset`.

Files:

- `train_clean.jsonl`, `val_clean.jsonl`, `test_clean.jsonl`: files used by the notebook.
- `{report_name}`: cleaning, split, leakage, and diversity report.

The original raw datasets are not modified.
"""
    path.write_text(content, encoding="utf-8")


def run(
    input_dir: Path,
    downloaded_dir: Path,
    explanation_source: Path,
    output_dir: Path,
    max_train_per_fluency_score: int,
    include_downloaded_in_train: bool,
    include_explanations: bool,
    explanation_quality_threshold: int,
    explanation_validation_ratio: float,
    explanation_test_ratio: float,
    write_downloaded_candidates: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "downloaded_dir": str(downloaded_dir),
        "explanation_source": str(explanation_source),
        "output_dir": str(output_dir),
        "clean_v2_policy": {
            "preserve_raw_splits": True,
            "remove_cross_split_duplicate_prompts": True,
            "vocabulary_keywords": "parse list/string, otherwise extract from input text",
            "fluency_target": "fluency_score + feedback",
            "explanation_target": "JSON object with Vietnamese explanation text",
            "include_explanations": include_explanations,
            "explanation_quality_threshold": explanation_quality_threshold,
            "explanation_validation_ratio": explanation_validation_ratio,
            "explanation_test_ratio": explanation_test_ratio,
            "max_train_per_fluency_score": max_train_per_fluency_score,
            "include_downloaded_in_train": include_downloaded_in_train,
        },
        "splits": {},
    }

    split_rows: dict[str, list[dict[str, Any]]] = {}
    seen_dedup: set[str] = set()

    for split_name, file_name in SPLITS.items():
        in_path = input_dir / file_name
        if not in_path.exists():
            raise FileNotFoundError(f"Missing required split: {in_path}")
        raw_rows = load_jsonl(in_path)
        cleaned, stats = clean_records(split_name, raw_rows, seen_dedup)
        if split_name == "train":
            cleaned, cap_stats = cap_train_fluency_scores(cleaned, max_train_per_fluency_score)
            stats.update(cap_stats)
            stats["total_out_after_cap"] = len(cleaned)
        split_rows[split_name] = cleaned
        report["splits"][split_name] = {
            "cleaning_stats": stats,
            "summary": summarize_rows(cleaned),
        }

    explanation_report: dict[str, Any] = {"used": False}
    if include_explanations:
        explanation_rows, source_report = load_explanation_rows(
            explanation_source,
            explanation_quality_threshold,
        )
        explanation_splits, split_report = split_explanation_rows(
            explanation_rows,
            explanation_validation_ratio,
            explanation_test_ratio,
        )
        explanation_report = {
            "used": True,
            "source_report": source_report,
            "split_report": split_report,
            "cleaning_stats": {},
        }
        for split_name in ["train", "validation", "test"]:
            cleaned_exp, exp_stats = clean_records(
                split_name,
                explanation_splits.get(split_name, []),
                seen_dedup,
            )
            split_rows[split_name].extend(cleaned_exp)
            explanation_report["cleaning_stats"][split_name] = exp_stats
            report["splits"][split_name]["explanation_rows_added"] = len(cleaned_exp)
            report["splits"][split_name]["summary"] = summarize_rows(split_rows[split_name])
    report["explanations"] = explanation_report

    downloaded_rows, downloaded_report = load_downloaded_rows(downloaded_dir)
    downloaded_candidates, downloaded_stats = build_downloaded_candidates(
        downloaded_rows,
        set(seen_dedup),
    )
    downloaded_report["candidate_cleaning_stats"] = downloaded_stats
    downloaded_report["candidate_summary"] = summarize_rows(downloaded_candidates)
    downloaded_report["used_in_train"] = include_downloaded_in_train

    if include_downloaded_in_train:
        split_rows["train"].extend(downloaded_candidates)
        report["splits"]["train"]["summary"] = summarize_rows(split_rows["train"])
        report["splits"]["train"]["downloaded_candidates_added"] = len(downloaded_candidates)

    report["cross_split_overlap_after_cleaning"] = cross_split_overlap(split_rows)
    report["downloaded_datasets"] = downloaded_report

    for split_name, rows in split_rows.items():
        compat_path = output_dir / COMPAT_OUTPUT_NAMES[split_name]
        write_jsonl(compat_path, rows)

    for file_name in LEGACY_UNUSED_DATA_FILES:
        legacy_path = output_dir / file_name
        if legacy_path.exists():
            legacy_path.unlink()

    if write_downloaded_candidates:
        candidate_path = output_dir / "downloaded_candidates_clean_v2.jsonl"
        write_jsonl(candidate_path, downloaded_candidates)

    report_path = output_dir / "clean_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(output_dir / "README_CLEAN_V2.md", report_path.name)

    print(f"Clean-v2 written to: {output_dir}")
    for split_name, rows in split_rows.items():
        summary = summarize_rows(rows)
        print(f"  {split_name:10s}: {summary['samples']:6d} {summary['task_counts']}")
    print(f"  downloaded candidates: {len(downloaded_candidates)} (not written by default)")
    print(f"  report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LexiLingo clean-v2 training data")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Folder with train/val/test JSONL")
    parser.add_argument("--downloaded-dir", default=str(DEFAULT_DOWNLOADED_DIR), help="Raw downloaded dataset folder")
    parser.add_argument(
        "--explanation-source",
        default=str(DEFAULT_EXPLANATION_SOURCE),
        help="Vietnamese tutor explanation source. Supports JSON array or JSONL.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder")
    parser.add_argument(
        "--max-train-per-fluency-score",
        type=int,
        default=1200,
        help="Cap train rows for any exact fluency_score; <=0 disables",
    )
    parser.add_argument(
        "--include-downloaded-in-train",
        action="store_true",
        help="Append cleaned non-primary downloaded candidates to train. Off by default.",
    )
    parser.add_argument(
        "--write-downloaded-candidates",
        action="store_true",
        help="Write downloaded_candidates_clean_v2.jsonl for manual inspection. Off by default.",
    )
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Disable adding Vietnamese tutor explanation samples.",
    )
    parser.add_argument(
        "--explanation-quality-threshold",
        type=int,
        default=50,
        help="Minimum quality_score for tutor explanations.",
    )
    parser.add_argument(
        "--explanation-validation-ratio",
        type=float,
        default=0.05,
        help="Stratified validation ratio for tutor explanations.",
    )
    parser.add_argument(
        "--explanation-test-ratio",
        type=float,
        default=0.05,
        help="Stratified test ratio for tutor explanations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_dir=Path(args.input_dir),
        downloaded_dir=Path(args.downloaded_dir),
        explanation_source=Path(args.explanation_source),
        output_dir=Path(args.output_dir),
        max_train_per_fluency_score=args.max_train_per_fluency_score,
        include_downloaded_in_train=args.include_downloaded_in_train,
        include_explanations=not args.no_explanations,
        explanation_quality_threshold=args.explanation_quality_threshold,
        explanation_validation_ratio=args.explanation_validation_ratio,
        explanation_test_ratio=args.explanation_test_ratio,
        write_downloaded_candidates=args.write_downloaded_candidates,
    )
