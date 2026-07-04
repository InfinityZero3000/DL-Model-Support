from __future__ import annotations

import json
import random
import re
from pathlib import Path

from ..schemas import ContextDocument, PublicQASample


def load_public_qa(path: Path, *, dataset: str, n: int | None, seed: int) -> list[PublicQASample]:
    rows = _read_jsonl(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [_parse_sample(row, dataset, path) for row in rows]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def _parse_sample(row: dict, dataset: str, path: Path) -> PublicQASample:
    question = str(row.get("text") or row.get("question") or "").strip()
    output = row.get("output") or {}
    metadata = dict(row.get("metadata") or {})
    primary = output.get("answer", row.get("answer", ""))
    aliases = output.get("answers") or metadata.get("answers") or []
    answers = tuple(dict.fromkeys(str(item).strip() for item in [primary, *aliases] if str(item).strip()))
    docs_raw = metadata.get("context_docs") or row.get("context_docs") or []
    docs = tuple(
        ContextDocument(
            item_id=str(doc.get("item_id") or doc.get("title") or ""),
            title=str(doc.get("title") or doc.get("item_id") or ""),
            text=str(doc.get("text") or doc.get("content") or ""),
        )
        for doc in docs_raw
        if isinstance(doc, dict)
    )
    support = metadata.get("supporting_titles")
    if support is None:
        facts = output.get("supporting_facts") or {}
        support = facts.get("title", []) if isinstance(facts, dict) else []
    supporting_titles = tuple(dict.fromkeys(str(item).strip() for item in support if str(item).strip()))
    if not question or not answers:
        raise ValueError(f"{path}: sample requires question and answer")
    if not docs:
        docs = _docs_from_context(str(metadata.get("context") or row.get("context") or ""))
    return PublicQASample(
        sample_id=str(metadata.get("source_id") or row.get("id") or question),
        dataset=dataset,
        question=question,
        answers=answers,
        context_docs=docs,
        supporting_titles=supporting_titles,
        context=str(metadata.get("context") or row.get("context") or ""),
        cluster_id=str(metadata["cluster_id"]) if "cluster_id" in metadata else None,
        cluster_variant=int(metadata["cluster_variant"]) if "cluster_variant" in metadata else None,
        expected_cache_decision=metadata.get("expected_cache_decision"),
        expected_cache_layer=metadata.get("expected_cache_layer"),
        metadata=metadata,
    )


def _docs_from_context(context: str) -> tuple[ContextDocument, ...]:
    if not context.strip():
        return ()
    docs = []
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^\[([^\]]+)\]\s*(.+)$", line)
        if not match:
            continue
        title = match.group(1).strip()
        text = match.group(2).strip()
        if title and text:
            docs.append(ContextDocument(title, title, text))
    if docs:
        return tuple(docs)
    return (ContextDocument("benchmark_context", "benchmark_context", context),)
