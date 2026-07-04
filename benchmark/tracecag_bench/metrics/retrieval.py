from __future__ import annotations

import math


def retrieval_metrics(trace: list[dict], relevant_titles: tuple[str, ...], k: int = 5) -> dict[str, float]:
    relevant = {_norm(item) for item in relevant_titles if item}
    ranked = trace[:k]
    flags = [_is_relevant(item, relevant) for item in ranked]
    total_relevant = len(relevant)
    result: dict[str, float] = {}
    for cutoff in (1, 3, 5):
        selected = flags[:cutoff]
        result[f"recall_at_{cutoff}"] = sum(selected) / total_relevant if total_relevant else 0.0
    result["precision_at_5"] = sum(flags) / len(ranked) if ranked else 0.0
    result["mrr_at_5"] = next((1.0 / rank for rank, hit in enumerate(flags, 1) if hit), 0.0)
    dcg = sum((1.0 if hit else 0.0) / math.log2(rank + 1) for rank, hit in enumerate(flags, 1))
    ideal_count = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    result["ndcg_at_5"] = dcg / idcg if idcg else 0.0
    return result


def _is_relevant(item: dict, relevant: set[str]) -> bool:
    if "is_relevant" in item:
        return bool(item["is_relevant"])
    return _norm(str(item.get("title") or item.get("item_id") or "")) in relevant


def _norm(value: str) -> str:
    return " ".join(value.lower().split())
