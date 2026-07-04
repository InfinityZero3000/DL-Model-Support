from __future__ import annotations

from collections import Counter
import re


def normalize_answer(text: str) -> str:
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, answers: tuple[str, ...]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def token_f1(prediction: str, answers: tuple[str, ...]) -> float:
    return max((_single_f1(prediction, answer) for answer in answers), default=0.0)


def _single_f1(prediction: str, answer: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)
