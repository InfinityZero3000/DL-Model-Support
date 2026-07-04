from __future__ import annotations

import statistics

from ..catalog import ModeSpec
from ..config import BenchmarkConfig
from ..kg.preflight import run_kg_preflight
from ..metrics.cache import cache_metrics
from ..metrics.latency import latency_metrics
from ..metrics.retrieval import retrieval_metrics
from ..metrics.text import exact_match, token_f1
from ..runtime.ai_service import AIServiceRuntime
from ..runtime.reset import reset_runtime_state
from ..schemas import PublicQASample, RunObservation


async def run_public_qa_protocol(
    samples: list[PublicQASample],
    modes: list[ModeSpec],
    config: BenchmarkConfig,
    *,
    runtime: AIServiceRuntime | None = None,
) -> dict:
    runtime = runtime or AIServiceRuntime()
    all_observations: list[RunObservation] = []
    summaries = {}
    resets = []
    for mode in modes:
        resets.append({"scope": "mode", "mode": mode.name, "cleared": reset_runtime_state()})
        observations: list[RunObservation] = []
        for repeat in range(config.cache_repeats):
            for index, sample in enumerate(samples):
                item = await runtime.run_public_qa(
                    sample, mode,
                    session_id=f"bench_{mode.name}_{index:05d}",
                    generation_policy=config.generation_policy,
                    evidence_mode=config.evidence_mode,
                    is_warm=repeat > 0,
                )
                observations.append(item)
                all_observations.append(item)
        summaries[mode.name] = summarize_public_qa(samples, observations)
    titles = [title for sample in samples for title in sample.supporting_titles]
    return {
        "protocol": "public_qa",
        "evidence_mode": config.evidence_mode,
        "summaries": summaries,
        "observations": [item.as_dict() for item in all_observations],
        "kg_preflight": run_kg_preflight(titles),
        "resets": resets,
    }


def summarize_public_qa(samples: list[PublicQASample], observations: list[RunObservation]) -> dict:
    sample_map = {sample.sample_id: sample for sample in samples}
    em_values = [exact_match(item.answer, item.gold_answers) for item in observations]
    f1_values = [token_f1(item.answer, item.gold_answers) for item in observations]
    retrieval = [
        retrieval_metrics(item.retrieval_trace, sample_map[item.sample_id].supporting_titles)
        for item in observations if item.sample_id in sample_map
    ]
    retrieval_summary = {
        key: statistics.mean(row[key] for row in retrieval) if retrieval else 0.0
        for key in ("recall_at_1", "recall_at_3", "recall_at_5", "precision_at_5", "mrr_at_5", "ndcg_at_5")
    }
    return {
        "n_total": len(observations),
        "exact_match": statistics.mean(em_values) if em_values else 0.0,
        "token_f1": statistics.mean(f1_values) if f1_values else 0.0,
        "retrieval": retrieval_summary,
        "cache": cache_metrics(observations),
        "latency": latency_metrics(observations),
        "provider": provider_summary(observations),
        "errors": sum(bool(item.error) for item in observations),
    }


def provider_summary(observations: list[RunObservation]) -> dict:
    return {
        "observed_providers": sorted({item.observed_provider for item in observations if item.observed_provider}),
        "observed_models": sorted({item.observed_model for item in observations if item.observed_model}),
        "fallback_rate": sum(bool(item.fallback_provider) for item in observations) / len(observations) if observations else 0.0,
    }
