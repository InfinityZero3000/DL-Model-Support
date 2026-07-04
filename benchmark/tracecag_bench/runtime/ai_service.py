from __future__ import annotations

import time
from typing import Any

from ..catalog import ModeSpec
from ..schemas import DriftRequest, PublicQASample, RunObservation


class AIServiceRuntime:
    def __init__(self, pipeline: Any | None = None) -> None:
        if pipeline is None:
            from api.services.model_gateway import get_model_gateway
            from api.services.trace_cag.graph import TraceCAGPipeline

            get_model_gateway()
            pipeline = TraceCAGPipeline()
        self.pipeline = pipeline

    async def run_public_qa(
        self,
        sample: PublicQASample,
        mode: ModeSpec,
        *,
        session_id: str,
        generation_policy: str,
        evidence_mode: str,
        is_warm: bool,
    ) -> RunObservation:
        metadata = {
            "_benchmark_mode": mode.name,
            "_benchmark_ranker": mode.ranker,
            "source_id": sample.sample_id,
            "supporting_titles": list(sample.supporting_titles),
        }
        context = ""
        if evidence_mode == "candidate_pool":
            metadata["context_docs"] = [doc.__dict__ for doc in sample.context_docs]
        else:
            metadata["_kg_only"] = True
        return await self._invoke(
            sample_id=sample.sample_id,
            mode=mode,
            question=sample.question,
            answers=sample.answers,
            session_id=session_id,
            generation_policy=generation_policy,
            benchmark_context=context,
            benchmark_metadata=metadata,
            learner_profile={"level": str(sample.metadata.get("benchmark_level") or "B1")},
            is_warm=is_warm,
        )

    async def run_drift(
        self,
        request: DriftRequest,
        mode: ModeSpec,
        *,
        session_id: str,
        generation_policy: str,
        context_docs: list[dict] | None = None,
    ) -> RunObservation:
        reference_docs = context_docs or (
            [{"item_id": "drift_reference", "title": "drift_reference", "text": request.expected_output}]
            if request.expected_output else []
        )
        metadata = {
            "_benchmark_mode": mode.name,
            "_benchmark_ranker": mode.ranker,
            "_tracecag_state": request.state,
            "context_docs": reference_docs,
            "supporting_titles": ["drift_reference"] if reference_docs else [],
            "source_id": request.request_id,
        }
        return await self._invoke(
            sample_id=request.request_id,
            mode=mode,
            question=request.query,
            answers=(request.expected_output,) if request.expected_output else (),
            session_id=session_id,
            generation_policy=generation_policy,
            benchmark_context="",
            benchmark_metadata=metadata,
            learner_profile={"level": str(request.state.get("level") or "B1")},
            expected_route=request.expected_route,
            safety_label=request.safety_label,
            drift_type=request.drift_type,
        )

    async def _invoke(
        self,
        *,
        sample_id: str,
        mode: ModeSpec,
        question: str,
        answers: tuple[str, ...],
        session_id: str,
        generation_policy: str,
        benchmark_context: str,
        benchmark_metadata: dict,
        learner_profile: dict,
        expected_route: str = "",
        safety_label: str = "",
        drift_type: str = "",
        is_warm: bool = False,
    ) -> RunObservation:
        started = time.monotonic()
        try:
            result = await self.pipeline.analyze(
                user_input=question,
                session_id=session_id,
                learner_profile=learner_profile,
                conversation_history=[],
                cache_policy=mode.cache_policy,
                retrieval_policy=mode.retrieval_policy,
                generation_policy=generation_policy,
                benchmark_task="multihop_qa",
                benchmark_context=benchmark_context,
                benchmark_metadata=benchmark_metadata,
                return_raw_state=True,
            )
        except Exception as exc:
            return RunObservation(
                sample_id=sample_id, mode=mode.name, gold_answers=answers,
                latency_ms=(time.monotonic() - started) * 1000.0, error=str(exc),
                expected_route=expected_route, safety_label=safety_label, drift_type=drift_type,
                is_warm=is_warm,
            )
        models = [str(item) for item in result.get("models_used") or []]
        provider, model, fallback = classify_provider(models)
        return RunObservation(
            sample_id=sample_id,
            mode=mode.name,
            answer=str(result.get("tutor_response") or ""),
            gold_answers=answers,
            latency_ms=float(result.get("latency_ms") or (time.monotonic() - started) * 1000.0),
            ttft_ms=float(result.get("ttft_ms") or 0.0),
            cache_hit=bool(result.get("cache_hit")),
            cache_decision=str(result.get("cache_decision") or "full"),
            cache_layer=str(result.get("cache_layer") or "none"),
            reuse_risk=float(result.get("reuse_risk") or 0.0),
            cache_gate_meta=dict(result.get("cache_gate_meta") or {}),
            retrieval_trace=list(result.get("retrieval_trace") or []),
            retrieval_meta=dict(result.get("retrieval_meta") or {}),
            kg_seed_concepts=list(result.get("kg_seed_concepts") or []),
            kg_expanded_nodes=list(result.get("kg_expanded_nodes") or []),
            graph_update=dict(result.get("graph_update") or {}),
            models_used=models,
            observed_provider=provider,
            observed_model=model,
            fallback_provider=fallback,
            expected_route=expected_route,
            safety_label=safety_label,
            drift_type=drift_type,
            is_warm=is_warm,
            error=str(result.get("error") or ""),
        )


def classify_provider(models: list[str]) -> tuple[str, str, str]:
    for value in reversed(models):
        lowered = value.lower()
        if lowered.startswith("groq/"):
            return "groq", value.split("/", 1)[1], ""
        if "gemini" in lowered:
            return "gemini", value, "gemini"
        if "ollama" in lowered:
            return "ollama", value, "ollama"
        if "benchmark_bypass" in lowered or "extractive" in lowered:
            return "bypass", value, "bypass"
    return "", "", ""
