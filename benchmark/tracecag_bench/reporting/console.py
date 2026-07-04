from __future__ import annotations


def print_summary(report: dict) -> None:
    print(f"Protocol: {report.get('protocol')}  Dataset: {report['dataset']['name']}")
    summaries = report.get("summaries") or {}
    for mode, summary in summaries.items():
        print(
            f"{mode}: EM={summary['exact_match']:.1%} F1={summary['token_f1']:.1%} "
            f"R@5={summary['retrieval']['recall_at_5']:.1%} "
            f"Hit={summary['cache']['cache_hit_rate']:.1%} "
            f"Lat={summary['latency']['mean_ms']:.0f}ms"
        )
    if report.get("summary"):
        summary = report["summary"]
        print(
            f"Safety={summary['safe_reuse_precision']:.1%} "
            f"AdmRecall={summary['admissible_recall']:.1%} "
            f"Unsafe={summary['unsafe_acceptance_rate']:.1%} "
            f"RouteAcc={summary['route_accuracy']:.1%}"
        )
    validation = report.get("run_validation") or {}
    print(f"Validation: {'PASS' if validation.get('passed') else 'FAIL'}")
