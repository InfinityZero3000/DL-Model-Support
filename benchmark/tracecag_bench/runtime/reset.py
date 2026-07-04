from __future__ import annotations


def reset_runtime_state(*, provider_state: bool = False) -> dict[str, int]:
    import api.services.trace_cag.nodes_v2 as nodes
    import api.services.trace_cag.retrieval_ranker as ranker

    counts = {
        "response_cache": len(nodes._MEM_RESPONSE_CACHE),
        "graph_buckets": len(nodes._MEM_GRAPH_BUCKETS),
        "bucket_versions": len(nodes._MEM_BUCKET_VERSIONS),
        "kg_query_cache": len(nodes._KG_QUERY_CACHE),
    }
    nodes._MEM_RESPONSE_CACHE.clear()
    nodes._MEM_GRAPH_BUCKETS.clear()
    nodes._MEM_BUCKET_VERSIONS.clear()
    nodes._KG_QUERY_CACHE.clear()
    ranker._RANKER_INSTANCE = None
    if provider_state:
        nodes._PROVIDER_NEXT_REQUEST_AT.clear()
        nodes._PROVIDER_LAST_WAIT_LOG_AT.clear()
        nodes._PROVIDER_DISABLED_UNTIL.clear()
    return counts
