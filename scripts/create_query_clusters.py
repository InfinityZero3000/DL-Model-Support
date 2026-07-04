#!/usr/bin/env python3
"""
create_query_clusters.py — Generate L1-cache benchmark từ query clusters có semantic overlap.

Vấn đề của benchmark hiện tại: n=20 unique samples → L1 = 0% vì không có semantic overlap.
Script này tạo clusters, mỗi cluster 3-5 paraphrases của cùng 1 câu hỏi.
Khi Q1 (cold) → L0 hit; Q2 (paraphrase) → L1 hit vì cùng bucket.

Output: datasets/benchmarks/query_clusters/validation.jsonl
Usage:
    python scripts/create_query_clusters.py --n-clusters 10 --cluster-size 4
"""
from __future__ import annotations

import argparse
import json
import re
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets" / "benchmarks"
HOTPOTQA_PATH = DATASETS_DIR / "hotpotqa" / "validation.jsonl"
OUTPUT_DIR = DATASETS_DIR / "query_clusters"


def _load_hotpotqa(path: Path, n: int) -> list[dict]:
    samples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task") == "multihop_qa" and obj.get("metadata", {}).get("context"):
                samples.append(obj)
            if len(samples) >= n * 3:
                break
    return samples


def _paraphrase_question(q: str, variant: int) -> str:
    """
    Tạo paraphrase thủ công không cần LLM — dùng template patterns.
    Đủ để trigger L1 semantic similarity mà không cần LLM key.
    """
    q = q.strip().rstrip("?")
    templates = [
        # Variant 0: original
        q + "?",
        # Variant 1: wh-reorder
        lambda s: re.sub(r"^(Were|Was|Did|Is|Are|Do|Does) (.+?) and (.+?) (.+?)\??$",
                         r"Did \2 and \3 \4?", s, flags=re.I) if re.search(r"^(Were|Was|Did)", s, re.I) else f"Tell me: {s}?",
        # Variant 2: "Can you tell me..." wrapper
        lambda s: f"Can you tell me {s[0].lower() + s[1:]}?",
        # Variant 3: "What is known about..." for non-yes/no
        lambda s: f"What do we know about {s[0].lower() + s[1:]}?" if not re.search(r"^(Were|Was|Did|Is|Are|Do|Does)", s, re.I) else f"Is it true that {s[0].lower() + s[1:]}?",
        # Variant 4: passive/different structure
        lambda s: f"Regarding the question '{s}', what is the answer?",
    ]

    if variant == 0:
        return q + "?"
    fn = templates[min(variant, len(templates) - 1)]
    if callable(fn):
        try:
            result = fn(q)
            return result if result.strip() else q + "?"
        except Exception:
            return q + "?"
    return str(fn)


def _make_cluster(sample: dict, cluster_id: int, cluster_size: int) -> list[dict]:
    """Create cluster_size paraphrase variants of the same QA sample."""
    metadata = sample.get("metadata", {}) or {}
    base_q = str(sample.get("text") or sample.get("metadata", {}).get("raw_text") or "")
    answer = sample.get("output") or {}
    context = metadata.get("context") or ""
    context_docs = metadata.get("context_docs") or []
    facts = answer.get("supporting_facts")
    supporting_titles = metadata.get("supporting_titles") or (
        facts.get("title") if isinstance(facts, dict) else []
    )
    source_id = metadata.get("source_id") or str(cluster_id)

    cluster = []
    for variant_idx in range(cluster_size):
        query_text = _paraphrase_question(base_q, variant_idx)
        item = {
            "text": query_text,
            "task": "multihop_qa",
            "output": answer,
            "metadata": {
                "source_dataset": "query_clusters",
                "source_id": source_id,
                "cluster_id": cluster_id,
                "cluster_variant": variant_idx,
                "cluster_size": cluster_size,
                "base_question": base_q,
                "context": context,
                "context_docs": context_docs,
                "supporting_titles": supporting_titles,
                "raw_text": query_text,
                # L1 cache test label: variant > 0 should hit L1 after variant 0 is cached
                "expected_cache_decision": "reuse" if variant_idx > 0 else "full",
                "expected_cache_layer": "L1" if variant_idx > 0 else "none",
            },
        }
        cluster.append(item)
    return cluster


def main() -> None:
    parser = argparse.ArgumentParser(description="Create query cluster dataset for L1 cache testing")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of question clusters")
    parser.add_argument("--cluster-size", type=int, default=4, help="Paraphrases per cluster (3-5 recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading HotpotQA from {HOTPOTQA_PATH}...")
    samples = _load_hotpotqa(HOTPOTQA_PATH, args.n_clusters)
    if not samples:
        raise SystemExit(f"No samples found at {HOTPOTQA_PATH}")

    selected = random.sample(samples, k=min(args.n_clusters, len(samples)))
    print(f"Selected {len(selected)} base questions → {len(selected) * args.cluster_size} total samples")

    all_items = []
    for cluster_id, sample in enumerate(selected):
        cluster = _make_cluster(sample, cluster_id, args.cluster_size)
        all_items.extend(cluster)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "validation.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_items)} samples ({len(selected)} clusters × {args.cluster_size} variants)")
    print(f"Output: {output_path}")
    print("\nCluster preview:")
    for i in range(min(2, len(selected))):
        base_q = selected[i].get("text", "")
        print(f"  Cluster {i}: '{base_q[:60]}...'")
        for v in range(args.cluster_size):
            print(f"    v{v}: '{_paraphrase_question(base_q.rstrip('?'), v)[:70]}'")

    readme = output_path.parent / "README.md"
    readme.write_text(
        f"# Query Clusters Benchmark\n\n"
        f"**Purpose**: Test L1 graph-bucket cache with semantically similar queries.\n\n"
        f"**Structure**: {len(selected)} clusters × {args.cluster_size} paraphrase variants per cluster.\n"
        f"- Variant 0: cold pass (expected L2 full retrieval)\n"
        f"- Variants 1+: warm paraphrase (expected L1 cache hit from same graph bucket)\n\n"
        f"**Generated from**: HotpotQA validation set (seed={args.seed})\n\n"
        f"**How to run**:\n"
        f"```bash\n"
        f"bash benchmark/run_benchmark_all_datasets.sh 10 cluster public_cag_compare\n"
        f"# Or directly:\n"
        f"python benchmark/benchmark_public_qa.py --dataset-preset query_clusters --n 40 --cache-repeats 1\n"
        f"```\n",
        encoding="utf-8",
    )
    print(f"README: {readme}")


if __name__ == "__main__":
    main()
