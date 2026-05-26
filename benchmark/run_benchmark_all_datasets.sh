#!/usr/bin/env bash
# =============================================================================
# run_benchmark_all_datasets.sh
# Public QA benchmark for Vanilla CAG vs HippoRAG proxy vs TRACE-CAG
# with quota-aware dataset selection and key rotation
#
# Groq default model: llama-3.1-8b-instant
# Override with GROQ_MODEL if you want to benchmark provider-side prompt caching
# on supported models such as moonshotai/kimi-k2-instruct-0905 or openai/gpt-oss-*.
# Cached-token accounting in the benchmark report is only populated when the
# serving backend returns usage fields. Each run now writes a full report plus
# two sidecar summaries: one for application-level reuse metrics and one for
# provider-side prompt-caching/token-accounting metrics.
#
# Groq  llama-3.1-8b-instant: RPM=30 / RPD=14.4K / TPM=6K per key
#   3 keys → effective ~24 RPM (8 RPM/key × 3, TPM-safe)
#
# Gemini 2.5 Flash: RPM=5 / RPD=20 per key
#   Keep as fallback only when all Groq keys are cooling
#
# Usage:
#   GROQ_KEYS="k1,k2,k3" bash run_benchmark_all_datasets.sh
#   GROQ_KEYS="k1,k2,k3" bash run_benchmark_all_datasets.sh 32 core public_cag_compare
#   GROQ_KEYS="k1,k2,k3" GEMINI_KEYS="g1,g2" bash run_benchmark_all_datasets.sh 24 all public_cag_quality
#   GROQ_KEYS="k1,k2,k3" bash run_benchmark_all_datasets.sh 40 core state_drift
#   GROQ_KEYS="k1,k2,k3" bash run_benchmark_all_datasets.sh 16 drift state_drift
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLMS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$DLMS_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/ai-service/venv/bin/python}"
BENCH_DIR="$DLMS_DIR/benchmark"
REPORT_DIR="$DLMS_DIR/datasets/benchmarks"
SAMPLE_COUNT="${1:-40}"
DATASET_GROUP="${2:-core}"
COMPARISON_PROFILE="${3:-public_cag_compare}"
QUOTA_STATE_FILE="$REPORT_DIR/provider_quota_state.json"

# Load .env (DL-Model-Support/.env)
# ---------------------------------------------------------------------------
ENV_FILE="$DLMS_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# CLI overrides take precedence over .env
GROQ_KEYS="${GROQ_KEYS:-${GROQ_API_KEY:-}}"
GEMINI_KEYS="${GEMINI_KEYS:-${GEMINI_API_KEY:-}}"

if [[ -z "$GROQ_KEYS" ]]; then
  echo "No Groq keys found. Set GROQ_KEYS in $ENV_FILE or pass via environment." >&2
  echo "Missing GROQ_KEYS (comma-separated) or GROQ_API_KEY in environment." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Common args
# ---------------------------------------------------------------------------
COMMON_ARGS=(
  --comparison-profile "$COMPARISON_PROFILE"
  --generation-policy auto
  --llm-provider groq
  --groq-model "${GROQ_MODEL:-llama-3.1-8b-instant}"
  --groq-rpm 8
  --groq-keys "$GROQ_KEYS"
  --quota-state-file "$QUOTA_STATE_FILE"
  --n "$SAMPLE_COUNT"
)

if [[ -n "$GEMINI_KEYS" ]]; then
  COMMON_ARGS+=(
    --gemini-keys "$GEMINI_KEYS"
    --enable-gemini-fallback
  )
fi

cd "$ROOT_DIR"

run_dataset() {
  local preset="$1"
  local report="$2"
  echo ""
  echo "============================================================"
  echo " Dataset: $preset   n=$SAMPLE_COUNT"
  echo " Report : $report"
  echo "============================================================"
  "$PYTHON_BIN" "$BENCH_DIR/benchmark_public_qa.py" \
    --dataset-preset "$preset" \
    "${COMMON_ARGS[@]}" \
    --report-json "$report"
  echo "[done] $preset"
}

build_report_path() {
  local preset="$1"
  echo "$REPORT_DIR/$preset/compare-${COMPARISON_PROFILE}-n${SAMPLE_COUNT}-groq.json"
}

DATASETS=()
case "$DATASET_GROUP" in
  core)
    DATASETS=(hotpotqa 2wikimultihopqa musique)
    ;;
  drift)
    # State Drift evaluation — run the curated drift probes first, then larger multi-hop sets.
    DATASETS=(tracecag_drift_probes hotpotqa 2wikimultihopqa musique)
    ;;
  all)
    # All available datasets including the curated drift probes
    DATASETS=(tracecag_drift_probes hotpotqa 2wikimultihopqa musique)
    ;;
  *)
    echo "Unknown dataset group: $DATASET_GROUP" >&2
    echo "Use one of: core, drift, all" >&2
    exit 1
    ;;
esac

START_TIME=$(date +%s)

# Default path is multi-hop first because it is the most informative setup for
# Vanilla CAG vs HippoRAG-style memory retrieval vs TRACE-CAG.
for dataset in "${DATASETS[@]}"; do
  run_dataset "$dataset" "$(build_report_path "$dataset")"
done

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
echo ""
echo "Dataset group '${DATASET_GROUP}' complete in ${ELAPSED}s."
echo "Reports:"
for dataset in "${DATASETS[@]}"; do
  f="$(build_report_path "$dataset")"
  if [[ -f "$f" ]]; then
    echo "  ✓ $f"
  else
    echo "  ✗ MISSING: $f"
  fi
done
