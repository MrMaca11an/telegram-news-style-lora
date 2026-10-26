#!/usr/bin/env bash
# Production server candidate for one 24 GB NVIDIA GPU (RTX 3090 / A10G class).
# STATUS: candidate — requires smoke test on the target GPU/version before production
# (see deploy/smoke_test.py; not executed: 24 GB hardware unavailable).
#
# Validates prerequisites, then starts the OpenAI-compatible vLLM server with two
# LoRA adapters (type1, type2) on the same BF16 base model.
set -euo pipefail

BASE_MODEL="unsloth/Qwen3.5-2B"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# --- validation ---
fail() { echo "ERROR: $*" >&2; exit 1; }
command -v python >/dev/null || fail "python not found"
python -c "import vllm" 2>/dev/null || fail "vllm not importable; pip install -r deploy/requirements-vllm.txt"
python - <<EOF || fail "Qwen3.5 architecture not supported by installed vLLM version"
import vllm, importlib
assert tuple(int(x) for x in vllm.__version__.split(".")[:2]) >= (0, 11), "vllm too old for qwen3_5"
from vllm.model_executor.models import registry as r
import vllm.model_executor.models.qwen3_5  # noqa: F401  (raises if absent)
print("vllm", vllm.__version__, "with qwen3_5 support OK")
EOF
[ -d "$ADAPTER_DIR/type1" ] && [ -f "$ADAPTER_DIR/type1/adapter_config.json" ] || fail "missing $ADAPTER_DIR/type1"
[ -d "$ADAPTER_DIR/type2" ] && [ -f "$ADAPTER_DIR/type2/adapter_config.json" ] || fail "missing $ADAPTER_DIR/type2"
python - "$BASE_MODEL" <<'EOF' || fail "adapters base_model mismatch"
import sys, json
base = sys.argv[1]
for a in ("type1", "type2"):
    cfg = json.load(open(f"adapters/{a}/adapter_config.json"))
    assert cfg.get("base_model_name_or_path", base).endswith("Qwen3.5-2B"), f"{a}: wrong base model"
print("adapter base_model check OK")
EOF

exec python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 15 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 16 \
  --lora-modules "type1=$ADAPTER_DIR/type1" "type2=$ADAPTER_DIR/type2" \
  --host "$HOST" --port "$PORT"
