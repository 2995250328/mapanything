#!/usr/bin/env bash
set -euo pipefail
export HYDRA_FULL_ERROR=1

usage() {
  cat <<'USAGE' >&2
Usage: run_demo_reconstruction.sh [--device DEVICE] [--num-samples N] [--indices LIST] [--data-root PATH] \
                                  <conda_env> <stored_feature_file> <output_root> [overrides...]
  --device       Torch device passed to the demo (default: cuda)
  --num-samples  Number of dataset items to process (default: 1)
  --indices      Comma-separated dataset indices to process (overrides --num-samples)
  --data-root    Root directory containing the converted WAI dataset
USAGE
}

DEVICE="cuda"
NUM_SAMPLES="1"
INDICES=""
DATA_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      DEVICE="$2"
      shift 2
      ;;
    --num-samples)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --indices)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      INDICES="$2"
      shift 2
      ;;
    --data-root)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      DATA_ROOT=$(realpath "$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

CONDA_ENV="$1"
STORED_FEATURE_FILE=$(realpath "$2")
OUTPUT_ROOT=$(realpath "$3")
shift 3

EXTRA_OVERRIDES=("$@")

mkdir -p "$OUTPUT_ROOT"

CLI_ARGS=(
  "fusion.stored_feature_file=$STORED_FEATURE_FILE"
  "demo.output_dir=$OUTPUT_ROOT"
  "demo.device=$DEVICE"
  "demo.num_samples=$NUM_SAMPLES"
)

if [[ -n "$INDICES" ]]; then
  CLI_ARGS+=("demo.sample_indices=[$INDICES]")
fi

if [[ -n "$DATA_ROOT" ]]; then
  CLI_ARGS+=("root_data_dir=$DATA_ROOT")
fi

CLI_ARGS+=("${EXTRA_OVERRIDES[@]}")

set -x
conda run -n "$CONDA_ENV" python -m mapanything.tasks.aa_feature_fusion.demo "${CLI_ARGS[@]}"
set +x

echo "Reconstruction artifacts written to $OUTPUT_ROOT"
