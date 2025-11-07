#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: seven_scenes_to_wai.sh [--datasets scene1[,scene2...]] [--device DEVICE] [--moge-batch-size N] \
                              <processed_root> <wai_output_dir> <conda_env> [conversion overrides...]
  processed_root 目录需要包含 pgt_7scenes_* 子目录 (train/test/calibration/depth/poses/rgb)。
  --datasets 支持用逗号分隔的场景列表，例如 --datasets chess 或 --datasets chess,heads。
  --device 控制转换与后处理脚本使用的 PyTorch 设备（默认 cuda，可设置为 cpu、cuda:1 等）。
  --moge-batch-size 指定 MoGe 推理批大小；默认保持配置文件中的设定。
EOF
}

DATASET_FILTER=""
DEVICE="cuda"
MOGE_BATCH_SIZE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      DATASET_FILTER="$2"
      shift 2
      ;;
    --device)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --moge-batch-size)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      MOGE_BATCH_SIZE="$2"
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

PROCESSED_ROOT=$(realpath "$1")
WAI_DIR=$(realpath "$2")
CONDA_ENV="$3"
shift 3

CONVERSION_OVERRIDES=("$@")

if [[ -n "${DATASET_FILTER}" ]]; then
  IFS=',' read -r -a _dataset_array <<< "${DATASET_FILTER}"
  dataset_entries=()
  for raw_name in "${_dataset_array[@]}"; do
    name=${raw_name//[[:space:]]/}
    if [[ -n "${name}" ]]; then
      dataset_entries+=("'${name}'")
    fi
  done
  if [[ ${#dataset_entries[@]} -gt 0 ]]; then
    joined=$(printf '%s,' "${dataset_entries[@]}")
    joined=${joined%,}
    CONVERSION_OVERRIDES+=("dataset_whitelist=[${joined}]")
  fi
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

mkdir -p "${WAI_DIR}"

set -x

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.conversion.seven_scenes \
  original_root="${PROCESSED_ROOT}" \
  root="${WAI_DIR}" \
  device="${DEVICE}" \
  "${CONVERSION_OVERRIDES[@]}"

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.covisibility \
  "${REPO_ROOT}/data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml" \
  root="${WAI_DIR}" \
  device="${DEVICE}"

MOGE_ARGS=(
  root="${WAI_DIR}"
  device="${DEVICE}"
)

if [[ -n "${MOGE_BATCH_SIZE}" ]]; then
  MOGE_ARGS+=("batch_size=${MOGE_BATCH_SIZE}")
fi

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.run_moge \
  "${MOGE_ARGS[@]}"
