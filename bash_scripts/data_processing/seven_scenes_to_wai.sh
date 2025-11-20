#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: seven_scenes_to_wai.sh [--datasets scene1[,scene2...]] [--device DEVICE] [--moge-batch-size N] \
                              [--moge-model PATH_OR_REPO] <processed_root> <wai_output_dir> <conda_env> \
                              [conversion overrides...]
EOF
}

DATASET_FILTER=""
DEVICE="cuda"
MOGE_BATCH_SIZE=""
MOGE_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      if [[ $# -lt 2 ]]; then usage; exit 1; fi
      DATASET_FILTER="$2"
      shift 2
      ;;
    --device)
      if [[ $# -lt 2 ]]; then usage; exit 1; fi
      DEVICE="$2"
      shift 2
      ;;
    --moge-batch-size)
      if [[ $# -lt 2 ]]; then usage; exit 1; fi
      MOGE_BATCH_SIZE="$2"
      shift 2
      ;;
    --moge-model)
      if [[ $# -lt 2 ]]; then usage; exit 1; fi
      MOGE_MODEL="$2"
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

# ---------------------------------------------------------
# 【核心修改】：定义 Python 的绝对路径
# 逻辑：不再依赖系统 path 中的 conda，而是直接指向 Miniconda3 的环境路径
# ---------------------------------------------------------
if [[ "${CONDA_ENV}" == "base" ]]; then
    PYTHON_CMD="/home/xwh/miniconda3/bin/python"
else
    PYTHON_CMD="/home/xwh/miniconda3/envs/${CONDA_ENV}/bin/python"
fi

echo "Using Python executable: ${PYTHON_CMD}"

# ---------------------------------------------------------

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

# 修改：直接使用 PYTHON_CMD 替代 conda run -n ... python
"${PYTHON_CMD}" -m wai_processing.scripts.conversion.seven_scenes \
  original_root="${PROCESSED_ROOT}" \
  root="${WAI_DIR}" \
  device="${DEVICE}" \
  "${CONVERSION_OVERRIDES[@]}"

"${PYTHON_CMD}" -m wai_processing.scripts.covisibility \
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

if [[ -n "${MOGE_MODEL}" ]]; then
  MOGE_ARGS+=("model_path=${MOGE_MODEL}")
fi

"${PYTHON_CMD}" -m wai_processing.scripts.run_moge \
  "${MOGE_ARGS[@]}"