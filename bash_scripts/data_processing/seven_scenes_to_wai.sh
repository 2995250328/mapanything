#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <processed_root> <wai_output_dir> <conda_env> [conversion overrides...]" >&2
  echo "  processed_root 目录需要包含 pgt_7scenes_* 子目录 (train/test/calibration/depth/poses/rgb)。" >&2
  exit 1
fi

PROCESSED_ROOT=$(realpath "$1")
WAI_DIR=$(realpath "$2")
CONDA_ENV="$3"
shift 3

CONVERSION_OVERRIDES=("$@")

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

mkdir -p "${WAI_DIR}"

set -x

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.conversion.seven_scenes \
  original_root="${PROCESSED_ROOT}" \
  root="${WAI_DIR}" \
  "${CONVERSION_OVERRIDES[@]}"

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.covisibility \
  "${REPO_ROOT}/data_processing/wai_processing/configs/covisibility/covisibility_gt_depth_224x224.yaml" \
  root="${WAI_DIR}"

conda run -n "${CONDA_ENV}" \
  python -m wai_processing.scripts.run_moge \
  root="${WAI_DIR}" \
  batch_size=1
