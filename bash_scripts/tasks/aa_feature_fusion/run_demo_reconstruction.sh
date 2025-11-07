#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

set -euo pipefail
export HYDRA_FULL_ERROR=1

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage: run_demo_reconstruction.sh <stored_feature_file> <output_root> [overrides...]

Environment overrides:
  DEVICE=cuda              # Torch device for the demo run
  NUM_SAMPLES=1            # Number of dataset items to process
  SAMPLE_INDICES=0,5,7     # Optional comma-separated indices (overrides NUM_SAMPLES)
  DATA_ROOT=/path/to/wai   # Optional WAI dataset root passed as root_data_dir
USAGE
  exit 1
fi

STORED_FEATURE_FILE=$(realpath "$1")
OUTPUT_ROOT=$(realpath "$2")
shift 2

mkdir -p "${OUTPUT_ROOT}"

DEVICE=${DEVICE:-cuda}
NUM_SAMPLES=${NUM_SAMPLES:-1}
SAMPLE_INDICES=${SAMPLE_INDICES:-}
DATA_ROOT=${DATA_ROOT:-}

HYDRA_ARGS=(
  "fusion.stored_feature_file=${STORED_FEATURE_FILE}"
  "demo.output_dir=${OUTPUT_ROOT}"
  "demo.device=${DEVICE}"
  "demo.num_samples=${NUM_SAMPLES}"
)

if [[ -n "${SAMPLE_INDICES}" ]]; then
  HYDRA_ARGS+=("demo.sample_indices=[${SAMPLE_INDICES}]")
fi

if [[ -n "${DATA_ROOT}" ]]; then
  HYDRA_ARGS+=("root_data_dir=${DATA_ROOT}")
fi

HYDRA_ARGS+=("$@")

python3 -m mapanything.tasks.aa_feature_fusion.demo \
  "${HYDRA_ARGS[@]}"

echo "Reconstruction artifacts written to ${OUTPUT_ROOT}"
