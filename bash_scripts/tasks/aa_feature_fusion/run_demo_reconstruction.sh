#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

set -euo pipefail
export HYDRA_FULL_ERROR=1

: "${STORED_FEATURE_FILE:?Set STORED_FEATURE_FILE to the info_sharing_outputs.pt path}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the reconstruction artifact directory}"

STORED_FEATURE_FILE=$(realpath "${STORED_FEATURE_FILE}")

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")

DEVICE=${DEVICE:-cuda}
NUM_SAMPLES=${NUM_SAMPLES:-1}
SAMPLE_INDICES=${SAMPLE_INDICES:-}
DATA_ROOT=${DATA_ROOT:-}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}

EXTRA_OVERRIDES=()
if [[ -n "${HYDRA_OVERRIDES}" ]]; then
  read -r -a EXTRA_OVERRIDES <<< "${HYDRA_OVERRIDES}"
fi

PY_ARGS=(
  "fusion.stored_feature_file=${STORED_FEATURE_FILE}"
  "demo.output_dir=${OUTPUT_ROOT}"
  "demo.device=${DEVICE}"
  "demo.num_samples=${NUM_SAMPLES}"
)

if [[ -n "${SAMPLE_INDICES}" ]]; then
  PY_ARGS+=("demo.sample_indices=[${SAMPLE_INDICES}]")
fi

if [[ -n "${DATA_ROOT}" ]]; then
  PY_ARGS+=("root_data_dir=${DATA_ROOT}")
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  PY_ARGS+=("${EXTRA_OVERRIDES[@]}")
fi

python3 \
  -m mapanything.tasks.aa_feature_fusion.demo \
  "${PY_ARGS[@]}"

echo "Reconstruction artifacts written to ${OUTPUT_ROOT}"
