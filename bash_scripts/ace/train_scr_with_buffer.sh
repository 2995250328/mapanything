#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

set -euo pipefail
export HYDRA_FULL_ERROR=1

: "${STORED_FEATURE_FILE:?Set STORED_FEATURE_FILE to the info_sharing_outputs.pt path}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the SCR training directory}"

STORED_FEATURE_FILE=$(realpath "${STORED_FEATURE_FILE}")

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")

DEVICE=${DEVICE:-cuda}
NUM_SAMPLES=${NUM_SAMPLES:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
BUFFER_CAPACITY=${BUFFER_CAPACITY:-256}
HIDDEN_DIM=${HIDDEN_DIM:-128}
SAMPLE_INDICES=${SAMPLE_INDICES:-}
DATA_ROOT=${DATA_ROOT:-}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}

CLI_ARGS=(
  "fusion.stored_feature_file=${STORED_FEATURE_FILE}"
  "training.output_dir=${OUTPUT_ROOT}"
  "training.device=${DEVICE}"
  "training.num_samples=${NUM_SAMPLES}"
  "training.batch_size=${BATCH_SIZE}"
  "buffer.capacity=${BUFFER_CAPACITY}"
  "scr_head.hidden_dim=${HIDDEN_DIM}"
)

if [[ -n "${SAMPLE_INDICES}" ]]; then
  CLI_ARGS+=("training.sample_indices=[${SAMPLE_INDICES}]")
fi

if [[ -n "${DATA_ROOT}" ]]; then
  CLI_ARGS+=("root_data_dir=${DATA_ROOT}")
fi

if [[ -n "${HYDRA_OVERRIDES}" ]]; then
  read -r -a EXTRA <<< "${HYDRA_OVERRIDES}"
  CLI_ARGS+=("${EXTRA[@]}")
fi

python3 \
  -m mapanything.tasks.aa_feature_fusion.scr_training \
  "${CLI_ARGS[@]}"

echo "SCR head checkpoint written to ${OUTPUT_ROOT}"
