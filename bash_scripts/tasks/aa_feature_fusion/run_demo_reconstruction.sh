#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

set -euo pipefail
export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# ---------------------------------------------------------------------------
# User-adjustable defaults. Edit the variables in this block to change
# standard behaviour without having to export a long list of environment
# variables for every invocation. Environment variables with the same name
# still take precedence over the values declared here.
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT="${WAI_ROOT:-${PROJECT_ROOT}}"
DEFAULT_OUTPUT_ROOT="${DEFAULT_DATA_ROOT}/dataset_runs"
DEFAULT_DEVICE="cuda"
DEFAULT_NUM_SAMPLES=6
DEFAULT_VIEWS_PER_SAMPLE=""
DEFAULT_START_INDEX=0
DEFAULT_MAX_INDEX=""
DEFAULT_SAMPLE_INDICES=""
DEFAULT_HYDRA_OVERRIDES=""
# ---------------------------------------------------------------------------

OUTPUT_ROOT=${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}
DEVICE=${DEVICE:-${DEFAULT_DEVICE}}
NUM_SAMPLES=${NUM_SAMPLES:-${DEFAULT_NUM_SAMPLES}}
VIEWS_PER_SAMPLE=${VIEWS_PER_SAMPLE:-${DEFAULT_VIEWS_PER_SAMPLE}}
SAMPLE_INDICES=${SAMPLE_INDICES:-${DEFAULT_SAMPLE_INDICES}}
START_INDEX=${START_INDEX:-${DEFAULT_START_INDEX}}
MAX_INDEX=${MAX_INDEX:-${DEFAULT_MAX_INDEX}}
DATA_ROOT=${DATA_ROOT:-${DEFAULT_DATA_ROOT}}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-${DEFAULT_HYDRA_OVERRIDES}}

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")

CLI_ARGS=(
  "reconstruction.output_dir=${OUTPUT_ROOT}"
  "reconstruction.device=${DEVICE}"
  "reconstruction.num_samples=${NUM_SAMPLES}"
  "reconstruction.start_index=${START_INDEX}"
)

if [[ -n "${MAX_INDEX}" ]]; then
  CLI_ARGS+=("reconstruction.max_index=${MAX_INDEX}")
fi

if [[ -n "${VIEWS_PER_SAMPLE}" ]]; then
  CLI_ARGS+=("reconstruction.views_per_sample=${VIEWS_PER_SAMPLE}")
fi

if [[ -n "${SAMPLE_INDICES}" ]]; then
  CLI_ARGS+=("reconstruction.sample_indices=[${SAMPLE_INDICES}]")
fi

CLI_ARGS+=("root_data_dir=${DATA_ROOT}")

if [[ -n "${HYDRA_OVERRIDES}" ]]; then
  read -r -a EXTRA <<< "${HYDRA_OVERRIDES}"
  CLI_ARGS+=("${EXTRA[@]}")
fi

python3 \
  -m mapanything.tasks.aa_feature_fusion.dataset_reconstruction \
  "${CLI_ARGS[@]}"

echo "Dataset reconstructions saved to ${OUTPUT_ROOT}"
