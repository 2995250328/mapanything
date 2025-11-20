#!/bin/bash

# Query-only ACE regression training (no intermediate memory fusion).
# Requires OUTPUT_ROOT and PRETRAINED_CKPT to be set before invocation.
#
# The defaults mirror the original ACE launch script: you can sweep over
# (batch_size, num_views, dataset) tuples via BATCH_VIEWS_GRID. Each tuple is
# a space-separated triple, e.g.:
#   export BATCH_VIEWS_GRID="1 1 ace_dataset  2 4 ace_dataset"
# Leave empty to run a single job with the configured BATCH_SIZE / NUM_VIEWS.

set -euo pipefail
export HYDRA_FULL_ERROR=1

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the directory for checkpoints and logs}" 
: "${PRETRAINED_CKPT:?Set PRETRAINED_CKPT to the map-anything checkpoint path}"

OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")
PRETRAINED_CKPT=$(realpath "${PRETRAINED_CKPT}")
mkdir -p "${OUTPUT_ROOT}"

DEVICE=${DEVICE:-cuda}
DATASET=${DATASET:-ace_dataset}
NUM_VIEWS=${NUM_VIEWS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MODEL=${MODEL:-mapanything_memory}
MODEL_TASK=${MODEL_TASK:-images_only}
FUSION_PLACEHOLDER=${FUSION_PLACEHOLDER:-/dev/null}
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}
BATCH_VIEWS_GRID=${BATCH_VIEWS_GRID:-}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-}
FEATURE_FILE=${FEATURE_FILE:-}

run_job() {
  local batch_size="$1"
  local num_views="$2"
  local dataset="$3"

  CLI_ARGS=(
    "machine=aws"
    "dataset=${dataset}"
    "dataset.num_views=${num_views}"
    "training.batch_size=${batch_size}"
    "model=${MODEL}"
    "model/task=${MODEL_TASK}"
    "model.pretrained=${PRETRAINED_CKPT}"
    "fusion.stored_feature_file=${FUSION_PLACEHOLDER}"
    "training.output_dir=${OUTPUT_ROOT}"
    "training.device=${DEVICE}"
  )

  # Optional path overrides
  if [[ -n "${ROOT_DATA_DIR:-}" ]]; then
    CLI_ARGS+=("root_data_dir=${ROOT_DATA_DIR}")
  fi
  if [[ -n "${METADATA_ROOT:-}" ]]; then
    CLI_ARGS+=("mapanything_dataset_metadata_dir=${METADATA_ROOT}")
  fi
  if [[ -n "${EXPERIMENTS_ROOT:-}" ]]; then
    CLI_ARGS+=("root_experiments_dir=${EXPERIMENTS_ROOT}")
  fi
  if [[ -n "${PRETRAINED_ROOT:-}" ]]; then
    CLI_ARGS+=("root_pretrained_checkpoints_dir=${PRETRAINED_ROOT}")
  fi
  if [[ -n "${UNICEPTION_ROOT:-}" ]]; then
    CLI_ARGS+=("root_uniception_pretrained_checkpoints_dir=${UNICEPTION_ROOT}")
  fi

  # Optional training hyperparameters
  if [[ -n "${BUFFER_SIZE:-}" ]]; then
    CLI_ARGS+=("training.buffer_size=${BUFFER_SIZE}")
  fi
  if [[ -n "${EPOCHS:-}" ]]; then
    CLI_ARGS+=("training.epochs=${EPOCHS}")
  fi
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    CLI_ARGS+=("training.num_workers=${NUM_WORKERS}")
  fi
  if [[ -n "${LR:-}" ]]; then
    CLI_ARGS+=("training.lr=${LR}")
  fi
  if [[ -n "${WEIGHT_DECAY:-}" ]]; then
    CLI_ARGS+=("training.weight_decay=${WEIGHT_DECAY}")
  fi
  if [[ -n "${SAMPLES_PER_VIEW:-}" ]]; then
    CLI_ARGS+=("training.samples_per_view=${SAMPLES_PER_VIEW}")
  fi

  # Optional feature file placeholder (keeps Hydra happy even though query-only)
  if [[ -n "${FEATURE_FILE}" ]]; then
    CLI_ARGS+=("fusion.stored_feature_file=${FEATURE_FILE}")
  fi

  # Extra hydra overrides (space-separated string)
  if [[ -n "${HYDRA_OVERRIDES}" ]]; then
    read -r -a EXTRA <<< "${HYDRA_OVERRIDES}"
    CLI_ARGS+=("${EXTRA[@]}")
  fi

  # Optional hydra run dir override
  if [[ -n "${HYDRA_RUN_DIR}" ]]; then
    CLI_ARGS+=("hydra.run.dir=${HYDRA_RUN_DIR}")
  fi

  echo "Running ${dataset} with batch_size=${batch_size}, num_views=${num_views}"
  python3 -m mapanything.tasks.train_query_only "${CLI_ARGS[@]}"
}

if [[ -n "${BATCH_VIEWS_GRID}" ]]; then
  read -r -a GRID <<< "${BATCH_VIEWS_GRID}"
  if (( ${#GRID[@]} % 3 != 0 )); then
    echo "BATCH_VIEWS_GRID must contain triples: <batch_size> <num_views> <dataset>" >&2
    exit 1
  fi
  for ((i=0; i<${#GRID[@]}; i+=3)); do
    run_job "${GRID[i]}" "${GRID[i+1]}" "${GRID[i+2]}"
  done
else
  run_job "${BATCH_SIZE}" "${NUM_VIEWS}" "${DATASET}"
fi
