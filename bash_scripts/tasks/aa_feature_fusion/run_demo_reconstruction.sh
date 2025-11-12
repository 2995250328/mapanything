#!/bin/bash
set -euo pipefail
export HYDRA_FULL_ERROR=1

# ------------------------------------------------------------------
# Define the combinations you want to run
# Format: "NUM_SAMPLES VIEWS_PER_SAMPLE DATASET_NAME"
# ------------------------------------------------------------------
combinations=(
  "24 1 seven_scenes_wai/test/default"
  # 例如继续添加：
  # "50 2 seven_scenes_wai/test/default"
  # "100 2 seven_scenes_wai/test/default"
)

# Root dirs
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUTPUT_ROOT="${PROJECT_ROOT}/dataset_runs"
DATA_ROOT="${PROJECT_ROOT}"

mkdir -p "$OUTPUT_ROOT"

# ------------------------------------------------------------------
# Loop over combinations
# ------------------------------------------------------------------
for combo in "${combinations[@]}"; do
    read -r num_samples views_per_sample dataset <<< "$combo"

    echo "Running reconstruction:"
    echo "  dataset = $dataset"
    echo "  num_samples = $num_samples"
    echo "  views_per_sample = $views_per_sample"
    echo

    python3 -m mapanything.tasks.aa_feature_fusion.dataset_reconstruction \
        reconstruction.output_dir="${OUTPUT_ROOT}" \
        reconstruction.device="cuda" \
        reconstruction.num_samples="${num_samples}" \
        reconstruction.views_per_sample="${views_per_sample}" \
        dataset="${dataset}" \
        root_data_dir="${DATA_ROOT}" \
        hydra.run.dir="${OUTPUT_ROOT}/runs/num_${num_samples}_view_${views_per_sample}"

    echo "Finished combination: num_samples=$num_samples, views_per_sample=$views_per_sample"
    echo
done

echo "All reconstruction runs complete!"
