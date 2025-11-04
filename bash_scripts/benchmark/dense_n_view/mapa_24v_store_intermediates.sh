#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

set -euo pipefail
export HYDRA_FULL_ERROR=1

# Example batch size / view configuration for capturing alternating-attention features.
batch_sizes_and_views=(
    "1 100 benchmark_518_eth3d_snpp_tav2"
)

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size, num_views=$num_views (AA feature capture enabled)"

    run_dir="${root_experiments_dir}/mapanything/benchmarking/dense_${num_views}_view/mapa_24v_store_intermediates"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=1 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything_store_intermediates \
        model/task=images_and_full_geometry \
        model.encoder.uses_torch_hub=True \
        model.pretrained="${root_pretrained_checkpoints_dir}/facebook_map-anything.pth" \
        +model.memory_efficient_inference=True \
        hydra.run.dir="$run_dir"

    echo "Finished $dataset with AA feature capture. Serialized tensors live under $run_dir/aa_features."
done
