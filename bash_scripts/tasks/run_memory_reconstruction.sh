#!/bin/bash

export HYDRA_FULL_ERROR=1

batch_sizes_and_views=(
    "1 1 benchmark_518_seven_scenes"
)
# 3. 输入特征文件 (使用插值变量 ${root_experiments_dir})
FEATURE_FILE='/home/xwh/project/map-anything-experiments/ace_tasks/map-anything/benchmark_518_seven_scenes_100v_aa_capture/ACE/aa_blocks/1113T09151762996545/7Scenes_chess_test_100v_intermediates.pt/7Scenes_chess_test_100v_intermediates.pt'

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"
    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"
    python3  \
        map-anything/tasks/demo.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=1 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=mapanything_memory \
        model/task=images_only \
        model.encoder.uses_torch_hub=True \
        model.pretrained='${root_pretrained_checkpoints_dir}/facebook_map-anything.pth' \
        +model.memory_efficient_inference=False \
        fusion.stored_feature_file=${FEATURE_FILE} \
        hydra.run.dir='${root_experiments_dir}/map-anything/memory/'"${dataset}"''
done
