#!/bin/bash

export HYDRA_FULL_ERROR=1

batch_sizes_and_views=(
    "1 1 ace_dataset"
)
# 3. 输入特征文件 (使用插值变量 ${root_experiments_dir})
FEATURE_FILE='/home/xwh/project/map-anything-experiments/ace_tasks/mapanything/benchmark_518_seven_scenes_24v_aa_capture/ACE/aa_blocks/1125T0205/7Scenes_chess_train_24v_batch0000_intermediates.pt/7Scenes_chess_train_24v_batch0000_intermediates.pt'

for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"
    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"
    python3  \
        mapanything/tasks/test_downsample.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=1 \
        dataset.num_views=$num_views \
        dataset.sequential_view_mode=True \
        batch_size=$batch_size \
        device=cuda:3 \
        head.type="ace_homogeneous"\
        head.checkpoint="/home/xwh/project/map-anything-experiments/mapanything/memory/ace_dataset/ace_regression/ace-downsample-debug1_task-images_only_head-ace_homogeneous_loss-reproj_scale-on_ep64_buf10240000.pt" \
        model=mapanything_memory \
        model/task=images_only \
        model.encoder.uses_torch_hub=True \
        model.pretrained='${root_pretrained_checkpoints_dir}/facebook_map-anything.pth' \
        +model.memory_efficient_inference=True \
        fusion.stored_feature_file=${FEATURE_FILE} \
        hydra.run.dir='${root_experiments_dir}/map-anything/memory/'"${dataset}"''
done
