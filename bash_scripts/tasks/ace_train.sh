#!/bin/bash

export HYDRA_FULL_ERROR=1

batch_sizes_and_views=(
    "1 1 ace_dataset"
)
# 3. 输入特征文件 (使用插值变量 ${root_experiments_dir})
FEATURE_FILE='/home/xwh/project/map-anything-experiments/ace_tasks/mapanything/benchmark_518_seven_scenes_48v_aa_capture/ACE/aa_blocks/1118T0514/7Scenes_chess_train_48v_intermediates.pt/7Scenes_chess_train_48v_intermediates.pt'

#model/task=calibrated_sfm            images_only
for combo in "${batch_sizes_and_views[@]}"; do
    read -r batch_size num_views dataset <<< "$combo"
    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"
    python3  \
        mapanything/tasks/train.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=1 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        loss.mode=xyz \
        loss.scale_reg.enabled=False \
        loss.scale_reg.variant=match_to_gt\
        model.head_mode="film" \
        training.device=cuda:0 \
        model=mapanything_memory \
        model/task=images_only \
        model.encoder.uses_torch_hub=True \
        model.pretrained='${root_pretrained_checkpoints_dir}/facebook_map-anything.pth' \
        +model.memory_efficient_inference=True \
        fusion.stored_feature_file=${FEATURE_FILE} \
        hydra.run.dir='${root_experiments_dir}/mapanything/memory/'"${dataset}"''
done
