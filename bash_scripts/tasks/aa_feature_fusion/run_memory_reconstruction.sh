#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# --- 脚本安全设置 ---
# set -e: 当命令失败时立即退出脚本
# set -u: 将未设置的变量视为错误
# set -o pipefail: 管道中的任何命令失败，整个管道都将失败
set -euo pipefail
# 导出此变量以便 Hydra (Python的配置库) 在出错时打印完整的堆栈跟踪，方便调试
export HYDRA_FULL_ERROR=1

# --- 必需的环境变量 ---
# 检查 STORED_FEATURE_FILE 是否已设置。如果未设置，脚本将退出并显示错误消息。
# STORED_FEATURE_FILE: 指向预先计算和存储的特征文件（例如 'info_sharing_outputs.pt'）的路径。
#                      这是本脚本的核心输入。
: "${STORED_FEATURE_FILE:?Set STORED_FEATURE_FILE to the info_sharing_outputs.pt path}"
# 检查 OUTPUT_ROOT 是否已设置。如果未设置，脚本将退出并显示错误消息。
# OUTPUT_ROOT: 指定一个目录，用于存放脚本生成的重建结果（工件）。
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the reconstruction artifact directory}"

# --- 路径准备 ---
# 将 STORED_FEATURE_FILE 转换为绝对路径
STORED_FEATURE_FILE=$(realpath "${STORED_FEATURE_FILE}")

# 创建输出目录（如果它尚不存在），-p 选项确保也会创建任何必需的父目录
mkdir -p "${OUTPUT_ROOT}"
# 将 OUTPUT_ROOT 也转换为绝对路径
OUTPUT_ROOT=$(realpath "${OUTPUT_ROOT}")

# --- 可选的环境变量 (带默认值) ---
# DEVICE: 指定用于计算的设备 (例如 'cuda', 'cuda:0', 'cpu')。
#         如果未设置，默认为 'cuda'。
DEVICE=${DEVICE:-cuda}
# NUM_SAMPLES: 指定要处理的样本数量。
#              如果未设置，默认为 1。
NUM_SAMPLES=${NUM_SAMPLES:-1}
# SAMPLE_INDICES: 指定要处理的特定样本索引列表 (例如 "1,5,10")。
#                 如果未设置，脚本将默认从数据集的开头开始处理。
SAMPLE_INDICES=${SAMPLE_INDICES:-}
# DATA_ROOT: 指定数据集的根目录。
#            如果未设置，Python 脚本可能会使用其配置中的默认路径。
DATA_ROOT=${DATA_ROOT:-}
# HYDRA_OVERRIDES: 允许传入任何其他的 Hydra 配置覆盖参数。
#                  这是一个“全能”变量，用于灵活修改配置。
#                  例如: HYDRA_OVERRIDES="model.some_param=value dataset.name=custom"
HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-}

# --- 构建 Hydra 命令行参数 ---
# 初始化一个数组来收集所有要传递给 Python 脚本的参数
CLI_ARGS=(
  # 传递预存特征文件的路径
  "fusion.stored_feature_file=${STORED_FEATURE_FILE}"
  # 传递输出目录的路径
  "demo.output_dir=${OUTPUT_ROOT}"
  # 传递要使用的设备
  "demo.device=${DEVICE}"
  # 传递要处理的样本数
  "demo.num_samples=${NUM_SAMPLES}"
)

# --- 动态添加可选参数 ---
# 如果 SAMPLE_INDICES 变量不是空的 (即用户提供了特定的索引)
if [[ -n "${SAMPLE_INDICES}" ]]; then
  # 将其格式化为 Hydra/Python 期望的列表格式 (例如 'demo.sample_indices=[1,5,10]')
  CLI_ARGS+=("demo.sample_indices=[${SAMPLE_INDICES}]")
fi

# 如果 DATA_ROOT 变量不是空的
if [[ -n "${DATA_ROOT}" ]]; then
  # 添加它以覆盖 Hydra 配置中的 'root_data_dir'
  CLI_ARGS+=("root_data_dir=${DATA_ROOT}")
fi

# 如果 HYDRA_OVERRIDES 变量不是空的
if [[ -n "${HYDRA_OVERRIDES}" ]]; then
  # 读取字符串并将其拆分为一个数组 (EXTRA)
  read -r -a EXTRA <<< "${HYDRA_OVERRIDES}"
  # 将 EXTRA 数组中的所有额外参数追加到 CLI_ARGS 数组中
  CLI_ARGS+=("${EXTRA[@]}")
fi

# --- 执行 Python 脚本 ---
# -m 标志告诉 Python 运行 'mapanything.tasks.aa_feature_fusion.demo' 作为主模块
# "${CLI_ARGS[@]}" 会将数组中的每个元素扩展为一个独立的、带引号的参数
# 传递给 Python 脚本。
python3 \
  -m mapanything.tasks.aa_feature_fusion.demo \
  "${CLI_ARGS[@]}"

# --- 完成 ---
echo "Single-view memory reconstructions saved to ${OUTPUT_ROOT}"