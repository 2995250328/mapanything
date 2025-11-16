# ACE 回归头再训练与测试指南

本文档说明针对 ACE 相关流程的改动、新增脚本以及训练/测试用法。

## 主要改动

1. **模型新增接口**：在 `mapanything/models/mapanything/model.py` 中增加了 `forward_with_memory_dense_feature`，只执行编码与 memory 融合，返回融合后的稠密特征和 scale token，便于冻结主干后单独训练回归头。
2. **ACE 训练任务**：`mapanything/tasks/ace/train.py` 采用 ACE 的 replay buffer 思路：为每个视角随机采样像素，将融合特征、下采样后的 GT 场景坐标、像素坐标、相机内参/位姿放入固定容量的缓冲区，再基于这些样本训练回归头（重投影 + 置信度 + XYZ 损失）。
3. **回归头实现与加载**：`mapanything/tasks/ace/regression_head.py` 定义了 `ACERegressionHead`（输出 XYZ+confidence），并提供 `load_regression_head`，支持先加载完整 mapanything 参数，再单独加载回归头，避免保存完整 pt。
4. **推理/评测流程**：`mapanything/tasks/ace/test.py` 基于训练得到的回归头进行推理并输出指标、预测结果；与 ACE 训练同样依赖 memory 特征。
5. **配置与脚本**：新增 `configs/ace/train.yaml`、`configs/ace/test.yaml` 以及 `bash_scripts/tasks/ace_train.sh`、`bash_scripts/tasks/ace_test.sh`，所有参数均可通过 yaml + shell 传入，默认路径与 `configs/memory.yaml` 保持一致的机器目录占位符。

## 训练脚本使用

1. 准备 AA memory 文件（`info_sharing_outputs.pt`）并设置环境变量：

   ```bash
   export STORED_FEATURE_FILE=/path/to/info_sharing_outputs.pt
   export OUTPUT_DIR=./outputs/ace_train_run
   export PRETRAINED=/path/to/facebook_map-anything.pth  # 可选
   ```

2. 运行启动脚本（可通过 `HYDRA_OVERRIDES` 追加覆盖，如修改 batch_size）：

   ```bash
   bash bash_scripts/tasks/ace_train.sh
   # 示例：bash bash_scripts/tasks/ace_train.sh HYDRA_OVERRIDES="training.batch_size=2 head.hidden_dim=64"
   ```

3. 训练会：
   - 使用 `forward_with_memory_dense_feature` 提取融合特征；
   - 将 GT `pts3d`、`valid_mask` 下采样至特征分辨率；
   - 从每个视角随机采样 `training.samples_per_view` 个像素（受 `training.buffer_capacity` 和 `training.max_buffer_views` 限制），把特征/GT/像素/相机参数写入 buffer；
   - 对 buffer 中的样本计算重投影误差 + 置信度 BCE + XYZ L1；
   - 只保存回归头权重（`ace_regression_head.pt`），并记录输入通道数。

## 推理/测试流程

1. 设置必要环境变量：

   ```bash
   export STORED_FEATURE_FILE=/path/to/info_sharing_outputs.pt
   export REGRESSION_HEAD=/path/to/ace_regression_head.pt
   export OUTPUT_DIR=./outputs/ace_test_run
   ```

2. 运行：

   ```bash
   bash bash_scripts/tasks/ace_test.sh
   ```

3. 脚本会自动加载 mapanything 预训练权重、回归头权重，运行指定数量的样本（`eval.max_samples`），输出：
   - 每个样本的预测三维坐标与置信度（`sample_xxxxx.pt`）；
   - 聚合指标（重投影/XYZ/置信度损失均值）保存在 `summary.json`。

## 参数定制

- `configs/ace/train.yaml` 与 `configs/ace/test.yaml` 提供完整默认参数，包括数据集构造字符串、损失权重、buffer 设置等。可通过 shell 覆盖任意 hydra 参数，例如：

  ```bash
  HYDRA_OVERRIDES="dataset.dataset_str='SevenScenesWAI(num_views=1, split=\"train\", ...)' training.max_epochs=10"
  ```

- 若需要修改 head 宽度或损失权重，直接在配置或 `HYDRA_OVERRIDES` 中调整 `head.hidden_dim`、`loss.*`。

## 结果加载/组合

- 训练输出的 `ace_regression_head.pt` 仅包含回归头参数与 `in_channels` 元信息。
- 推理时先按需加载完整 mapanything checkpoint（`model.pretrained`），再使用 `load_regression_head` 将回归头权重载入 `ACERegressionHead`，实现大模型与轻量头分离存储。

以上流程即可完成 ACE buffer 方式的回归头再训练、评测与参数管理。
