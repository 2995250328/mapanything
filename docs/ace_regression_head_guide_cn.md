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

# 对于loss_fn中关于Z轴信息正则化的合理性分析
## 1) 透视成像的尺度不变性：重投影对整体缩放不敏感
- 透视投影：
u = fx * X/Z + cx
v = fy * Y/Z + cy
- 若整体缩放 `(X, Y, Z) -> s · (X, Y, Z)`，则 `X/Z`、`Y/Z` 不变，像素 `(u, v)` 不变。  
- **结论**：仅用重投影误差无法确定绝对尺度（单目固有的尺度歧义），需要额外的“定标”约束。

## 2) 尺度只体现在“深度”上是**充分且必要**的
- 像素由比值 `X/Z, Y/Z` 决定；整体缩放对 `(u, v)` 无影响。  
- 只要规定一个“深度尺度”（如“预测深度中位数”匹配 GT/单位/先验），就等价于固定全局尺度。  
- **结论**：无需同时正则 `X、Y`——**管住 Z 的整体尺度即可**。

## 3) 相机系 z 是最稳定、最少受姿态影响的尺度指标
- 世界系的 `X、Y` 强依赖相机位姿/场景布局/坐标原点，不同帧差异大；拿它们做尺度约束易引偏。  
- 相机系 `z` 是“沿光轴距离”，与成像尺度直接相关，物理含义明确、鲁棒性高。  
- 采用 **中位数** 作为统计量可抵抗离群值与遮挡/错误预测的干扰。

## 4) 避免不必要的几何耦合
- 若对 `X、Y` 也做尺度正则，会隐含“横向分布先验”，而横向分布强受位姿/结构影响，不应被约束。  
- **仅对 z 做温和约束**，最小化与位姿/结构的耦合，把正则作用精准落在“解决尺度歧义”。

## 5) 与 ACE 的训练/失效模式匹配
- ACE 的重投影分支：主损是像素误差，对整体尺度不敏感；因此需附加与尺度直接相关的项——**选择 z** 恰到好处。  
- “无尺度 + scale”方案（Decoupled head）：只对**无尺度预测的相机深度**做 unit 正则，避免 `scale` 漫游；同样体现“只动 z 就够”。

---

### 小结
只对相机系的 **z（深度）** 做尺度正则，既能**打破单目尺度不确定性**，又不会额外束缚横向几何；这是**最小侵入、几何正确、工程稳定**的选择。
