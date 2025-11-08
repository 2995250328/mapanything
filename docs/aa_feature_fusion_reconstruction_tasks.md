# AA 特征融合重建与 SCR 训练任务指南

本文聚焦于仓库中新引入的三个脚本：

1. **数据集重建 Demo**（`run_demo_reconstruction.sh`）：直接从数据集中抽取多视图样本执行 MapAnything 推理。
2. **记忆驱动重建**（`run_memory_reconstruction.sh`）：复用 `info_sharing_outputs.pt` 中缓存的 AA 块，对单帧输入进行增强重建。
3. **ACE 缓冲区 SCR 训练**（`train_scr_with_buffer.sh`）：将 AA 融合后的密集特征写入训练缓冲区，再拟合一个轻量级的场景坐标回归头。

## 1. 数据集多视图重建

脚本位置：`bash_scripts/tasks/aa_feature_fusion/run_demo_reconstruction.sh`

示例命令：

```bash
bash bash_scripts/tasks/aa_feature_fusion/run_demo_reconstruction.sh
```

- 脚本顶部提供了一组“用户可编辑默认值”，可直接在文件中设置 `OUTPUT_ROOT`、`DATA_ROOT`、`NUM_SAMPLES`、`VIEWS_PER_SAMPLE` 等关键参数；如需临时覆盖，可继续通过同名环境变量传参。
- `VIEWS_PER_SAMPLE` 控制从数据集中抽取的视角数量；默认为脚本中的 `DEFAULT_VIEWS_PER_SAMPLE`，留空则回落到配置文件的 `dataset.num_views`。
- `START_INDEX`/`MAX_INDEX` 或 `SAMPLE_INDICES` 用于选择场景。
- 输出目录下会生成按场景划分的子文件夹，每个子文件夹包含 `reconstruction.pt` 与跨样本汇总的 `summary.json`。【F:bash_scripts/tasks/aa_feature_fusion/run_demo_reconstruction.sh†L1-L78】【F:mapanything/tasks/aa_feature_fusion/dataset_reconstruction.py†L1-L176】

内部流程：脚本会构建原生 `MapAnything` 模型，逐条加载数据集条目，将必要的几何信息（光线、深度、位姿、尺度标记等）转换为张量后调用 `model.forward` 获取每个视角的密集输出，并可选附带基于真值深度计算的点云目标。【F:mapanything/tasks/aa_feature_fusion/dataset_reconstruction.py†L20-L176】【F:configs/tasks/aa_feature_fusion/dataset_demo.yaml†L1-L17】

## 2. 记忆增强的单视图重建

脚本位置：`bash_scripts/tasks/aa_feature_fusion/run_memory_reconstruction.sh`

示例命令：

```bash
STORED_FEATURE_FILE=/path/to/info_sharing_outputs.pt \
OUTPUT_ROOT="$WAI_ROOT/memory_runs" \
DEVICE=cuda NUM_SAMPLES=8 DATA_ROOT="$WAI_ROOT" \
HYDRA_OVERRIDES="model.pretrained=/path/to/mapanything.ckpt" \
bash bash_scripts/tasks/aa_feature_fusion/run_memory_reconstruction.sh
```

- 需要先通过 `ace_store_intermediates.sh` 等流程生成 `info_sharing_outputs.pt`。
- 其余参数与旧版 Demo 保持一致，仍支持 `SAMPLE_INDICES` 定位单个样本。
- Python 端逻辑沿用 `mapanything.tasks.aa_feature_fusion.demo`，会加载 AA 记忆块、执行 token 融合与下游头推理，并序列化重建结果。【F:bash_scripts/tasks/aa_feature_fusion/run_memory_reconstruction.sh†L1-L61】【F:mapanything/tasks/aa_feature_fusion/demo.py†L17-L207】

## 3. ACE 缓冲区驱动的 SCR 训练

脚本位置：`bash_scripts/ace/train_scr_with_buffer.sh`

示例命令：

```bash
STORED_FEATURE_FILE=/path/to/info_sharing_outputs.pt \
OUTPUT_ROOT="$WAI_ROOT/scr_training" \
DEVICE=cuda NUM_SAMPLES=48 BATCH_SIZE=6 BUFFER_CAPACITY=384 \
REPROJECTION_WEIGHT=1.0 XYZ_WEIGHT=0.1 RELATIVE_WEIGHT=0.1 \
HYDRA_OVERRIDES="model.pretrained=/path/to/mapanything.ckpt" \
bash bash_scripts/ace/train_scr_with_buffer.sh
```

- 训练脚本首先调用 `mapanything.tasks.aa_feature_fusion.scr_training` 构建融合流水线，逐个样本收集 DPT 密集特征、世界坐标、相机内外参以及场景均值位姿，并写入 ACE 风格的 FIFO 缓冲区。【F:bash_scripts/ace/train_scr_with_buffer.sh†L1-L79】【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L29-L213】
- `scr_head.hidden_dim`、`buffer.capacity` 等参数可通过环境变量或 `HYDRA_OVERRIDES` 自定义。
- 缓冲区会按样本构建 `DataLoader`，采用 ACE 论文中的重投影主损失，辅以可调的 XYZ 与相对坐标正则项，最终在 `OUTPUT_ROOT` 下产出 `scr_head.pt` 检查点。【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L88-L213】【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L215-L311】

### 训练缓冲区设计要点

- `ACETrainingBuffer` 以固定长度 deque 保存包含特征图、世界/相对坐标、相机参数与均值位姿的字典条目，容量溢出时自动覆盖最旧样本。【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L35-L120】
- 缓冲构建阶段会为每个场景自动缓存/复用 `statistics/mean_pose.json`，确保均值位姿仅在首次访问时计算，后续直接读取。【F:mapanything/tasks/aa_feature_fusion/pose_stats.py†L12-L109】【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L237-L283】
- 由于重用 AA 记忆块，`fusion.stored_feature_file` 为必填项；脚本会在 GPU 不可用时自动降级到 CPU 以确保可运行性。【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L315-L347】
- `SCRRegressionHead` 默认输出三通道 XYZ，可根据任务需求在配置中改写输出维度或隐藏层大小。【F:mapanything/tasks/aa_feature_fusion/scr_training.py†L88-L108】【F:configs/tasks/aa_feature_fusion/scr_train.yaml†L1-L30】

通过上述三个脚本，可以快速完成“多视图数据载入 → 单视图记忆重建 → SCR 回归训练”的闭环流程，并在 `docs/aa_feature_fusion_workflow.md` 的基础上拓展更多 ACE 相关实验。 
