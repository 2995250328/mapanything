# 7Scenes 交替注意力特征融合功能介绍

本文汇总了仓库中新加入的交替注意力（Alternating-Attention，AA）特征融合组件、配置与脚本，帮助你从捕获多视图记忆、到单视图重建 Demo、再到后续训练/评估任务的完整流程。

## 功能总览

- **特征记忆载入与复原**：`StoredAAFeatureSequence` 可以直接从 `info_sharing_outputs.pt` 中重建 AA 块的特征序列（含附加 token 与空间尺寸），供后续注意力模块复用。【F:mapanything/tasks/aa_feature_fusion/fusion.py†L41-L141】
- **单视图-多记忆融合模块**：`AAFeatureFusionModule` 将单视图编码后的 token 作为查询，仅对其执行帧内注意力，并把存储的多视图记忆作为全局注意力的键值，实现查询与记忆的交互。【F:mapanything/tasks/aa_feature_fusion/fusion.py†L143-L274】
- **流水线封装**：`AAFeatureFusionPipeline` 负责根据配置筛选单视图模态、抽取编码 token，并调用融合模块输出可被下游头部消费的特征图。【F:mapanything/tasks/aa_feature_fusion/pipeline.py†L12-L53】
- **可复用的构建入口**：`build_pipeline_from_cfg` 将 Hydra 配置映射为 MapAnything 主干模型与 AA 融合模块，方便训练、评估与 Demo 共用同一套组件。【F:mapanything/tasks/aa_feature_fusion/builder.py†L12-L32】

## 数据与存储准备

1. **转换 7Scenes 数据为 WAI 布局**：按照《7Scenes WAI 数据流水线指南》下载原始数据、伪真值，并调用 `seven_scenes_to_wai.sh` 生成 `wai_data/7scenes` 目录。【F:docs/seven_scenes_wai_pipeline.md†L1-L22】【F:bash_scripts/data_processing/seven_scenes_to_wai.sh†L1-L156】
2. **捕获多视图 AA 中间变量**：使用 `ace_store_intermediates.sh`（或 `mapa_24v_store_intermediates.sh`）在 7Scenes 上运行 MapAnything 推理，将所有 AA 块写入单个 `info_sharing_outputs.pt` 文件。【F:bash_scripts/ace/ace_store_intermediates.sh†L1-L78】
3. **准备单视图输入配置**：`configs/model/task/single_view_optional_intrinsics.yaml` 仅保留图像与可选内参，避免在融合阶段重复加载多余的几何信号。【F:configs/model/task/single_view_optional_intrinsics.yaml†L1-L18】

## Demo：单视图重建示例

1. **运行脚本**：
   ```bash
   DEVICE=cuda NUM_SAMPLES=4 DATA_ROOT="$WAI_ROOT" \
   bash bash_scripts/tasks/aa_feature_fusion/run_demo_reconstruction.sh \
       /path/to/info_sharing_outputs.pt \
       "$WAI_ROOT/demo_runs" \
       model.pretrained=/path/to/mapanything.ckpt
   ```
   - 如需指定数据索引，可通过 `SAMPLE_INDICES=0,5,7` 覆盖，脚本会自动忽略 `NUM_SAMPLES`。
   - 额外的 Hydra 覆盖（如 `model.pretrained`、`root_data_dir`）继续以命令行参数形式追加。
   脚本会自动把存储文件传给 Hydra 配置，输出目录中会生成每个样本对应的 `.pt` 记录与 `summary.json`。【F:bash_scripts/tasks/aa_feature_fusion/run_demo_reconstruction.sh†L1-L55】【F:configs/tasks/aa_feature_fusion/demo.yaml†L1-L18】
2. **内部流程**：Demo 入口会加载 7Scenes 测试集的单视图样本，按配置取用内参/深度/位姿信息，然后通过融合模块与 MapAnything 的下游头部恢复点云或深度图，并序列化到磁盘。【F:mapanything/tasks/aa_feature_fusion/demo.py†L17-L137】【F:mapanything/tasks/aa_feature_fusion/demo.py†L139-L207】
3. **可视化**：在 WSL 或本地机器执行 `python scripts/visualization/view_aa_fusion_demo.py sample.pt --save sample.png --export-pts sample.ply` 即可预览 RGB 与重建深度，并可选导出 PLY 点云。【F:scripts/visualization/view_aa_fusion_demo.py†L1-L96】

## 训练与评估脚手架

- **训练入口**：`python -m mapanything.tasks.aa_feature_fusion.train fusion.stored_feature_file=/path/to/info_sharing_outputs.pt` 将构建流水线并打印占位训练信息，后续可在 `run_training` 中补全优化逻辑。【F:mapanything/tasks/aa_feature_fusion/train.py†L1-L36】【F:configs/tasks/aa_feature_fusion/train.yaml†L1-L19】
- **评估入口**：`python -m mapanything.tasks.aa_feature_fusion.eval fusion.stored_feature_file=/path/to/info_sharing_outputs.pt` 会按照 `test.yaml` 载入数据与流水线，为日后集成指标计算留出接口。【F:mapanything/tasks/aa_feature_fusion/eval.py†L1-L39】【F:configs/tasks/aa_feature_fusion/test.yaml†L1-L11】
- **共享默认配置**：`configs/tasks/aa_feature_fusion/default.yaml` 统一描述模型、数据集与融合参数，可通过命令行覆盖 `fusion.map_location`、`single_view.include_depth` 等字段进行实验扩展。【F:configs/tasks/aa_feature_fusion/default.yaml†L1-L23】

## 输出结构与后续处理

Demo 生成的 `.pt` 文件包含：

- 原始单视图 RGB、内参与相机位姿。
- 融合后经 MapAnything 头部推理出的点云、深度或其他场景表示，字段名称与 `scene_rep_type` 相匹配。
- 用于追踪样本来源的 `dataset`、`label` 与 `instance` 元数据。

这些字段对应 Demo 中写入磁盘的 `record` 字典，可直接在自定义脚本中读取并接入新的任务管线。【F:mapanything/tasks/aa_feature_fusion/demo.py†L172-L205】

通过以上文档与脚本，你可以在不改动核心模型的前提下，完成 7Scenes 上的 AA 特征捕获、单视图融合重建，以及后续训练评估的搭建，为扩展任务奠定基础。
