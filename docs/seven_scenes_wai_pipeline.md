# 7Scenes 数据集转化为 WAI 格式流程

本文档总结了如何利用仓库中新添加的脚本，将微软 7Scenes 数据集及其伪真值（Pseudo Ground Truth, PGT）处理成 MapAnything 使用的 WAI 数据格式。整体流程沿袭 MapAnything 既有的数据处理体系，依次完成下载、生成 `pgt_7scenes_*` 目录结构以及 WAI 转换。

## 1. 下载原始数据与伪真值

使用 `download_7scenes.py` 可以一次性下载 7Scenes 官方发布的 RGB-D 序列，并可选地克隆 `visloc_pseudo_gt_limitations` 仓库（其中包含更精确的位姿和焦距信息）。

```bash
python data_processing/wai_processing/download_scripts/download_7scenes.py \
    --target_dir /mnt/storage/xwh/7Scenes/downloads \
    --extract_dir /mnt/storage/xwh/7Scenes/7scenes_source \
    --pgt_dir /mnt/storage/xwh/7Scenes/visloc_pseudo_gt_limitations \
    --stages download extract pgt
```

- `--target_dir`：保存原始 ZIP 压缩包的位置。
- `--extract_dir`：解压后保留原始目录结构的位置（每个场景包含多个 `seq-XX` 子目录）。
- `--pgt_dir`：可选参数，用于指定伪真值仓库的克隆路径。加上 `--update_pgt` 可更新已有仓库。
- `--scenes`：可选参数，若只想快速验证单个场景（例如 `--scenes chess`），可在下载与解压阶段仅处理少量数据。

脚本会自动两层解压所有 ZIP 文件，确保 `7scenes_source/<scene>/seq-XX` 结构完整，供后续生成 `pgt_7scenes_*` 结构时使用。

> **提示**：若已经使用现有的 `setup_7scenes.py` 或其它工具生成了 `pgt_7scenes_*` 目录，可直接跳到第 3 节。

## 2. 构建 `pgt_7scenes_*` 目录

MapAnything 的转换脚本期望看到如下的预处理结果：

```
/mnt/storage/xwh/7Scenes/
├── pgt_7scenes_chess/
│   ├── train/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── poses/
│   │   └── calibration/
│   └── test/
│       └── ...
├── pgt_7scenes_heads/
└── pgt_7scenes_stairs/
```

如果您已经拥有这一结构，可直接进入下一节。否则，可继续沿用已有的 `setup_7scenes.py`（或自己编写的同等脚本）来：

1. 读取 `visloc_pseudo_gt_limitations` 中的 PGT 位姿与焦距；
2. 将相应的 RGB、深度、位姿以及焦距信息整理到每个 `pgt_7scenes_<scene>/<split>` 目录下；
3. （可选）生成与 MapAnything 兼容的符号链接结构。

## 3. 快速运行一键脚本

若希望按照 MapAnything 标准流程一次性完成下载、转换和后处理，可直接运行：

```bash
bash bash_scripts/data_processing/seven_scenes_to_wai.sh \
    --device cuda \
    --moge-batch-size 8 \
    /mnt/storage/xwh/7Scenes \
    /mnt/storage/xwh/mapanything-dataset/wai_data/7scenes \
    wai_processing
```

若只想对某几个场景进行转换，可在命令末尾追加 `--datasets chess`（或使用逗号分隔的列表，如 `--datasets chess,heads`）。脚本还支持：

- `--device`：显式指定 PyTorch 设备（默认为 `cuda`，可改为 `cpu` 或 `cuda:1` 等）。转换阶段会在该设备上执行深度数据的清理、`covisibility` 计算以及 `MoGe` 推理。
- `--moge-batch-size`：覆盖 `MoGe` 的推理批大小，值越大越能填满 GPU，但也需要更多显存。

脚本会自动把场景过滤条件和设备配置传递给转换与后处理脚本，仅为指定场景生成 WAI 数据。

脚本依次执行：

1. 调用新的转换脚本 `wai_processing.scripts.conversion.seven_scenes`；
2. 运行 `covisibility` 与 `run_moge`，生成与其他数据集一致的附加元数据。

`<conda_env>` 参数用于指定安装了 `wai_processing` 依赖的 Conda 环境，脚本会通过 `conda run` 自动切换环境。

## 4. 转换脚本细节

`data_processing/wai_processing/scripts/conversion/seven_scenes.py` 假定 `original_root` 目录内部已经整理为 `pgt_7scenes_*` 布局。例如：

```
pgt_7scenes_chess/
└── train/
    ├── rgb/seq-01-frame-000000.color.png
    ├── depth/seq-01-frame-000000.depth.png
    ├── poses/seq-01-frame-000000.pose.txt
    └── calibration/seq-01-frame-000000.calibration.txt
```

脚本按以下步骤处理每一帧：

1. **图像链接**：在目标 WAI 目录下创建指向原图像的符号链接，保持零拷贝。
2. **深度重映射**：读取 16-bit 深度 PNG，将 0 和 65535 视为无效值并归零，剩余深度由毫米转换为米，保存为 EXR 文件（`store_data(..., "depth")`）。若指定 `device=cuda`，掩码和单位转换会在 GPU 上执行，随后再写回 CPU。
3. **位姿加载**：读取 `*.pose.txt` 中的 4x4 相机到世界变换矩阵，直接写入 `scene_meta.json`。
4. **相机内参**：从 `*.calibration.txt` 解析 `fx, fy, cx, cy`，若文件为空则回退到配置中定义的默认焦距和图像中心。
5. **WAI 元数据**：为每个帧生成包含 `image`、`depth`、`transform_matrix` 以及分辨率、焦距、主点等信息的条目。

最终，每个场景只会按照 split 聚合为两个子数据集（如 `train` 与 `test`）。
例如 `pgt_7scenes_chess/train/` 下所有序列会被合并到同一个目标目录 `chess_train/` 中：

```
chess_train/
├── images/seq-01-frame-000000.color.png -> 原始图像符号链接
├── images/seq-02-frame-000000.color.png -> 原始图像符号链接
├── depth/seq-01-frame-000000.exr
├── depth/seq-02-frame-000000.exr
└── scene_meta.json
```

`scene_meta.json` 会列出所有来自 `train` split 的帧信息，便于后续统一抽样。

## 5. 配置文件与批处理

- `configs/conversion/seven_scenes.yaml`：控制原始路径、默认焦距、深度无效值等参数，可通过 `device` 字段选择在 CPU 还是 GPU 上执行深度清理。
- 同一配置还新增了 `dataset_whitelist`、`split_whitelist` 与 `sequence_whitelist` 三个可选过滤项，可在调试阶段只转换少量场景、分割或序列；留空即处理全部数据。
- `configs/launch/seven_scenes.yaml`：提供与其他数据集一致的 SLURM 批处理配置，可在集群上批量运行。

## 6. 目录组织建议

结合用户提供的磁盘布局，可采用如下分层：

```
/mnt/storage/xwh/7Scenes/
├── downloads/                 # download_7scenes.py 保存的 zip 包
├── 7scenes_source/            # 解压后的原始 RGB-D 序列
├── visloc_pseudo_gt_limitations/  # 克隆的 PGT 仓库
├── pgt_7scenes_chess/         # 预处理后的目录（可由现有脚本生成）
├── pgt_7scenes_heads/
├── pgt_7scenes_stairs/
└── wai_data/7scenes/          # 转换得到的 WAI 数据集
```

这样既保持原始数据，也方便后续复用 WAI 数据或再次更新伪真值。

## 7. 额外提示

- 转换脚本默认将深度单位视作毫米，如果自定义数据源请确认单位一致。
- 若 `calibration` 目录缺失或为空，脚本会回退到配置中的默认焦距（585.0）以及 `(320, 240)` 主点。
- 运行 `covisibility` 和 `run_moge` 阶段需要 GPU/CPU 资源，建议参考仓库中其他数据集的资源配置；必要时可以通过 `--device cpu` 改为纯 CPU 推理。

至此，7Scenes 的下载、处理与 WAI 格式转化流程即告完成。
