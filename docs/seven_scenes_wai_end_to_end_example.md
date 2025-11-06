# 7Scenes 数据集从下载到 MapAnything 使用的完整示例

本文给出一套可以直接照搬的 7Scenes 工作流，从原始数据下载、PGT 重建到 WAI 转换与 MapAnything 使用。示例命令均假设使用 Bash，并以 `/mnt/storage/xwh/7Scenes` 作为数据根目录，可根据实际情况修改。

## 0. 前置条件

- 已安装 `conda`，并创建好包含 `wai_processing` 依赖的环境，例如 `conda create -n wai_processing python=3.10` 并按仓库要求安装依赖。
- 安装好 MapAnything 的 Python 依赖，能够运行 `python -m compileall mapanything` 之类的检查。
- 确保磁盘空间充足（原始压缩包约 40 GB，解压及中间产物约 120 GB）。

为了简化后续命令，可以在 shell 中预先设置：

```bash
export DATA_ROOT=/mnt/storage/xwh/7Scenes
export WAI_ROOT=/mnt/storage/xwh/mapanything-dataset/wai_data/7scenes
export CONDA_ENV=wai_processing
mkdir -p "$DATA_ROOT" "$WAI_ROOT"
```

## 1. 下载原始 RGB-D 数据与伪真值仓库

使用仓库提供的脚本一次性下载所有场景并完成两层解压，同时克隆伪真值仓库：

```bash
python data_processing/wai_processing/download_scripts/download_7scenes.py \
    --target_dir "$DATA_ROOT/downloads" \
    --extract_dir "$DATA_ROOT/7scenes_source" \
    --pgt_dir "$DATA_ROOT/visloc_pseudo_gt_limitations" \
    --stages download extract pgt
```

- `downloads/` 会保存官方 ZIP 包。
- `7scenes_source/` 会展开为 `scene/seq-XX/frame-XXXXXX.*` 的原始结构。
- `visloc_pseudo_gt_limitations/` 是后续生成 PGT 位姿所需的仓库（可用 `--update_pgt` 更新）。

## 2. 生成 `pgt_7scenes_*` 结构

MapAnything 的转换脚本期待每个场景都已经整理为 `pgt_7scenes_<scene>/<split>/<modality>/` 形式。如果已经有该结构可以跳过本节。

官方 `visloc_pseudo_gt_limitations` 仓库提供了 `setup_7scenes.py` 工具，可以在刚才的 `DATA_ROOT` 中运行：

```bash
cd "$DATA_ROOT"
python path/to/setup_7scenes.py \
    --poses pgt \
    --depth calibrated \
    --eye calibrated
cd -
```

运行结束后，目录中会出现 `pgt_7scenes_chess/`、`pgt_7scenes_heads/`、`pgt_7scenes_stairs/` 等子目录，分别包含 `train/` 和 `test/` 下的 `rgb/`、`depth/`、`poses/`、`calibration/`（以及可选的 `eye/`）。

> 如果你已经通过自己的脚本完成了同样的重建，只需确保目录命名和文件后缀与上述一致即可。

## 3. 转换为 WAI 数据格式

仓库提供了一键脚本，会将 `pgt_7scenes_*` 转换为 WAI 场景，并生成共视图信息与 MoGe 元数据：

```bash
bash bash_scripts/data_processing/seven_scenes_to_wai.sh \
    "$DATA_ROOT" \
    "$WAI_ROOT" \
    "$CONDA_ENV" \
    original_depth_unit=millimeter \
    default_focal_length=585.0
```

脚本内部依次调用：

1. `python -m wai_processing.scripts.conversion.seven_scenes` —— 生成符号链接的 RGB、EXR 深度、`scene_meta.json`。
2. `python -m wai_processing.scripts.covisibility` —— 基于真值深度计算共视图指标。
3. `python -m wai_processing.scripts.run_moge` —— 生成 MapAnything 下游任务所需的统计量。

转换完成后，`$WAI_ROOT` 下会按场景与序列创建目录，例如 `chess_train_seq-01/`，内部包含 `images/`、`depth/`、`scene_meta.json` 等文件。

## 4. 验证转换结果

可以直接调用数据集类的 CLI 检查统计信息，确认路径与元数据可用：

```bash
python -m mapanything.datasets.wai.seven_scenes \
    --root_dir "$WAI_ROOT" \
    --dataset_metadata_dir /path/to/mapanything_dataset_metadata \
    --split test \
    --num_of_views 2
```

输出会展示样本数量、视角组合情况等。如果加上 `--viz --save /tmp/rr`（需安装 `rerun`），还能快速可视化。

## 5. 在 MapAnything 中使用数据集

### 5.1 运行多视角基准脚本

仓库已经准备好 `configs/dataset/benchmark_518_seven_scenes.yaml`，可以在 `benchmarking/dense_n_view` 基准中直接引用：

```bash
python benchmarking/dense_n_view/benchmark.py \
    dataset=benchmark_518_seven_scenes \
    model=mapanything \
    model.mapanything.checkpoint_path=/path/to/mapanything.ckpt \
    root_data_dir="$WAI_ROOT" \
    machine.mapanything_dataset_metadata_dir=/path/to/mapanything_dataset_metadata \
    dataset.num_views=2
```

- `root_data_dir` 会通过 Hydra 传给 `SevenScenesWAI` 加载器。
- `machine.mapanything_dataset_metadata_dir` 指向预先下载好的公共元数据包。
- `dataset.num_views` 可以按需改成 1～5，覆盖不同的视角组合数量。

### 5.2 启动训练/微调

若要把 7Scenes 加入训练，只需在调用 `scripts/train.py` 时覆盖训练集定义即可。例如以 512 分辨率、单数据集训练：

```bash
python scripts/train.py \
    dataset.train_dataset="[${dataset.seven_scenes_wai.train.dataset_str}]" \
    dataset.test_dataset="[${dataset.seven_scenes_wai.test.dataset_str}]" \
    dataset.resolution_train="${dataset.resolution_options.512_4_3_ar}" \
    dataset.resolution_val="${dataset.resolution_options.512_4_3_ar}" \
    root_data_dir="$WAI_ROOT" \
    machine.mapanything_dataset_metadata_dir=/path/to/mapanything_dataset_metadata \
    model=mapanything \
    optimizer=adamw \
    training.max_steps=20000
```

也可以把 7Scenes 与其他数据集组合，只需把 `dataset.train_dataset` 中的列表换成多个条目即可。

## 6. 目录检查速查表

运行完整流程后，关键目录应类似：

```
$DATA_ROOT/
├── downloads/                       # 原始 ZIP
├── 7scenes_source/                  # 官方解压结果
├── visloc_pseudo_gt_limitations/    # 伪真值仓库
├── pgt_7scenes_chess/
├── pgt_7scenes_heads/
├── pgt_7scenes_stairs/
└── ...

$WAI_ROOT/
├── chess_train_seq-01/
│   ├── images/seq-01-frame-000000.color.png -> ../../../../7scenes_source/... (符号链接)
│   ├── depth/seq-01-frame-000000.exr
│   └── scene_meta.json
├── chess_test_seq-03/
├── heads_train_seq-01/
└── ...
```

按照以上步骤即可在 MapAnything 中顺利使用 7Scenes 数据集，同时保留原始数据、伪真值和转换后的 WAI 数据。
