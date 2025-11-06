# MapAnything 交替注意力特征存储指南

本文详细说明 MapAnything 模型在推理阶段如何捕获、存储以及重新加载交替注意力（Alternating-Attention，AA）Transformer 产生的特征。你将看到需要开启的 Hydra 配置、运行期负责打包张量的辅助函数，以及它们在内存与磁盘之间切换时的具体行为。

## 通过配置开启特征捕获

Hydra 预设（例如 [`configs/model/mapanything_store_intermediates.yaml`](../configs/model/mapanything_store_intermediates.yaml) 与 [`configs/model/mapanything_store_intermediates_ace.yaml`](../configs/model/mapanything_store_intermediates_ace.yaml)）会把 `model_config.store_info_sharing_intermediate_features` 设为 `true`，同时通过 `model_config.info_sharing_storage_path` 指定存储位置。【F:configs/model/mapanything_store_intermediates.yaml†L12-L22】【F:configs/model/mapanything_store_intermediates_ace.yaml†L12-L22】如果配置了运行目录，模型会在 `${hydra:run.dir}` 下写出单个 `info_sharing_outputs.pt` 文件（例如 `${hydra:run.dir}/aa_features` 或 `${hydra:run.dir}/ACE/aa_blocks`）。

若未提供存储路径，同一开关会把特征保存在内存中，并根据 `model_config.info_sharing_storage_device` 决定目标设备，然后再把特征返回给调用方。【F:mapanything/models/mapanything/model.py†L140-L172】这让你可以在不落盘的情况下完成调试。

## Forward 流程中的捕获逻辑

在标准的 `forward` 方法里，只要 `info_sharing_return_type` 请求返回中间 AA 结果，模型就会在最终输出之外额外拿到 Transformer 的中间块。【F:mapanything/models/mapanything/model.py†L1568-L1587】Transformer 调用结束后，前面的开关会决定是否调用 `_capture_info_sharing_features` 来记录这些张量。【F:mapanything/models/mapanything/model.py†L1589-L1594】如果不开启，该方法会清理此前缓存的输出，避免复用过期数据。【F:mapanything/models/mapanything/model.py†L1595-L1596】

`_capture_info_sharing_features` 会构造后续通过 `get_info_sharing_intermediate_features` 取出的元数据，内部根据是否配置了磁盘路径分成两种模式：

* **内存模式：** 当 `info_sharing_storage_path` 为空时，`_serialize_multi_view_output` 会克隆每个 `MultiViewTransformerOutput`（必要时迁移到 `info_sharing_storage_device`），保留各视角特征以及额外 token 的嵌入。【F:mapanything/models/mapanything/model.py†L1976-L2010】返回的字典会带上 `storage_mode="memory"` 标记和克隆后的输出。【F:mapanything/models/mapanything/model.py†L2058-L2079】
* **磁盘模式：** 当存在存储路径时，`_create_info_sharing_run_directory` 会在配置目录下创建一个带时间戳的子目录。【F:mapanything/models/mapanything/model.py†L2032-L2056】随后 `_prepare_output_for_disk` 会把每个张量转到 CPU、标记每个 AA 块，再由 `_write_info_sharing_payload` 将整理好的字典写入 `info_sharing_outputs.pt`。【F:mapanything/models/mapanything/model.py†L2012-L2031】【F:mapanything/models/mapanything/model.py†L2079-L2099】与此同时，函数会返回轻量级元数据（形状、块索引、标签），便于调用方无需加载文件即可了解捕获内容。【F:mapanything/models/mapanything/model.py†L2024-L2031】【F:mapanything/models/mapanything/model.py†L2099-L2110】

无论哪种模式，存储的 payload 都会记录 Transformer 的 `return_type`、`info_sharing_type`、最终 AA 块以及所有中间 AA 块，确保下游使用者能完整还原注意力堆栈。【F:mapanything/models/mapanything/model.py†L2079-L2094】

## 取回缓存张量

`get_info_sharing_intermediate_features` 直接返回缓存字典，并可选择性地清空，避免下一次 forward 复用旧数据。【F:mapanything/models/mapanything/model.py†L2118-L2129】返回的结构与 `_capture_info_sharing_features` 生成的格式一致：若为内存模式，即得到 `MultiViewTransformerOutput` 对象；若为磁盘模式，则只含元数据。

需要离线分析时，可调用 `load_info_sharing_features_from_file` 读取 `info_sharing_outputs.pt`。该函数会执行 `torch.load`，重新构造最终与中间 Transformer 块的 `MultiViewTransformerOutput` 实例（除非显式请求返回原始字典）。【F:mapanything/models/mapanything/model.py†L2134-L2176】这样可以把反序列化逻辑集中在一个地方。

## 工作流回顾

1. 在 Hydra 配置中开启 `store_info_sharing_intermediate_features`，并按需设定 `info_sharing_storage_path`。
2. 运行推理；`forward` 会调用 `_capture_info_sharing_features` 保存交替注意力输出。
3. 通过 `get_info_sharing_intermediate_features` 立即查看内存模式的结果，或在磁盘模式下定位 `info_sharing_outputs.pt`。
4. 当需要离线分析或可视化时，使用 `load_info_sharing_features_from_file` 重新加载 AA 块。

借助这些工具，你可以在不修改其他推理流程的情况下，把 Transformer 的中间激活当作一等公民来管理。
