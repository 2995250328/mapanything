# MapAnything 交替注意力捕获全流程解析

本深度解析与概览文档 [`docs/mapanything_aa_feature_storage.md`](./mapanything_aa_feature_storage.md) 配套，逐段说明 `mapanything/models/mapanything/model.py` 中负责持久化交替注意力（AA）中间结果的代码路径。每一节都会指出涉及的辅助函数、入参以及关键实现细节，方便你在理解后进行扩展或定制。

## 入口：`forward`

工作流始于 [`MapAnythingModel.forward`](../mapanything/models/mapanything/model.py)。在完成稠密编码特征并调用多视角 Transformer 之后，函数会检查 `store_info_sharing_intermediate_features` 是否开启：

* 如果启用，该方法会把最终的 `MultiViewTransformerOutput` 以及可选的中间块列表传递给 `_capture_info_sharing_features`，并把返回的 payload 缓存在 `self._stored_info_sharing_features`。
* 如果未启用，则立即清空缓存，确保不会向调用方暴露过期张量。

关键代码要点：

* `final_info_sharing_multi_view_feat` 与 `intermediate_info_sharing_multi_view_feat` 会根据 `info_sharing_return_type` 是否请求中间输出来填充。
* `_capture_info_sharing_features` 接收两类输出，并在需要存储时把它们转换成可序列化的结构。
* 该结果会一直保留，直到调用 `get_info_sharing_intermediate_features` 或下一次 forward 覆盖它。

## 选择存储后端：`_capture_info_sharing_features`

`_capture_info_sharing_features(final_output, intermediate_outputs)` 是整个链路的核心分流函数，负责决定张量应保存在内存还是写入磁盘。判断依据是 `self.info_sharing_storage_path`：

* **内存模式（未配置存储路径）**：`_serialize_multi_view_output` 会克隆每个 `MultiViewTransformerOutput`，以避免后续修改原始张量。若设置了 `info_sharing_storage_device`，克隆体会被迁移到指定设备（常见于将数据移至 CPU 便于长时间保留）。函数随后构造包含 Transformer 元数据与克隆张量的字典，并标记 `storage_mode="memory"`，方便下游辨别。
* **磁盘模式（已配置存储路径）**：`_create_info_sharing_run_directory` 会在 `info_sharing_storage_path` 下创建新的时间戳子目录；然后 `_prepare_output_for_disk` 会把每个 AA 块转为普通字典：张量被 `detach()` 后移动到 CPU，并附带 `block_type`、`block_index` 等描述信息。最终 `_write_info_sharing_payload` 使用 `torch.save` 把所有数据写入 `info_sharing_outputs.pt`，并返回存储路径以及 `storage_mode="disk"` 的元数据。

无论哪种模式，函数都会附带 Transformer 的 `return_type` 与 `info_sharing_type`，使得后续在加载时可以重建当时的推理设置。

## 转换 AA 块：`_serialize_multi_view_output`

此函数仅在内存模式下使用。它会：

1. 克隆 `MultiViewTransformerOutput` 中的 `features` 张量以及可选的 `additional_token_features`；
2. 将克隆体移动到 `info_sharing_storage_device` 指定的设备；
3. 返回一个结构化字典：

```python
{
    "features": cloned_features,
    "additional_token_features": cloned_additional_tokens,
    "shape": tuple(cloned_features.shape),
}
```

克隆操作保证了后续网络层不会意外修改缓存激活，同时也允许在不影响原始 forward 张量的情况下迁移存储设备。

## 写盘三部曲

当存在存储目录时，会依次调用三个辅助函数：

1. **`_create_info_sharing_run_directory`**：解析根目录、生成形如 `2024-05-05T12-34-56` 的时间戳子目录并创建；当关闭存储功能时还会清理旧目录，避免不同运行的结果混淆。
2. **`_prepare_output_for_disk`**：接收最终或单个中间 Transformer 输出，以及标记其类型的 `tag`（如 `"final"`、`"intermediate"`）与可选索引。函数会把张量移至 CPU 并 `detach()`，然后打包成包含 `features`、`additional_token_features`、`block_index`、`block_type` 等字段的字典。
3. **`_write_info_sharing_payload`**：负责真正落盘，调用 `torch.save` 将聚合后的 `payload` 序列化到运行目录中的 `info_sharing_outputs.pt`。

通过这一组合，所有 AA 块（最终与各个中间层）都被保存在单个文件中，并带有详尽的上下文信息。

## 访问缓存结果：`get_info_sharing_intermediate_features`

在 `_capture_info_sharing_features` 执行完毕后，可调用 `get_info_sharing_intermediate_features(clear=False)` 获取存储结果：

* 内存模式会返回克隆后的 `MultiViewTransformerOutput`；
* 磁盘模式则返回包含文件路径、块信息、张量形状等的元数据。

将 `clear` 设为 `True` 可以在读取后立刻清空缓存，适合在同一 Python 会话中多次调用 `forward`。

## 从磁盘重新加载：`load_info_sharing_features_from_file`

若需重新加载历史运行结果，可调用 `load_info_sharing_features_from_file(path, map_location=None, as_outputs=True)`：

1. 函数会解析 `path` 并通过 `torch.load` 读取 `info_sharing_outputs.pt`；
2. 当 `as_outputs=True` 时，会把每个块重新封装为新的 `MultiViewTransformerOutput`；
3. 若设为 `False`，则直接返回原始字典，便于在自定义工具或非 PyTorch 环境中使用。

## 全流程回顾

1. 在 Hydra 配置中开启 `store_info_sharing_intermediate_features=true`，并按需设置 `info_sharing_storage_path`。
2. 执行推理，`forward` 会调用一系列辅助函数完成特征捕获。
3. 可立即使用 `get_info_sharing_intermediate_features` 查看结果，或稍后通过 `load_info_sharing_features_from_file` 加载 `info_sharing_outputs.pt`。

理解这套流程后，你可以将 AA 中间特征当作稳定的研究素材，也可以为 ACE 等下游任务构建自己的后处理与可视化工具链。
