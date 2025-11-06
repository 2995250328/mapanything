# MapAnything alternating-attention feature storage

This document explains how the MapAnything model captures, stores, and reloads the alternating-attention (AA) transformer features that are produced during inference. It summarises the configuration knobs you enable through Hydra and walks through the runtime helpers that package the tensors in-memory or on disk.

## Enabling capture through configuration

Hydra presets such as [`configs/model/mapanything_store_intermediates.yaml`](../configs/model/mapanything_store_intermediates.yaml) and [`configs/model/mapanything_store_intermediates_ace.yaml`](../configs/model/mapanything_store_intermediates_ace.yaml) set the `model_config.store_info_sharing_intermediate_features` flag to `true` and provide a storage location via `model_config.info_sharing_storage_path`.【F:configs/model/mapanything_store_intermediates.yaml†L12-L22】【F:configs/model/mapanything_store_intermediates_ace.yaml†L12-L22】 When a run directory is supplied, the model writes a single `info_sharing_outputs.pt` payload per inference into the `${hydra:run.dir}` tree (for example, `${hydra:run.dir}/aa_features` or `${hydra:run.dir}/ACE/aa_blocks`).

If you omit the storage path, the same flag keeps the features in memory on an optional target device controlled by `model_config.info_sharing_storage_device` before returning them to the caller.【F:mapanything/models/mapanything/model.py†L140-L172】 This lets you prototype without touching disk.

## Forward pass capture flow

During the standard `forward` method, the model optionally receives the intermediate transformer outputs alongside the final alternating-attention result when `info_sharing_return_type` requests them.【F:mapanything/models/mapanything/model.py†L1568-L1587】 Once the transformer call returns, the flag described above decides whether `_capture_info_sharing_features` is invoked to record the tensors.【F:mapanything/models/mapanything/model.py†L1589-L1594】 Disabling the flag clears any previously cached outputs.【F:mapanything/models/mapanything/model.py†L1595-L1596】

`_capture_info_sharing_features` creates the metadata that you later retrieve by calling `get_info_sharing_intermediate_features`. Internally it branches on whether a run directory was provisioned:

* **Memory mode:** When no `info_sharing_storage_path` is configured, each `MultiViewTransformerOutput` is cloned (and optionally moved to `info_sharing_storage_device`) through `_serialize_multi_view_output`, preserving both view-wise features and additional token embeddings.【F:mapanything/models/mapanything/model.py†L1976-L2010】 The returned dictionary includes the `storage_mode="memory"` marker together with the cloned outputs.【F:mapanything/models/mapanything/model.py†L2058-L2079】
* **Disk mode:** When a storage path is present, `_create_info_sharing_run_directory` materialises a timestamped subdirectory under the configured root.【F:mapanything/models/mapanything/model.py†L2032-L2056】 `_prepare_output_for_disk` then detaches each tensor to CPU, tags every block, and assembles a serialisable dictionary before `_write_info_sharing_payload` saves everything to `info_sharing_outputs.pt`.【F:mapanything/models/mapanything/model.py†L2012-L2031】【F:mapanything/models/mapanything/model.py†L2079-L2099】 Alongside the on-disk file, the method returns lightweight metadata (shapes, block indices, and tags) so callers can inspect what was captured without reloading the tensors.【F:mapanything/models/mapanything/model.py†L2024-L2031】【F:mapanything/models/mapanything/model.py†L2099-L2110】

In both modes the stored payload records the transformer's `return_type`, `info_sharing_type`, the final AA block, and every intermediate AA block in order, ensuring downstream consumers can reconstruct the complete attention stack.【F:mapanything/models/mapanything/model.py†L2079-L2094】

## Retrieving the cached tensors

`get_info_sharing_intermediate_features` simply hands back the cached dictionary and optionally clears it so the next forward call does not reuse stale tensors.【F:mapanything/models/mapanything/model.py†L2118-L2129】 The schema mirrors what `_capture_info_sharing_features` produced, meaning callers see either the in-memory `MultiViewTransformerOutput` objects or the disk metadata described above.

When you later want to inspect a saved run, `load_info_sharing_features_from_file` accepts the `info_sharing_outputs.pt` path, runs `torch.load`, and recreates the final and intermediate transformer blocks as fresh `MultiViewTransformerOutput` instances (unless you request the raw dictionary).【F:mapanything/models/mapanything/model.py†L2134-L2176】 This helper hides the on-disk layout and centralises all deserialisation logic.

## Summary of the capture lifecycle

1. Enable `store_info_sharing_intermediate_features` and optionally choose a disk path through the Hydra configuration.
2. Run inference. The `forward` method records the alternating-attention outputs by invoking `_capture_info_sharing_features`.
3. Inspect the metadata returned by `get_info_sharing_intermediate_features` (memory mode) or reference the saved `info_sharing_outputs.pt` file (disk mode).
4. Rehydrate the tensors later with `load_info_sharing_features_from_file` when you need to analyse or visualise the alternating-attention blocks offline.

These utilities let you treat intermediate transformer activations as first-class artefacts without modifying the rest of the inference pipeline.
