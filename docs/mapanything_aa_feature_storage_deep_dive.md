# MapAnything alternating-attention capture walkthrough

This deep-dive accompanies the high-level overview in
[`docs/mapanything_aa_feature_storage.md`](./mapanything_aa_feature_storage.md) by
explaining the exact code paths that persist alternating-attention (AA)
intermediates inside `mapanything/models/mapanything/model.py`. Every section
below highlights the responsible helper, the relevant arguments, and any
non-obvious implementation details so you can adapt or extend the
functionality with confidence.

## Entry point: `forward`

The capture workflow begins inside
[`MapAnythingModel.forward`](../mapanything/models/mapanything/model.py). After
computing the dense encoder features and invoking the multi-view transformer,
the method decides whether to stash the AA outputs by checking the
`store_info_sharing_intermediate_features` flag. When the flag is enabled, the
final transformer result (`MultiViewTransformerOutput`) and the optional list of
intermediate blocks are passed to `_capture_info_sharing_features` and the
returned payload is cached on `self._stored_info_sharing_features`. If the flag
is disabled the cache is cleared so stale tensors are never exposed to callers.

Key lines:

* `final_info_sharing_multi_view_feat` and
  `intermediate_info_sharing_multi_view_feat` are populated based on
  `info_sharing_return_type`.
* `_capture_info_sharing_features` receives both outputs and converts them to a
  serialisable structure when storage is requested.
* The result is retained until `get_info_sharing_intermediate_features` is
  called or the next forward pass overwrites it.

## Selecting the storage backend: `_capture_info_sharing_features`

`_capture_info_sharing_features(final_output, intermediate_outputs)` is the
central branching helper. Its job is to determine whether the tensors should be
kept in memory or written to disk. The decision hinges on
`self.info_sharing_storage_path`:

* **Memory mode** (no storage path): `_serialize_multi_view_output` clones each
  `MultiViewTransformerOutput` to avoid side effects, optionally moves the clone
  to `info_sharing_storage_device`, and assembles a dictionary containing the
  transformer metadata plus the cloned outputs. The method tags the dictionary
  with `storage_mode="memory"` so downstream code can reason about the payload
  without inspecting the tensors.
* **Disk mode** (storage path provided): `_create_info_sharing_run_directory`
  builds a fresh run directory under `info_sharing_storage_path`, then
  `_prepare_output_for_disk` converts every AA block into a plain dictionary with
  CPU tensors and descriptive tags. The helper aggregates the final and
  intermediate blocks into a `payload` dictionary, forwards it to
  `_write_info_sharing_payload`, and finally returns lightweight metadata about
  what was saved (paths, shapes, block indices) alongside
  `storage_mode="disk"`.

Regardless of the storage mode, the method captures the transformer
`return_type` and `info_sharing_type` so consumers can reconstruct the exact
inference configuration later on.

## Converting AA blocks: `_serialize_multi_view_output`

This helper is only used in memory mode. It receives a
`MultiViewTransformerOutput`, clones the `features` tensor as well as the
optional `additional_token_features`, and records the tensor shapes. If
`self.info_sharing_storage_device` is set, the clones are moved to that device
(e.g. CPU for long-lived storage). The returned dictionary mirrors the input
structure:

```python
{
    "features": cloned_features,
    "additional_token_features": cloned_additional_tokens,
    "shape": tuple(cloned_features.shape),
}
```

Cloning is crucial: it ensures that subsequent layers cannot mutate the stored
activations and that the underlying storage device can be changed without
impacting the original forward pass tensors.

## Preparing disk payloads

When a storage directory is configured, three helpers participate in writing the
payload to disk:

1. **`_create_info_sharing_run_directory`** resolves the base path, generates a
   timestamped run directory (e.g. `2024-05-05T12-34-56`), creates it on disk,
   and returns the `Path` object. It also removes any stale directory when the
   flag is disabled to avoid mixing runs.
2. **`_prepare_output_for_disk`** receives either the final transformer output or
   a single intermediate block plus a tag (`"final"` or `"intermediate"`) and an
   optional index. It moves tensors to CPU, detaches them, and packages them in a
   dictionary with fields such as `block_type`, `block_index`, `features`, and
   `additional_token_features`.
3. **`_write_info_sharing_payload`** materialises the `payload` file. It uses
   `torch.save` to serialize the dictionary to `info_sharing_outputs.pt` inside
   the run directory. The function returns the final `Path` to aid logging and
   later inspection.

Combining these helpers ensures that every AA block (final and intermediate) is
preserved in a single file along with rich metadata.

## Accessing cached results: `get_info_sharing_intermediate_features`

After `_capture_info_sharing_features` executes, the forward pass can expose the
stored payload through `get_info_sharing_intermediate_features(clear=False)`. In
memory mode the method returns the cloned `MultiViewTransformerOutput`
instances, while disk mode returns the metadata describing the on-disk file. The
optional `clear` flag erases the cached result, which is helpful when repeatedly
calling `forward` in the same Python session.

## Reloading from disk: `load_info_sharing_features_from_file`

To rehydrate a saved run, call
`load_info_sharing_features_from_file(path, map_location=None, as_outputs=True)`.
The helper expands the provided `path`, loads the serialized dictionary via
`torch.load`, and—when `as_outputs` is `True`—wraps each block back into a new
`MultiViewTransformerOutput`. Returning raw dictionaries is also supported by
setting `as_outputs=False`, which can be useful for custom tooling or
non-PyTorch inspection pipelines.

## Putting it together

1. Enable capture through the Hydra configuration by setting
   `store_info_sharing_intermediate_features=true` and optionally supplying
   `info_sharing_storage_path`.
2. Run inference. The `forward` method saves the transformer outputs using the
   helper chain described above.
3. Retrieve the results immediately with
   `get_info_sharing_intermediate_features` or later by loading the
   `info_sharing_outputs.pt` artifact with
   `load_info_sharing_features_from_file`.

By following these steps you can treat alternating-attention intermediates as a
first-class artifact, adapt the storage pipeline, or integrate the saved tensors
into downstream ACE tasks without revisiting the implementation.
