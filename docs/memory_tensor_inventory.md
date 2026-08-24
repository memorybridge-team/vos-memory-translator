# SAM 2.1 temporal-memory tensor inventory

## Reproducibility boundary

This inventory was verified against the official `facebookresearch/sam2`
revision:

```text
2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Relevant upstream files are `sam2/modeling/sam2_base.py` and
`sam2/sam2_video_predictor.py`. The predictor state layout and several methods
below are private APIs. A probe run must record this commit and stop on a commit
mismatch unless the operator explicitly accepts the version risk.

The current-frame image-encoder cache at `inference_state["cached_features"]`
is excluded. It is an RGB feature cache used to avoid recomputing a visited
frame, not accumulated temporal memory.

## End-to-end flow

For object index `i` and frame `t`, the compact output is stored at one of:

```text
inference_state["output_dict_per_obj"][i]
  ["cond_frame_outputs" or "non_cond_frame_outputs"][t]
```

On frame `t`, `SAM2Base._track_step` first calls
`_prepare_memory_conditioned_features`. That function selects conditioning
outputs and up to `num_maskmem - 1` recent non-conditioning outputs. It reads
their spatial memory and object pointers, concatenates them, and calls
`self.memory_attention(memory=..., memory_pos=...)`. After mask decoding,
`_encode_memory_in_output` calls `_encode_new_memory` to create the spatial
memory saved for later frames. `SAM2VideoPredictor._run_single_frame_inference`
then reduces the frame output to the five keys listed below and places it in the
per-object conditioning or non-conditioning dictionary.

The state behaves like a bounded indexed store: frame outputs are retained in
the dictionaries, while each next-frame query selects only the configured
conditioning frames, recent non-conditioning frames, and eligible object
pointers.

## Tensor inventory

Expected shapes below are configuration-derived for one object at the standard
1024-pixel input size. They must be confirmed from manifests for each actual
checkpoint and runtime.

| Tensor | Producer | State location | Read on a later frame | Expected stored shape | Role and frame variation |
|---|---|---|---|---|---|
| `maskmem_features` | `SAM2Base._encode_new_memory`: `memory_encoder(...)["vision_features"]`; predictor converts it to `bfloat16` and optional storage device | per-object `cond_frame_outputs[t]` or `non_cond_frame_outputs[t]` | `_prepare_memory_conditioned_features`: `prev["maskmem_features"]`; flattened into `memory` for `memory_attention` | `[1, 64, 64, 64]` | Primary spatial temporal memory. Expected to change with frame appearance and predicted mask. |
| `maskmem_pos_enc` | `SAM2Base._encode_new_memory`: `memory_encoder(...)["vision_pos_enc"]` | per-frame output contains an expanded list view; canonical single-object tensors are cached once at `inference_state["constants"]["maskmem_pos_enc"]` | `_prepare_memory_conditioned_features`: `prev["maskmem_pos_enc"][-1]`; temporal position embedding is added before `memory_attention(memory_pos=...)` | list whose used element is `[1, 64, 64, 64]` | Spatial positional encoding. Upstream explicitly treats it as constant across frames and objects; temporal offsets are added only while assembling the consumer input. |
| `obj_ptr` | `_forward_sam_heads`: SAM output token projected by `obj_ptr_proj`; optional no-object adjustment | same per-frame output dictionary | `_prepare_memory_conditioned_features`: selected outputs' `out["obj_ptr"]`; stacked and possibly split into 64-channel tokens appended to `memory` | `[1, 256]` before token splitting | Compact object identity/appearance path used by later memory attention. Expected to change by frame. |
| `pred_masks` | `_forward_sam_heads` / `_use_mask_as_output`, saved by `track_step`; predictor offloads the low-resolution logits | same per-frame output dictionary | Not read by `_prepare_memory_conditioned_features`. It is used for returned masks and as previous logits during later user correction on the same frame. The high-resolution current-frame mask, not the compact stored copy, feeds `_encode_new_memory`. | `[1, 1, 256, 256]` | Supporting prediction state, not a direct next-frame memory-attention input. Expected to change by frame. |
| `object_score_logits` | SAM mask decoder object-score head, returned by `_forward_sam_heads` | same per-frame output dictionary; kept on compute device | Not read from prior frame outputs by `_prepare_memory_conditioned_features`. On its production frame it gates the no-object embedding inside `_encode_new_memory`. | `[1, 1]` | Supporting occlusion/gating state. Expected to change by frame, but it is not independently consumed at `t+1`. |
| assembled `memory` | `_prepare_memory_conditioned_features`: concatenated flattened `maskmem_features` plus optional split `obj_ptr` tokens | transient; not stored in `inference_state` | `MemoryAttention.forward(memory=...)` | `[sequence, 1, 64]`; sequence depends on selected frames and pointer tokens | Actual cross-attention key/value input at the next frame. Probe-only observation; not an injection state key. |
| assembled `memory_pos` | `_prepare_memory_conditioned_features`: spatial memory PE + learned temporal PE, plus object-pointer temporal PE | transient; not stored | `MemoryAttention.forward(memory_pos=...)` | same as assembled `memory` | Actual positional input paired with `memory`. Changes with selected source frames and temporal distances. |

## Conditioning and non-conditioning storage

- A newly prompted, previously untracked frame is a conditioning frame.
- Propagated frames are stored in `non_cond_frame_outputs`.
- During `propagate_in_video_preflight`, temporary prompted outputs receive their
  memory encoding and move from `temp_output_dict_per_obj` into
  `output_dict_per_obj`.
- The next frame always considers selected conditioning outputs. It additionally
  considers recent non-conditioning outputs according to `num_maskmem` and
  `memory_temporal_stride_for_eval`.
- SAM 2.1 Tiny and Large both configure `num_maskmem=7`, memory encoder
  `out_dim=64`, memory attention `d_model=256`, and image size 1024.

## Tiny/Large compatibility and translator candidate points

Static configuration inspection predicts equal shapes for all five compact
state keys between SAM 2.1 Tiny and Large. This permits a shape-level direct-copy
test but does not establish representation alignment.

| Candidate | Shape-level action | Current decision |
|---|---|---|
| `maskmem_pos_enc` | Direct reuse is structurally possible | Prefer target-native regeneration/cache in later injection work because it is model-owned positional state; verify equality first. |
| `pred_masks` | Direct copy is structurally possible | Keep as supporting mask state, not a learned temporal-memory translation target. |
| `object_score_logits` | Direct copy is structurally possible | Keep as diagnostic/gating state; it is not a prior-frame memory-attention input. |
| `maskmem_features` | Equal channel and spatial shape permits direct-copy baseline | Primary feature translator candidate if measured values and future rollout show representation mismatch. A 1x1 channel map is only a hypothesis; equal channels do not prove a linear relation. |
| `obj_ptr` | Equal 256-dimensional shape permits direct-copy baseline | Pointer translator candidate if direct copy is not behaviorally compatible. Test separately from spatial memory. |
| assembled `memory` / `memory_pos` | Do not persist or translate as the primary interface | These depend on frame selection and temporal offsets. Inject compact state and let the target assemble its own attention input. |

Translator training should begin only after sequential Tiny and Large runs on the
same video, prompt, prefix, and switch frame produce manifests that confirm
shape, dtype, device placement, per-frame variation, and consumer-input capture.

## Private API and version risks

- `inference_state["output_dict_per_obj"]`, `temp_output_dict_per_obj`, and
  `constants` are internal implementation details.
- `_prepare_memory_conditioned_features`, `_run_single_frame_inference`, and
  `_get_maskmem_pos_enc` are private methods.
- The current predictor processes per-object slices. Older SAM 2 predictor code
  used a consolidated `output_dict`; adapters written for that layout are not
  interchangeable.
- A forward hook can observe the assembled memory-attention inputs, but assigning
  a frame/object context requires a small private-method wrapper. The probe must
  fail clearly if the expected methods or keyword arguments disappear.
- Checkpoint architecture equality must be checked from runtime manifests, not
  inferred solely from YAML files.

