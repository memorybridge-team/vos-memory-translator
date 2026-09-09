# SAM 2.1 Small/Large memory schema

## Verification boundary

This document describes implementation facts verified against the official
`facebookresearch/sam2` commit
`2b90b9f5ceec907a1c18123530e92e794ad901a4`. The relevant files are
`sam2/modeling/sam2_base.py`, `sam2/sam2_video_predictor.py`, and the
`sam2/configs/sam2.1/sam2.1_hiera_{s,l}.yaml` model configurations.

All state paths and methods beginning with an underscore are private upstream
APIs. The adapter verifies the pinned revision and required keys before it
extracts or injects state. A newer upstream revision is not assumed compatible.

`inference_state["cached_features"]` is deliberately excluded: it caches the
image encoder output for an individually visited current frame and is not the
accumulated temporal memory bank.

## Confirmed flow

```text
frame t image + mask logits
  -> SAM2Base._encode_new_memory
  -> memory_encoder(...)["vision_features"/"vision_pos_enc"]
  -> SAM2VideoPredictor._run_single_frame_inference compacts the output
  -> inference_state["output_dict_per_obj"][object_idx]
       ["cond_frame_outputs" | "non_cond_frame_outputs"][t]

frame t+1
  -> SAM2Base._prepare_memory_conditioned_features
  -> select conditioning frames and up to num_maskmem-1 non-conditioning frames
  -> read maskmem_features, maskmem_pos_enc, and obj_ptr
  -> add target-owned temporal position embeddings
  -> MemoryAttention.forward(memory=..., memory_pos=...)
```

The dictionaries can contain a longer history, but the attention input is a
bounded selection. Both pinned SAM 2.1 Small and Large configurations use
`num_maskmem=7`, `memory_encoder.out_dim=64`, memory attention `d_model=256`,
and a 1024-pixel model input.

## State fields

Shapes are the runtime shapes expected for one object and the standard model
input. The probe manifest is the authority for an actual checkpoint run.

| Component | Producer | Stored at | Later consumer | Shape / axes | Position? | Changes per frame? | Injection policy |
|---|---|---|---|---|---|---|---|
| `maskmem_features` | `SAM2Base._encode_new_memory`, from `memory_encoder(...)["vision_features"]` | `output_dict_per_obj[i][cond/non_cond][t]["maskmem_features"]` | `_prepare_memory_conditioned_features`, flattened from `[B,C,H,W]` to `[HW,B,C]` | expected `[1,64,64,64]`; B=object batch, C=memory channel, H/W=space | No | Yes | Primary translator target; inject translated tensor on target storage device |
| `maskmem_pos_enc` | `_encode_new_memory`, from `memory_encoder(...)["vision_pos_enc"]` | a list in every compact output; canonical one-object copy cached in `inference_state["constants"]["maskmem_pos_enc"]` | `_prepare_memory_conditioned_features`; the last list element is flattened and target temporal PE is added | used element expected `[1,64,64,64]` | Spatial PE | No; upstream treats it as constant across frames/objects | Regenerate with target memory encoder position module; do not fit with memory features |
| `obj_ptr` | `_forward_sam_heads`, `obj_ptr_proj(sam_output_token)` plus optional no-object adjustment | same compact frame output | `_prepare_memory_conditioned_features`; selected pointers are stacked and, because `mem_dim < hidden_dim`, split into 64-channel tokens | expected `[1,256]`; B=object batch, C=pointer channel | No; target generates pointer temporal PE later | Yes | Separate translator candidate; inject on target compute device |
| `pred_masks` | `track_step`, produced by SAM mask heads or direct mask path | same compact frame output | not read by next-frame memory attention; used for output and same-frame correction | expected `[1,1,256,256]`; B, mask channel, H, W | No | Yes | Copy only as supporting state / mask-only baseline; not a feature-alignment target |
| `object_score_logits` | mask decoder object-score head | same compact frame output | current-frame no-object gating during memory creation; not independently read from the past by memory attention | expected `[1,1]`; B, score | No | Yes | Direct copy as supporting metadata; not a default translator target |
| frame index | predictor dictionary key | `...[t]` and `frames_tracked_per_obj[i][t]` | memory selection and temporal distance calculation | scalar integer | Temporal metadata | Yes | Preserve exactly |
| object ID/index | `SAM2VideoPredictor._obj_id_to_idx` | bidirectional mappings and per-object dictionaries | predictor routing | Python metadata | No | Stable per object | Rebuild through target predictor mapping; never regress |
| assembled `memory` | `_prepare_memory_conditioned_features`, concatenation of spatial memory and optional pointer tokens | transient only | `MemoryAttention.forward(memory=...)` | `[tokens,B,64]`; token count depends on selected frames/pointers | No | Yes | Do not serialize or inject; let target assemble it |
| assembled `memory_pos` | same function, spatial PE plus target temporal PE and pointer temporal PE | transient only | `MemoryAttention.forward(memory_pos=...)` | same leading axes as `memory` | Yes | Yes, due to temporal distance | Do not serialize or inject; let target assemble it |

The compact output has no explicit time or object dimension inside each tensor.
Those axes live in the enclosing dictionaries: `frame_idx` is the key, and
`object_idx` selects the outer per-object state.

## Small/Large comparison and translator points

Static configuration inspection predicts equal tensor shapes for Small and
Large. Equal shape permits a direct-copy control, but does not establish that
the two pretrained models use the same representation basis.

| Component | Direct copy | Learned map | Rationale |
|---|---:|---:|---|
| `maskmem_features` | required baseline | yes, default | primary spatial memory consumed at the next frame; use independent 1x1 affine maps for each direction |
| `obj_ptr` | required baseline | optional, separate | independent identity/appearance representation consumed by memory attention |
| `maskmem_pos_enc` | no by default | no | target-owned spatial encoding can be regenerated; target also adds its own temporal encoding |
| `pred_masks` | supporting baseline | no | output mask, not the memory-attention representation |
| `object_score_logits` | yes | no initially | small gating value, not read as historical memory |
| frame/object metadata | yes | no | structural identity must be preserved exactly |

Small-to-Large and Large-to-Small maps are independent parameters. No
invertibility assumption is made. The first baseline ladder is shape-only,
direct copy, affine, normalized affine, and closed-form ridge. Low-rank affine
is optional after the full affine path is validated.

## Replay-free injection contract

1. Source and target each receive the same video prefix only while native
   paired states are collected.
2. For a replay-free handoff evaluation, the target is initialized from a
   directory containing frames `t+1...end` only. Its image tensor is padded
   with placeholders for earlier global indices, and the warmed suffix feature
   is remapped from local index 0 to global index `t+1`.
3. The adapter injects compact source history at its original global frame
   indices, regenerates target spatial PE, and marks those frames tracked.
4. Propagation starts at `t+1`. A guard rejects any target image-feature request
   for `frame_idx <= t`.

This contract distinguishes “no target image-encoder replay” from merely
avoiding a second call to `propagate_in_video`: official `init_state` eagerly
loads frames and warms index 0, so initializing it on the full video would not
be a strict loader-level replay-free test.

## Private API risks

- `output_dict_per_obj`, `frames_tracked_per_obj`, `constants`, and the object
  ID maps are internal predictor state.
- `_obj_id_to_idx`, `_get_image_feature`, `_get_maskmem_pos_enc`,
  `_prepare_memory_conditioned_features`, and `_run_single_frame_inference` are
  private methods.
- The target PE regeneration relies on `memory_encoder.position_encoding`
  accepting a `[B,C,H,W]` tensor and depending on its spatial layout. Runtime
  shape checks remain mandatory.
- Multi-object output consolidation shares storage internally. The first smoke
  test is deliberately one object; multi-object validation remains a later
  milestone.
- The adapter must fail on absent keys or inconsistent shapes rather than
  silently synthesize undocumented SAM 2 state.

