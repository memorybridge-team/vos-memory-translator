# SAM 2 memory compatibility report

- Source model(s): `sam2.1-hiera-tiny`
- Target model(s): `sam2.1-hiera-large`
- Matched rows: 47
- All matched shapes equal: True

| Tensor | Frame | Object | Source shape | Target shape | Dtype equal | Disposition |
|---|---:|---|---|---|---|---|
| maskmem_features | 0 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 0 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| obj_ptr | 0 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 0 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 0 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 1 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 1 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 1 | 1 | `[4100, 1, 64]` | `[4100, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 1 | 1 | `[4100, 1, 64]` | `[4100, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 1 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 1 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 1 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 2 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 2 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 2 | 1 | `[8200, 1, 64]` | `[8200, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 2 | 1 | `[8200, 1, 64]` | `[8200, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 2 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 2 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 2 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 3 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 3 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 3 | 1 | `[12300, 1, 64]` | `[12300, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 3 | 1 | `[12300, 1, 64]` | `[12300, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 3 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 3 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 3 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 4 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 4 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 4 | 1 | `[16400, 1, 64]` | `[16400, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 4 | 1 | `[16400, 1, 64]` | `[16400, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 4 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 4 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 4 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 5 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 5 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 5 | 1 | `[20500, 1, 64]` | `[20500, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 5 | 1 | `[20500, 1, 64]` | `[20500, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 5 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 5 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 5 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |
| maskmem_features | 6 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| maskmem_pos_enc[0] | 6 | 1 | `[1, 64, 64, 64]` | `[1, 64, 64, 64]` | True | direct_copy_shape_candidate_semantics_unverified |
| memory_attention.memory | 6 | 1 | `[24600, 1, 64]` | `[24600, 1, 64]` | True | transient_target_assembled_input |
| memory_attention.memory_pos | 6 | 1 | `[24600, 1, 64]` | `[24600, 1, 64]` | True | transient_target_assembled_input |
| obj_ptr | 6 | 1 | `[1, 256]` | `[1, 256]` | True | direct_copy_shape_candidate_semantics_unverified |
| object_score_logits | 6 | 1 | `[1, 1]` | `[1, 1]` | True | supporting_state_not_primary_translator_target |
| pred_masks | 6 | 1 | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | True | supporting_state_not_primary_translator_target |

> Shape compatibility does not establish representational alignment. A direct-copy candidate still requires a future-frame rollout test.
