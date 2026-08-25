"""SAM 2 predictor-state probing and canonical handoff conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from .state_inspector import InspectionReport, inspect_state
from .state_schema import CanonicalState


COND_KEY = "cond_frame_outputs"
NON_COND_KEY = "non_cond_frame_outputs"
COMPACT_OUTPUT_KEYS = (
    "maskmem_features",
    "maskmem_pos_enc",
    "pred_masks",
    "obj_ptr",
    "object_score_logits",
)


@dataclass(frozen=True)
class _SAM2Record:
    object_index: int
    object_id: Any
    frame_index: int
    is_conditioning: bool
    output: Mapping[str, Any]


def _object_id(inference_state: Mapping[str, Any], object_index: int) -> Any:
    reverse = inference_state.get("obj_idx_to_id", {})
    if object_index in reverse:
        return reverse[object_index]
    ids = inference_state.get("obj_ids", ())
    if object_index < len(ids):
        return ids[object_index]
    return object_index


def _collect_records(
    inference_state: Mapping[str, Any], switch_frame: int
) -> tuple[list[int], dict[int, list[_SAM2Record]]]:
    per_object = inference_state.get("output_dict_per_obj")
    if not isinstance(per_object, Mapping) or not per_object:
        raise ValueError("inference_state has no output_dict_per_obj records")
    object_indices = sorted(int(index) for index in per_object)
    result: dict[int, list[_SAM2Record]] = {}
    for object_index in object_indices:
        object_output = per_object[object_index]
        records: list[_SAM2Record] = []
        for storage_key, is_conditioning in ((COND_KEY, True), (NON_COND_KEY, False)):
            frame_outputs = object_output.get(storage_key, {})
            for raw_frame, output in frame_outputs.items():
                frame = int(raw_frame)
                if frame <= switch_frame:
                    records.append(
                        _SAM2Record(
                            object_index=object_index,
                            object_id=_object_id(inference_state, object_index),
                            frame_index=frame,
                            is_conditioning=is_conditioning,
                            output=output,
                        )
                    )
        records.sort(key=lambda record: (record.frame_index, not record.is_conditioning))
        if not records:
            raise ValueError(
                f"object {object_index} has no finalized output at or before frame {switch_frame}"
            )
        result[object_index] = records
    return object_indices, result


def canonicalize_sam2_inference_state(
    inference_state: Mapping[str, Any],
    *,
    switch_frame: int,
    strict: bool = True,
) -> CanonicalState:
    """Extract finalized per-object histories into ``CanonicalState``.

    The function does not assume Tiny/Large dimensions.  It discovers the first
    complete compact output and validates every later record against it.  Prompt
    tensors, frame-tracking metadata and stored masks are retained as opaque
    metadata; they are not translator inputs.
    """

    object_indices, records_by_object = _collect_records(inference_state, switch_frame)
    complete = [
        record
        for records in records_by_object.values()
        for record in records
        if isinstance(record.output.get("maskmem_features"), torch.Tensor)
        and isinstance(record.output.get("obj_ptr"), torch.Tensor)
        and isinstance(record.output.get("object_score_logits"), torch.Tensor)
    ]
    if not complete:
        raise ValueError(
            "No complete SAM 2 compact output found. Run propagation preflight so "
            "conditioning records receive mask-memory features."
        )
    exemplar = complete[0]
    feature0 = exemplar.output["maskmem_features"]
    pointer0 = exemplar.output["obj_ptr"]
    presence0 = exemplar.output["object_score_logits"]
    if feature0.ndim != 4 or feature0.shape[0] != 1:
        raise ValueError(f"maskmem_features must be [1,C,H,W], got {feature0.shape}")
    if pointer0.ndim != 2 or pointer0.shape[0] != 1:
        raise ValueError(f"obj_ptr must be [1,D], got {pointer0.shape}")
    if presence0.shape != (1, 1):
        raise ValueError(
            f"object_score_logits must be [1,1], got {presence0.shape}"
        )

    _, channels, height, width = feature0.shape
    pointer_dim = pointer0.shape[-1]
    objects = len(object_indices)
    records = max(len(items) for items in records_by_object.values())
    spatial = feature0.new_zeros((1, objects, records, channels, height, width))
    pointer = pointer0.new_zeros((1, objects, records, pointer_dim))
    presence = presence0.new_zeros((1, objects, records, 1))
    frame_indices = torch.full((1, objects, records), -1, dtype=torch.long)
    slot_order = torch.full((1, objects, records), -1, dtype=torch.long)
    is_conditioning = torch.zeros((1, objects, records), dtype=torch.bool)
    validity = torch.zeros((1, objects, records), dtype=torch.bool)
    positional_records: dict[str, Any] = {}
    preserved_masks: dict[str, torch.Tensor] = {}

    for object_slot, object_index in enumerate(object_indices):
        for record_slot, record in enumerate(records_by_object[object_index]):
            output = record.output
            missing = [
                key
                for key in ("maskmem_features", "obj_ptr", "object_score_logits")
                if not isinstance(output.get(key), torch.Tensor)
            ]
            if missing:
                if strict:
                    raise ValueError(
                        f"incomplete SAM 2 output object={record.object_id} "
                        f"frame={record.frame_index}: missing {missing}"
                    )
                continue
            feature = output["maskmem_features"]
            obj_ptr = output["obj_ptr"]
            score = output["object_score_logits"]
            expected = {
                "maskmem_features": (1, channels, height, width),
                "obj_ptr": (1, pointer_dim),
                "object_score_logits": (1, 1),
            }
            actual = {
                "maskmem_features": tuple(feature.shape),
                "obj_ptr": tuple(obj_ptr.shape),
                "object_score_logits": tuple(score.shape),
            }
            if actual != expected:
                raise ValueError(
                    f"SAM 2 runtime shape changed within one state at object "
                    f"{record.object_id}, frame {record.frame_index}: {actual} vs {expected}"
                )
            spatial[0, object_slot, record_slot].copy_(feature[0].to(spatial.device))
            pointer[0, object_slot, record_slot].copy_(obj_ptr[0].to(pointer.device))
            presence[0, object_slot, record_slot].copy_(score[0].to(presence.device))
            frame_indices[0, object_slot, record_slot] = record.frame_index
            slot_order[0, object_slot, record_slot] = record_slot
            is_conditioning[0, object_slot, record_slot] = record.is_conditioning
            validity[0, object_slot, record_slot] = True
            record_key = f"object={object_slot}/record={record_slot}"
            if output.get("maskmem_pos_enc") is not None:
                positional_records[record_key] = output["maskmem_pos_enc"]
            if isinstance(output.get("pred_masks"), torch.Tensor):
                preserved_masks[record_key] = output["pred_masks"]

    object_ids = tuple(_object_id(inference_state, index) for index in object_indices)
    metadata = {
        "source": "sam2_inference_state",
        "object_indices": object_indices,
        "frames_tracked_per_obj": deepcopy(
            inference_state.get("frames_tracked_per_obj", {})
        ),
        "preserved_inputs": {
            "point_inputs_per_obj": deepcopy(
                inference_state.get("point_inputs_per_obj", {})
            ),
            "mask_inputs_per_obj": deepcopy(inference_state.get("mask_inputs_per_obj", {})),
        },
        "preserved_pred_masks": preserved_masks,
        "storage_device": str(inference_state.get("storage_device", "unknown")),
        "compute_device": str(inference_state.get("device", "unknown")),
        "num_frames": inference_state.get("num_frames"),
        "video_height": inference_state.get("video_height"),
        "video_width": inference_state.get("video_width"),
    }
    return CanonicalState(
        spatial_memory=spatial,
        object_pointer=pointer,
        presence_logits=presence,
        frame_indices=frame_indices,
        slot_order=slot_order,
        is_conditioning=is_conditioning,
        validity=validity,
        object_ids=object_ids,
        switch_frame=int(switch_frame),
        positional_information={
            "policy": "source_observed_do_not_translate",
            "records": positional_records,
        },
        metadata=metadata,
    ).validate()


def probe_sam2_inference_state(inference_state: Mapping[str, Any]) -> InspectionReport:
    """Recursively inventory the full predictor container without modifying it."""

    return inspect_state(inference_state, root_name="sam2_inference_state")


def materialize_sam2_history(
    state: CanonicalState,
    *,
    positional_factory: Callable[
        [Any, int, bool, torch.Tensor], list[torch.Tensor] | None
    ],
    mask_factory: Callable[
        [Any, int, bool, torch.Tensor], torch.Tensor
    ]
    | None = None,
) -> dict[int, dict[str, dict[int, dict[str, Any]]]]:
    """Materialize translated compact histories for a target predictor.

    ``positional_factory`` must be target-owned.  This intentional boundary keeps
    source positional encodings out of the learned translator and makes an actual
    injection smoke test fail closed when target regeneration is unavailable.
    """

    state.validate()
    if state.spatial_memory.shape[0] != 1:
        raise ValueError("SAM 2 materialization currently requires B=1")
    result: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    preserved_masks = state.metadata.get("preserved_pred_masks", {})
    for object_slot, object_id in enumerate(state.object_ids):
        result[object_slot] = {COND_KEY: {}, NON_COND_KEY: {}}
        for record_slot in range(state.spatial_memory.shape[2]):
            if not bool(state.validity[0, object_slot, record_slot]):
                continue
            frame = int(state.frame_indices[0, object_slot, record_slot].item())
            is_cond = bool(state.is_conditioning[0, object_slot, record_slot].item())
            feature = state.spatial_memory[0, object_slot, record_slot].unsqueeze(0)
            positional = positional_factory(object_id, frame, is_cond, feature)
            if positional is None:
                raise ValueError(
                    "target positional_factory returned None for a valid memory record"
                )
            record_key = f"object={object_slot}/record={record_slot}"
            mask = preserved_masks.get(record_key)
            if not isinstance(mask, torch.Tensor):
                raise ValueError(
                    f"missing preserved pred_masks for {record_key}; history is not "
                    "continuation-closed"
                )
            if mask_factory is not None:
                mask = mask_factory(object_id, frame, is_cond, mask)
            output = {
                "maskmem_features": feature,
                "maskmem_pos_enc": positional,
                "pred_masks": mask,
                "obj_ptr": state.object_pointer[0, object_slot, record_slot].unsqueeze(0),
                "object_score_logits": state.presence_logits[
                    0, object_slot, record_slot
                ].unsqueeze(0),
            }
            storage_key = COND_KEY if is_cond else NON_COND_KEY
            result[object_slot][storage_key][frame] = output
    return result
