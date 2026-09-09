from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from .state import FrameMemory, MemoryState


COMPACT_KEYS = (
    "maskmem_features",
    "maskmem_pos_enc",
    "pred_masks",
    "obj_ptr",
    "object_score_logits",
)


class Sam2StateAdapter:
    """Version-bound adapter around the pinned SAM 2 video predictor state."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self._verify_predictor_contract()

    def _verify_predictor_contract(self) -> None:
        required = (
            "_obj_id_to_idx",
            "_get_image_feature",
            "memory_encoder",
            "mem_dim",
            "hidden_dim",
        )
        missing = [name for name in required if not hasattr(self.predictor, name)]
        if missing:
            raise RuntimeError(
                "SAM 2 predictor does not satisfy the pinned adapter contract: "
                + ", ".join(missing)
            )

    def extract_state(
        self,
        inference_state: dict[str, Any],
        *,
        model_id: str,
        checkpoint_id: str,
        upstream_commit: str,
        sequence_id: str,
        switch_frame: int,
        prompt: dict[str, Any],
    ) -> MemoryState:
        _validate_inference_state(inference_state)
        frames: list[FrameMemory] = []
        for object_idx, output_by_kind in inference_state[
            "output_dict_per_obj"
        ].items():
            object_id = int(inference_state["obj_idx_to_id"][object_idx])
            for storage_kind in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for frame_idx, compact in sorted(output_by_kind[storage_kind].items()):
                    if int(frame_idx) > switch_frame:
                        continue
                    tensors = _flatten_compact_output(compact)
                    frames.append(
                        FrameMemory(
                            frame_idx=int(frame_idx),
                            object_id=object_id,
                            storage_kind=storage_kind,
                            tensors={
                                name: value.detach().to("cpu").contiguous().clone()
                                for name, value in tensors.items()
                            },
                        )
                    )
        state = MemoryState(
            model_id=model_id,
            checkpoint_id=Path(checkpoint_id).name,
            upstream_commit=upstream_commit,
            sequence_id=sequence_id,
            switch_frame=switch_frame,
            prompt=dict(prompt),
            frames=frames,
            extra={
                "num_frames": int(inference_state["num_frames"]),
                "video_height": int(inference_state["video_height"]),
                "video_width": int(inference_state["video_width"]),
            },
        )
        state.validate()
        return state

    def inject_state(
        self,
        inference_state: dict[str, Any],
        state: MemoryState,
        *,
        replace_existing: bool = False,
        regenerate_position_encoding: bool = True,
    ) -> None:
        state.validate()
        _validate_inference_state(inference_state)
        if not replace_existing and any(
            outputs[kind]
            for outputs in inference_state["output_dict_per_obj"].values()
            for kind in ("cond_frame_outputs", "non_cond_frame_outputs")
        ):
            raise ValueError("Target state already contains frame outputs")

        inference_state["constants"].pop("maskmem_pos_enc", None)
        for frame in state.frames:
            obj_idx = self.predictor._obj_id_to_idx(
                inference_state, frame.object_id
            )
            output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if frame.frame_idx in output_dict[frame.storage_kind] and not replace_existing:
                raise ValueError(
                    f"Target already contains object={frame.object_id}, "
                    f"frame={frame.frame_idx}, kind={frame.storage_kind}"
                )
            compact = self._to_target_compact_output(
                frame.tensors,
                inference_state,
                regenerate_position_encoding=regenerate_position_encoding,
            )
            output_dict[frame.storage_kind][frame.frame_idx] = compact
            inference_state["frames_tracked_per_obj"][obj_idx][frame.frame_idx] = {
                "reverse": False,
                "injected": True,
            }

    def _to_target_compact_output(
        self,
        tensors: dict[str, torch.Tensor],
        inference_state: dict[str, Any],
        *,
        regenerate_position_encoding: bool,
    ) -> dict[str, Any]:
        required = {
            "maskmem_features",
            "pred_masks",
            "obj_ptr",
            "object_score_logits",
        }
        missing = sorted(required.difference(tensors))
        if missing:
            raise ValueError(f"Translated compact output is missing: {missing}")

        storage_device = inference_state["storage_device"]
        compute_device = inference_state["device"]
        feature = tensors["maskmem_features"]
        pointer = tensors["obj_ptr"]
        if feature.ndim != 4 or int(feature.shape[1]) != int(self.predictor.mem_dim):
            raise ValueError(
                f"maskmem_features target schema mismatch: shape={tuple(feature.shape)}, "
                f"expected channel={self.predictor.mem_dim}"
            )
        if pointer.ndim != 2 or int(pointer.shape[-1]) != int(
            self.predictor.hidden_dim
        ):
            raise ValueError(
                f"obj_ptr target schema mismatch: shape={tuple(pointer.shape)}, "
                f"expected last dimension={self.predictor.hidden_dim}"
            )
        stored_feature = feature.to(dtype=torch.bfloat16, device=storage_device)
        if regenerate_position_encoding:
            position = self.predictor.memory_encoder.position_encoding(
                stored_feature.to(device=compute_device, dtype=torch.float32)
            ).to(dtype=stored_feature.dtype)
            position_list = [position]
        else:
            position_names = sorted(
                name for name in tensors if name.startswith("maskmem_pos_enc.")
            )
            if not position_names:
                raise ValueError("No maskmem_pos_enc.* tensors available for direct PE copy")
            position_list = [tensors[name].to(compute_device) for name in position_names]

        canonical = inference_state["constants"].get("maskmem_pos_enc")
        if canonical is None:
            canonical = [value[0:1].clone() for value in position_list]
            inference_state["constants"]["maskmem_pos_enc"] = canonical
        batch_size = int(stored_feature.shape[0])
        expanded_position = [
            value.expand(batch_size, -1, -1, -1) for value in canonical
        ]
        return {
            "maskmem_features": stored_feature,
            "maskmem_pos_enc": expanded_position,
            "pred_masks": tensors["pred_masks"].to(storage_device),
            "obj_ptr": pointer.to(compute_device),
            "object_score_logits": tensors["object_score_logits"].to(compute_device),
        }


class ReplayGuard:
    """Reject target image-feature access to the handed-off prefix."""

    def __init__(self, predictor: Any, switch_frame: int) -> None:
        self.predictor = predictor
        self.switch_frame = switch_frame
        self.requested_frames: list[int] = []
        self._original: Any | None = None

    def __enter__(self) -> "ReplayGuard":
        if self._original is not None:
            raise RuntimeError("ReplayGuard is already active")
        self._original = self.predictor._get_image_feature

        def guarded(inference_state: dict[str, Any], frame_idx: int, batch_size: int):
            frame_idx = int(frame_idx)
            self.requested_frames.append(frame_idx)
            if frame_idx <= self.switch_frame:
                raise RuntimeError(
                    f"Replay-free violation: target requested prefix frame {frame_idx} "
                    f"at or before switch_frame={self.switch_frame}"
                )
            return self._original(inference_state, frame_idx, batch_size)

        self.predictor._get_image_feature = guarded
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._original is not None:
            self.predictor._get_image_feature = self._original
            self._original = None


def initialize_replay_free_suffix_state(
    predictor: Any,
    *,
    suffix_video_dir: str | Path,
    switch_frame: int,
    total_num_frames: int,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
) -> dict[str, Any]:
    """Initialize official SAM 2 on suffix frames and preserve global indices.

    The suffix directory must start with global frame ``switch_frame + 1``. The
    official predictor loads only that directory. Placeholder pixels occupy the
    inaccessible prefix indices so existing global frame keys remain valid.
    """

    if switch_frame < 0 or total_num_frames <= switch_frame + 1:
        raise ValueError("A non-empty suffix after switch_frame is required")
    state = predictor.init_state(
        video_path=str(Path(suffix_video_dir).resolve()),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )
    suffix_count = int(state["num_frames"])
    expected_suffix = total_num_frames - switch_frame - 1
    if suffix_count != expected_suffix:
        raise ValueError(
            f"Suffix contains {suffix_count} frames; expected {expected_suffix} "
            f"for total_num_frames={total_num_frames} and switch_frame={switch_frame}"
        )
    images = state["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("Replay-free initialization requires eager tensor frame loading")
    prefix = images.new_zeros((switch_frame + 1, *images.shape[1:]))
    state["images"] = torch.cat((prefix, images), dim=0)
    state["num_frames"] = int(total_num_frames)
    cached = state["cached_features"]
    if set(cached) != {0}:
        raise RuntimeError(f"Unexpected initial SAM 2 feature cache keys: {sorted(cached)}")
    cached[switch_frame + 1] = cached.pop(0)
    state["replay_free"] = {
        "switch_frame": switch_frame,
        "loaded_suffix_frames": suffix_count,
        "prefix_pixels_loaded": False,
    }
    return state


@contextmanager
def frozen_models(*models: torch.nn.Module) -> Iterator[None]:
    previous: list[list[bool]] = []
    for model in models:
        flags = [parameter.requires_grad for parameter in model.parameters()]
        previous.append(flags)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for model, flags in zip(models, previous):
            for parameter, flag in zip(model.parameters(), flags):
                parameter.requires_grad_(flag)


def _validate_inference_state(state: dict[str, Any]) -> None:
    required = {
        "output_dict_per_obj",
        "frames_tracked_per_obj",
        "obj_idx_to_id",
        "constants",
        "storage_device",
        "device",
        "num_frames",
        "video_height",
        "video_width",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"SAM 2 inference_state is missing keys: {missing}")


def _flatten_compact_output(compact: dict[str, Any]) -> dict[str, torch.Tensor]:
    missing = sorted(set(COMPACT_KEYS).difference(compact))
    if missing:
        raise RuntimeError(f"SAM 2 compact output is missing keys: {missing}")
    tensors: dict[str, torch.Tensor] = {}
    for key in ("maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"):
        value = compact[key]
        if value is not None:
            tensors[key] = value
    positions = compact["maskmem_pos_enc"]
    if positions is not None:
        for idx, value in enumerate(positions):
            tensors[f"maskmem_pos_enc.{idx}"] = value
    return tensors

