from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch


STATE_SCHEMA_VERSION = "sam2-compact-state-v1"


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous().clone()


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: str
    num_bytes: int
    axes: tuple[str, ...]
    positional_encoding: bool = False

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        axes: Iterable[str],
        positional_encoding: bool = False,
    ) -> "TensorMetadata":
        return cls(
            shape=tuple(int(value) for value in tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            num_bytes=int(tensor.numel() * tensor.element_size()),
            axes=tuple(axes),
            positional_encoding=positional_encoding,
        )


@dataclass
class FrameMemory:
    frame_idx: int
    object_id: int
    storage_kind: str
    tensors: dict[str, torch.Tensor]
    tensor_metadata: dict[str, TensorMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.storage_kind not in {"cond_frame_outputs", "non_cond_frame_outputs"}:
            raise ValueError(f"Invalid storage_kind: {self.storage_kind}")
        if not self.tensor_metadata:
            self.tensor_metadata = {
                name: TensorMetadata.from_tensor(
                    tensor,
                    axes=_axes_for(name, tensor.ndim),
                    positional_encoding=name.startswith("maskmem_pos_enc."),
                )
                for name, tensor in self.tensors.items()
            }

    def cpu_clone(self) -> "FrameMemory":
        tensors = {name: _cpu_clone(value) for name, value in self.tensors.items()}
        return FrameMemory(
            frame_idx=self.frame_idx,
            object_id=self.object_id,
            storage_kind=self.storage_kind,
            tensors=tensors,
        )


@dataclass
class MemoryState:
    model_id: str
    checkpoint_id: str
    upstream_commit: str
    sequence_id: str
    switch_frame: int
    prompt: dict[str, Any]
    frames: list[FrameMemory]
    schema_version: str = STATE_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(sorted({frame.frame_idx for frame in self.frames}))

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(sorted({frame.object_id for frame in self.frames}))

    def iter_tensors(
        self, tensor_name: str | None = None
    ) -> Iterable[tuple[FrameMemory, str, torch.Tensor]]:
        for frame in self.frames:
            for name, tensor in frame.tensors.items():
                if tensor_name is None or name == tensor_name:
                    yield frame, name, tensor

    def validate(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported state schema {self.schema_version}; "
                f"expected {STATE_SCHEMA_VERSION}"
            )
        if self.switch_frame < 0:
            raise ValueError("switch_frame must be non-negative")
        seen: set[tuple[int, int, str]] = set()
        for frame in self.frames:
            key = (frame.object_id, frame.frame_idx, frame.storage_kind)
            if key in seen:
                raise ValueError(f"Duplicate compact frame state: {key}")
            seen.add(key)
            if frame.frame_idx > self.switch_frame:
                raise ValueError(
                    f"State frame {frame.frame_idx} exceeds switch_frame={self.switch_frame}"
                )
            for name, tensor in frame.tensors.items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{name} is not a torch.Tensor")
                expected = frame.tensor_metadata[name]
                if tuple(tensor.shape) != expected.shape:
                    raise ValueError(f"Metadata shape mismatch for {name}")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "upstream_commit": self.upstream_commit,
            "sequence_id": self.sequence_id,
            "switch_frame": self.switch_frame,
            "prompt": self.prompt,
            "extra": self.extra,
            "frames": [
                {
                    "frame_idx": frame.frame_idx,
                    "object_id": frame.object_id,
                    "storage_kind": frame.storage_kind,
                    "tensors": {
                        name: _cpu_clone(value) for name, value in frame.tensors.items()
                    },
                }
                for frame in self.frames
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MemoryState":
        state = cls(
            schema_version=str(payload["schema_version"]),
            model_id=str(payload["model_id"]),
            checkpoint_id=str(payload["checkpoint_id"]),
            upstream_commit=str(payload["upstream_commit"]),
            sequence_id=str(payload["sequence_id"]),
            switch_frame=int(payload["switch_frame"]),
            prompt=dict(payload["prompt"]),
            extra=dict(payload.get("extra", {})),
            frames=[
                FrameMemory(
                    frame_idx=int(item["frame_idx"]),
                    object_id=int(item["object_id"]),
                    storage_kind=str(item["storage_kind"]),
                    tensors={
                        str(name): value
                        for name, value in dict(item["tensors"]).items()
                    },
                )
                for item in payload["frames"]
            ],
        )
        state.validate()
        return state


def _axes_for(name: str, ndim: int) -> tuple[str, ...]:
    if name in {"maskmem_features", "pred_masks"} or name.startswith(
        "maskmem_pos_enc."
    ):
        axes = ("object_batch", "channel", "height", "width")
    elif name == "obj_ptr":
        axes = ("object_batch", "channel")
    elif name == "object_score_logits":
        axes = ("object_batch", "score")
    else:
        axes = tuple(f"dim_{idx}" for idx in range(ndim))
    return axes[:ndim]

