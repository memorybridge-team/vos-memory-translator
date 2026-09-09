from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vos_memory_inspector.adapter import (
    ReplayGuard,
    Sam2StateAdapter,
    initialize_replay_free_suffix_state,
)
from vos_memory_inspector.paired_dataset import (
    PairedState,
    PairedStateDataset,
    PairShardWriter,
    iter_aligned_tensors,
    validate_video_splits,
)
from vos_memory_inspector.state import FrameMemory, MemoryState


class FakePositionEncoding:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor)


class FakePredictor:
    mem_dim = 2
    hidden_dim = 3
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.memory_encoder = SimpleNamespace(position_encoding=FakePositionEncoding())
        self.feature_requests: list[int] = []

    def _obj_id_to_idx(self, state, obj_id):
        if obj_id in state["obj_id_to_idx"]:
            return state["obj_id_to_idx"][obj_id]
        idx = len(state["obj_id_to_idx"])
        state["obj_id_to_idx"][obj_id] = idx
        state["obj_idx_to_id"][idx] = obj_id
        state["output_dict_per_obj"][idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        state["frames_tracked_per_obj"][idx] = {}
        return idx

    def _get_image_feature(self, state, frame_idx, batch_size):
        self.feature_requests.append(frame_idx)
        return (frame_idx, batch_size)

    def init_state(self, **kwargs):
        state = empty_inference_state()
        state["images"] = torch.arange(12, dtype=torch.float32).view(2, 1, 2, 3)
        state["num_frames"] = 2
        state["cached_features"] = {0: ("warmed",)}
        return state


def empty_inference_state():
    return {
        "output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "constants": {},
        "storage_device": torch.device("cpu"),
        "device": torch.device("cpu"),
        "num_frames": 3,
        "video_height": 8,
        "video_width": 8,
    }


def make_state(model_id: str) -> MemoryState:
    return MemoryState(
        model_id=model_id,
        checkpoint_id=f"{model_id}.pt",
        upstream_commit="abc",
        sequence_id="video-a",
        switch_frame=1,
        prompt={"type": "mask", "frame_idx": 0, "object_id": 1},
        frames=[
            FrameMemory(
                frame_idx=0,
                object_id=1,
                storage_kind="cond_frame_outputs",
                tensors={
                    "maskmem_features": torch.randn(1, 2, 2, 2),
                    "maskmem_pos_enc.0": torch.zeros(1, 2, 2, 2),
                    "pred_masks": torch.randn(1, 1, 4, 4),
                    "obj_ptr": torch.randn(1, 3),
                    "object_score_logits": torch.randn(1, 1),
                },
            ),
            FrameMemory(
                frame_idx=1,
                object_id=1,
                storage_kind="non_cond_frame_outputs",
                tensors={
                    "maskmem_features": torch.randn(1, 2, 2, 2),
                    "maskmem_pos_enc.0": torch.zeros(1, 2, 2, 2),
                    "pred_masks": torch.randn(1, 1, 4, 4),
                    "obj_ptr": torch.randn(1, 3),
                    "object_score_logits": torch.randn(1, 1),
                },
            ),
        ],
        extra={"num_frames": 3, "video_height": 8, "video_width": 8},
    )


def test_pair_shards_stream_from_disk(tmp_path: Path):
    pair = PairedState(
        pair_id="video-a-t1",
        direction="small_to_large",
        source=make_state("sam2-small"),
        target=make_state("sam2-large"),
    )
    with PairShardWriter(tmp_path, pairs_per_shard=1) as writer:
        writer.add(pair)

    loaded = list(PairedStateDataset(tmp_path / "manifest.json"))
    assert len(loaded) == 1
    aligned = list(iter_aligned_tensors(loaded[0], "maskmem_features"))
    assert len(aligned) == 2
    assert aligned[0][2]["sequence_id"] == "video-a"


def test_video_split_leakage_is_rejected():
    validate_video_splits({"train": ["a", "b"], "val": ["c"]})
    with pytest.raises(ValueError, match="split leakage"):
        validate_video_splits({"train": ["a"], "val": ["a"]})


def test_adapter_injects_target_native_position_encoding():
    predictor = FakePredictor()
    target = empty_inference_state()
    state = make_state("translated")
    Sam2StateAdapter(predictor).inject_state(target, state)

    output = target["output_dict_per_obj"][0]["cond_frame_outputs"][0]
    assert output["maskmem_features"].dtype == torch.bfloat16
    assert torch.equal(
        output["maskmem_pos_enc"][0],
        torch.ones_like(output["maskmem_features"]),
    )
    assert target["frames_tracked_per_obj"][0][1]["injected"] is True


def test_replay_guard_rejects_prefix_access():
    predictor = FakePredictor()
    with ReplayGuard(predictor, switch_frame=2) as guard:
        assert predictor._get_image_feature({}, 3, 1) == (3, 1)
        with pytest.raises(RuntimeError, match="Replay-free violation"):
            predictor._get_image_feature({}, 2, 1)
    assert guard.requested_frames == [3, 2]


def test_suffix_state_preserves_global_frame_indices(tmp_path: Path):
    predictor = FakePredictor()
    state = initialize_replay_free_suffix_state(
        predictor,
        suffix_video_dir=tmp_path,
        switch_frame=1,
        total_num_frames=4,
    )
    assert state["images"].shape[0] == 4
    assert torch.count_nonzero(state["images"][:2]) == 0
    assert set(state["cached_features"]) == {2}
    assert state["replay_free"]["prefix_pixels_loaded"] is False

