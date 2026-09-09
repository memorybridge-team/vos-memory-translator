from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vos_memory_inspector.case_cache import load_case_cache, write_case_cache
from vos_memory_inspector.state_schema import CanonicalState


def _state(switch_frame: int, value: float) -> CanonicalState:
    records = switch_frame + 1
    indices = torch.arange(records).view(1, 1, records)
    return CanonicalState(
        spatial_memory=torch.full((1, 1, records, 3, 2, 2), value),
        object_pointer=torch.full((1, 1, records, 4), value),
        presence_logits=torch.full((1, 1, records, 1), value),
        frame_indices=indices.clone(),
        slot_order=indices.clone(),
        is_conditioning=torch.zeros(1, 1, records, dtype=torch.bool),
        validity=torch.ones(1, 1, records, dtype=torch.bool),
        object_ids=(1,),
        switch_frame=switch_frame,
        metadata={"preserved_pred_masks": {}},
    ).validate()


def test_case_cache_roundtrip_and_checksum(tmp_path: Path) -> None:
    output = tmp_path / "case.pt"
    source = _state(2, 1.0)
    target = _state(2, 2.0)
    summary = write_case_cache(
        output,
        source_canonical=source,
        target_canonical=target,
        source_prefix_masks={frame: torch.full((1, 1, 4, 5), frame) for frame in range(3)},
        target_oracle_future_masks={
            frame: torch.full((1, 1, 4, 5), frame) for frame in range(3, 6)
        },
        metadata={"video_id": "demo", "path_policy": "runtime_paths_not_stored"},
    )

    assert summary["source_prefix_frames"] == 3
    assert output.is_file()
    assert output.with_suffix(".pt.sha256").is_file()
    assert not output.with_suffix(".pt.partial").exists()
    restored = load_case_cache(output)
    assert restored["metadata"]["video_id"] == "demo"
    assert restored["source_canonical"].switch_frame == 2
    assert list(restored["target_oracle_future_masks"]) == [3, 4, 5]


def test_case_cache_rejects_noncontiguous_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source prefix masks"):
        write_case_cache(
            tmp_path / "case.pt",
            source_canonical=_state(2, 1.0),
            target_canonical=_state(2, 2.0),
            source_prefix_masks={0: torch.zeros(1), 2: torch.zeros(1)},
            target_oracle_future_masks={3: torch.zeros(1)},
            metadata={},
        )


def test_case_cache_detects_file_tampering(tmp_path: Path) -> None:
    output = tmp_path / "case.pt"
    write_case_cache(
        output,
        source_canonical=_state(1, 1.0),
        target_canonical=_state(1, 2.0),
        source_prefix_masks={0: torch.zeros(1), 1: torch.zeros(1)},
        target_oracle_future_masks={2: torch.zeros(1)},
        metadata={},
    )
    with output.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_case_cache(output)


def test_case_cache_rejects_incomplete_future_against_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="final frame"):
        write_case_cache(
            tmp_path / "case.pt",
            source_canonical=_state(1, 1.0),
            target_canonical=_state(1, 2.0),
            source_prefix_masks={0: torch.zeros(1), 1: torch.zeros(1)},
            target_oracle_future_masks={2: torch.zeros(1)},
            metadata={"switch_frame": 1, "num_frames": 4},
        )
