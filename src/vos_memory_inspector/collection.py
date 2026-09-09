from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .adapter import Sam2StateAdapter
from .paired_dataset import PairedState, PairShardWriter
from .runner import load_binary_prompt
from .state import MemoryState
from .upstream import verify_sam2_checkout


@dataclass(frozen=True)
class ModelRunSpec:
    model_id: str
    config_file: str
    checkpoint: Path


def collect_sam2_pair(
    *,
    sam2_repo: str | Path,
    source: ModelRunSpec,
    target: ModelRunSpec,
    direction: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    output_dir: str | Path,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    allow_upstream_mismatch: bool = False,
) -> PairedState:
    """Run source and target sequentially and persist one paired CPU shard."""

    sam2_repo = Path(sam2_repo).resolve()
    video_dir = Path(video_dir).resolve()
    upstream_commit = verify_sam2_checkout(
        sam2_repo, allow_mismatch=allow_upstream_mismatch
    )
    if direction not in {"small_to_large", "large_to_small"}:
        raise ValueError(f"Unsupported direction: {direction}")
    expected = (
        ("small", "large") if direction == "small_to_large" else ("large", "small")
    )
    if expected[0] not in source.model_id.lower() or expected[1] not in target.model_id.lower():
        raise ValueError(
            f"Direction {direction} conflicts with model IDs "
            f"{source.model_id!r} -> {target.model_id!r}"
        )
    mask = load_binary_prompt(prompt_mask, object_id)
    prompt = {
        "type": "mask",
        "frame_idx": 0,
        "object_id": object_id,
        "mask_file": Path(prompt_mask).name,
    }
    source_state = _run_prefix(
        sam2_repo=sam2_repo,
        spec=source,
        video_dir=video_dir,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
        prompt=prompt,
        upstream_commit=upstream_commit,
        device=device,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    _release_cuda()
    target_state = _run_prefix(
        sam2_repo=sam2_repo,
        spec=target,
        video_dir=video_dir,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
        prompt=prompt,
        upstream_commit=upstream_commit,
        device=device,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    _release_cuda()
    pair = PairedState(
        pair_id=f"{video_dir.name}-obj{object_id}-t{switch_frame}-{direction}",
        direction=direction,
        source=source_state,
        target=target_state,
    )
    pair.validate()
    with PairShardWriter(output_dir, pairs_per_shard=1) as writer:
        writer.add(pair)
    return pair


def _run_prefix(
    *,
    sam2_repo: Path,
    spec: ModelRunSpec,
    video_dir: Path,
    mask: Any,
    object_id: int,
    switch_frame: int,
    prompt: dict[str, Any],
    upstream_commit: str,
    device: str,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> MemoryState:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {spec.checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=spec.config_file,
        ckpt_path=str(spec.checkpoint.resolve()),
        device=device,
    )
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    inference_state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if switch_frame >= int(inference_state["num_frames"]):
        raise ValueError("switch_frame is outside the video")
    predictor.add_new_mask(
        inference_state, frame_idx=0, obj_id=object_id, mask=mask
    )
    final_frame = None
    for frame_idx, _, _ in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=switch_frame,
        reverse=False,
    ):
        final_frame = int(frame_idx)
        if final_frame >= switch_frame:
            break
    if final_frame != switch_frame:
        raise RuntimeError(
            f"Prefix inference ended at {final_frame}; expected {switch_frame}"
        )
    state = Sam2StateAdapter(predictor).extract_state(
        inference_state,
        model_id=spec.model_id,
        checkpoint_id=str(spec.checkpoint),
        upstream_commit=upstream_commit,
        sequence_id=video_dir.name,
        switch_frame=switch_frame,
        prompt=prompt,
    )
    del inference_state, predictor
    return state


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

