"""Checkpoint-backed same-model export/inject continuation validation."""

from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .runner import load_binary_prompt
from .artifacts import write_handoff_artifacts
from .case_cache import load_case_cache, write_case_cache
from .metrics import evaluate_state
from .sam2_state import (
    canonicalize_sam2_inference_state,
    init_sam2_inference_state_without_warmup,
    inject_sam2_canonical_state,
)
from .upstream import verify_sam2_checkout
from .translators import DirectCopyTranslator


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _start_resource_measurement(device: str) -> float:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _resource_measurement(started_at: float, device: str) -> dict[str, Any]:
    peak_cuda_memory = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        peak_cuda_memory = int(torch.cuda.max_memory_allocated())
    return {
        "wall_time_seconds": float(time.perf_counter() - started_at),
        "peak_cuda_memory_bytes": peak_cuda_memory,
    }


def _collect_native(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
) -> tuple[Any, dict[int, torch.Tensor]]:
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=object_id,
        mask=mask,
    )
    canonical = None
    future: dict[int, torch.Tensor] = {}
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=int(inference_state["num_frames"]),
        reverse=False,
    ):
        frame = int(frame_idx)
        if frame == switch_frame:
            canonical = canonicalize_sam2_inference_state(
                inference_state,
                switch_frame=switch_frame,
                strict=True,
            )
        if frame > switch_frame:
            future[frame] = masks.detach().cpu().float()
    if canonical is None:
        raise RuntimeError("native run did not reach switch_frame")
    return canonical, future


def _collect_prefix(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
):
    canonical, _masks = _collect_prefix_reference(
        predictor,
        inference_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    return canonical


def _collect_prefix_reference(
    predictor: Any,
    inference_state: dict[str, Any],
    *,
    mask: np.ndarray,
    object_id: int,
    switch_frame: int,
) -> tuple[Any, dict[int, torch.Tensor]]:
    predictor.add_new_mask(
        inference_state,
        frame_idx=0,
        obj_id=object_id,
        mask=mask,
    )
    prefix_masks: dict[int, torch.Tensor] = {}
    canonical = None
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=switch_frame,
        reverse=False,
    ):
        frame = int(frame_idx)
        prefix_masks[frame] = masks.detach().cpu().float()
        if frame == switch_frame:
            canonical = canonicalize_sam2_inference_state(
                inference_state,
                switch_frame=switch_frame,
                strict=True,
            )
    if canonical is None:
        raise RuntimeError("source run did not reach switch_frame")
    return canonical, prefix_masks


def prepare_cross_model_case_reference(
    *,
    sam2_repo: str | Path,
    source_config_file: str,
    source_checkpoint: str | Path,
    source_model_id: str,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    output: str | Path,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compute source prefix and target oracle once for reuse by all baselines."""

    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    for label, checkpoint in (("source", source_checkpoint), ("target", target_checkpoint)):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    prompt = load_binary_prompt(prompt_mask, object_id)
    _seed_everything(seed)
    source_predictor = build_sam2_video_predictor(
        config_file=source_config_file,
        ckpt_path=str(source_checkpoint),
        device=device,
    )
    source_state = source_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(source_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    source_canonical, source_prefix_masks = _collect_prefix_reference(
        source_predictor,
        source_state,
        mask=prompt,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(source_state["num_frames"])
    del source_state, source_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    target_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    target_state = target_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    target_canonical, target_oracle_future = _collect_native(
        target_predictor,
        target_state,
        mask=prompt,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    del target_state, target_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metadata = {
        "source_model_id": source_model_id,
        "target_model_id": target_model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": object_id,
        "switch_frame": switch_frame,
        "num_frames": num_frames,
        "seed": seed,
        "device": device,
        "path_policy": "runtime_paths_not_stored",
    }
    preparation_resources = _resource_measurement(started_at, device)
    metadata["preparation_resources"] = preparation_resources
    cache = write_case_cache(
        output,
        source_canonical=source_canonical,
        target_canonical=target_canonical,
        source_prefix_masks=source_prefix_masks,
        target_oracle_future_masks=target_oracle_future,
        metadata=metadata,
    )
    return {
        **metadata,
        "source_contract": source_canonical.contract_dict(),
        "target_contract": target_canonical.contract_dict(),
        "resources": preparation_resources,
        "cache": cache,
    }


def run_cached_translator_handoff(
    *,
    case_cache: str | Path,
    sam2_repo: str | Path,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    annotation_dir: str | Path | None = None,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run only target continuation from a checksummed prepared case cache."""

    payload = load_case_cache(case_cache)
    metadata = payload["metadata"]
    if metadata.get("target_model_id") != target_model_id:
        raise ValueError(
            f"case cache target {metadata.get('target_model_id')!r} differs from "
            f"requested {target_model_id!r}"
        )
    video_dir = Path(video_dir).resolve()
    if metadata.get("video_id") != video_dir.name:
        raise ValueError(
            f"case cache video {metadata.get('video_id')!r} differs from {video_dir.name!r}"
        )
    sam2_repo = Path(sam2_repo).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    if metadata.get("upstream_commit") != commit:
        raise ValueError("case cache and runtime SAM 2 commits differ")
    if not target_checkpoint.is_file():
        raise FileNotFoundError(f"target checkpoint not found: {target_checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    source_canonical = payload["source_canonical"]
    target_canonical = payload["target_canonical"]
    oracle_future = payload["target_oracle_future_masks"]
    if translator is None:
        translator = DirectCopyTranslator(target_canonical.spec)
    target_spec = getattr(translator, "target_spec", None)
    if target_spec is not None and target_spec != target_canonical.spec:
        raise ValueError(
            f"translator target spec {target_spec} differs from cached target spec "
            f"{target_canonical.spec}"
        )
    started_at = _start_resource_measurement(device)
    translation_started_at = time.perf_counter()
    translated = translator.translate(source_canonical)
    translation_wall_time_seconds = time.perf_counter() - translation_started_at

    _seed_everything(seed)
    predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    predictor.forward_image = counted_forward_image
    inference_state = init_sam2_inference_state_without_warmup(
        predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        translated,
        predictor=predictor,
        inference_state=inference_state,
    )
    calls_after_injection = len(backbone_calls)
    switch_frame = int(metadata["switch_frame"])
    start_frame = switch_frame + 1
    candidate_future: dict[int, torch.Tensor] = {}
    for frame_idx, _object_ids, masks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=int(metadata["num_frames"]) - start_frame,
        reverse=False,
    ):
        candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    downstream = _compare_future_masks(oracle_future, candidate_future)
    candidate_resources = _resource_measurement(started_at, device)
    state_alignment = evaluate_state(translated, target_canonical)
    report = {
        "source_model_id": metadata["source_model_id"],
        "target_model_id": target_model_id,
        "translator": translator_name,
        "translator_parameter_count": int(translator.parameter_count()),
        "translation_wall_time_seconds": float(translation_wall_time_seconds),
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "object_id": int(metadata["object_id"]),
        "switch_frame": switch_frame,
        "case_cache_sha256": Path(case_cache).with_suffix(
            Path(case_cache).suffix + ".sha256"
        ).read_text(encoding="ascii").split()[0],
        "state_alignment_to_target_native": state_alignment,
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls) - calls_after_injection,
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        # Keep the generic key for the existing report renderer while naming the
        # scope explicitly for machine-readable comparisons.
        "resources": candidate_resources,
        "resources_candidate_only": candidate_resources,
        "resources_shared_reference_preparation": metadata.get("preparation_resources"),
    }
    if artifact_dir is not None:
        if annotation_dir is None:
            raise ValueError("annotation_dir is required when artifact_dir is set")
        report["artifacts"] = write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=annotation_dir,
            object_id=int(metadata["object_id"]),
            oracle_masks=oracle_future,
            candidate_masks=candidate_future,
            output_dir=artifact_dir,
            report=report,
            candidate_label=candidate_label,
        )
    del inference_state, predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def _compare_future_masks(
    native: dict[int, torch.Tensor],
    injected: dict[int, torch.Tensor],
) -> dict[str, Any]:
    if native.keys() != injected.keys():
        raise ValueError(
            f"continuation frames differ: native={sorted(native)}, "
            f"injected={sorted(injected)}"
        )
    rows: list[dict[str, Any]] = []
    for frame in native:
        reference = native[frame]
        prediction = injected[frame]
        if reference.shape != prediction.shape:
            raise ValueError(
                f"frame {frame} mask shapes differ: {reference.shape} vs "
                f"{prediction.shape}"
            )
        difference = prediction - reference
        reference_binary = reference > 0
        prediction_binary = prediction > 0
        intersection = (reference_binary & prediction_binary).sum().item()
        union = (reference_binary | prediction_binary).sum().item()
        rows.append(
            {
                "frame": frame,
                "mse": float(difference.square().mean()),
                "max_abs_error": float(difference.abs().max()),
                "binary_iou": 1.0 if union == 0 else float(intersection / union),
            }
        )
    return {
        "frames": rows,
        "mean_mse": float(sum(row["mse"] for row in rows) / max(len(rows), 1)),
        "max_abs_error": float(max((row["max_abs_error"] for row in rows), default=0.0)),
        "mean_binary_iou": float(
            sum(row["binary_iou"] for row in rows) / max(len(rows), 1)
        ),
    }


def run_same_checkpoint_roundtrip(
    *,
    sam2_repo: str | Path,
    config_file: str,
    checkpoint: str | Path,
    model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """Compare native continuation with export→inject continuation."""

    sam2_repo = Path(sam2_repo).resolve()
    checkpoint = Path(checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    _seed_everything(seed)
    mask = load_binary_prompt(prompt_mask, object_id)
    native_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    native_state = native_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(native_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    canonical, native_future = _collect_native(
        native_predictor,
        native_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(native_state["num_frames"])
    del native_state, native_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    injected_predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=str(checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = injected_predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    injected_predictor.forward_image = counted_forward_image
    injected_state = init_sam2_inference_state_without_warmup(
        injected_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        canonical,
        predictor=injected_predictor,
        inference_state=injected_state,
    )
    calls_after_injection = len(backbone_calls)
    injected_future: dict[int, torch.Tensor] = {}
    start_frame = switch_frame + 1
    for frame_idx, _object_ids, masks in injected_predictor.propagate_in_video(
        injected_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=num_frames - start_frame,
        reverse=False,
    ):
        injected_future[int(frame_idx)] = masks.detach().cpu().float()
    comparison = _compare_future_masks(native_future, injected_future)
    report = {
        "model_id": model_id,
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "future_frames": sorted(injected_future),
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection
        - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls)
        - calls_after_injection,
        "comparison": comparison,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    del injected_state, injected_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_cross_model_translator_handoff(
    *,
    sam2_repo: str | Path,
    source_config_file: str,
    source_checkpoint: str | Path,
    source_model_id: str,
    target_config_file: str,
    target_checkpoint: str | Path,
    target_model_id: str,
    video_dir: str | Path,
    prompt_mask: str | Path,
    object_id: int,
    switch_frame: int,
    device: str = "cuda",
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    seed: int = 7,
    artifact_dir: str | Path | None = None,
    translator: Any | None = None,
    translator_name: str = "direct_copy",
    candidate_label: str = "Direct Copy",
) -> dict[str, Any]:
    """Run an end-to-end cross-model handoff with a supplied translator."""

    sam2_repo = Path(sam2_repo).resolve()
    source_checkpoint = Path(source_checkpoint).resolve()
    target_checkpoint = Path(target_checkpoint).resolve()
    video_dir = Path(video_dir).resolve()
    commit = verify_sam2_checkout(sam2_repo)
    for label, checkpoint in (
        ("source", source_checkpoint),
        ("target", target_checkpoint),
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint}")
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    started_at = _start_resource_measurement(device)
    mask = load_binary_prompt(prompt_mask, object_id)
    _seed_everything(seed)
    source_predictor = build_sam2_video_predictor(
        config_file=source_config_file,
        ckpt_path=str(source_checkpoint),
        device=device,
    )
    source_state = source_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    if not 0 <= switch_frame < int(source_state["num_frames"]) - 1:
        raise ValueError("switch_frame must leave at least one continuation frame")
    source_canonical = _collect_prefix(
        source_predictor,
        source_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    num_frames = int(source_state["num_frames"])
    del source_state, source_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(seed)
    oracle_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    oracle_state = oracle_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    target_canonical, oracle_future = _collect_native(
        oracle_predictor,
        oracle_state,
        mask=mask,
        object_id=object_id,
        switch_frame=switch_frame,
    )
    del oracle_state, oracle_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if translator is None:
        translator = DirectCopyTranslator(target_canonical.spec)
    target_spec = getattr(translator, "target_spec", None)
    if target_spec is not None and target_spec != target_canonical.spec:
        raise ValueError(
            f"translator target spec {target_spec} differs from runtime "
            f"target spec {target_canonical.spec}"
        )
    translated = translator.translate(source_canonical)
    state_alignment = evaluate_state(translated, target_canonical)

    _seed_everything(seed)
    target_predictor = build_sam2_video_predictor(
        config_file=target_config_file,
        ckpt_path=str(target_checkpoint),
        device=device,
    )
    backbone_calls: list[tuple[int, ...]] = []
    original_forward_image = target_predictor.forward_image

    def counted_forward_image(image: torch.Tensor):
        backbone_calls.append(tuple(image.shape))
        return original_forward_image(image)

    target_predictor.forward_image = counted_forward_image
    target_state = init_sam2_inference_state_without_warmup(
        target_predictor,
        video_path=str(video_dir),
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    calls_before_injection = len(backbone_calls)
    injection = inject_sam2_canonical_state(
        translated,
        predictor=target_predictor,
        inference_state=target_state,
    )
    calls_after_injection = len(backbone_calls)
    candidate_future: dict[int, torch.Tensor] = {}
    start_frame = switch_frame + 1
    for frame_idx, _object_ids, masks in target_predictor.propagate_in_video(
        target_state,
        start_frame_idx=start_frame,
        max_frame_num_to_track=num_frames - start_frame,
        reverse=False,
    ):
        candidate_future[int(frame_idx)] = masks.detach().cpu().float()
    downstream = _compare_future_masks(oracle_future, candidate_future)
    report = {
        "source_model_id": source_model_id,
        "target_model_id": target_model_id,
        "translator": translator_name,
        "translator_parameter_count": int(translator.parameter_count()),
        "upstream_commit": commit,
        "video_id": video_dir.name,
        "switch_frame": switch_frame,
        "future_frames": sorted(candidate_future),
        "state_alignment_to_target_native": state_alignment,
        "injection": injection,
        "backbone_calls_before_injection": calls_before_injection,
        "backbone_calls_during_injection": calls_after_injection
        - calls_before_injection,
        "backbone_calls_during_future_continuation": len(backbone_calls)
        - calls_after_injection,
        "mask_comparison_to_target_native": downstream,
        "seed": seed,
        "device": device,
        "resources": _resource_measurement(started_at, device),
    }
    if artifact_dir is not None:
        report["artifacts"] = write_handoff_artifacts(
            video_dir=video_dir,
            annotation_dir=Path(prompt_mask).resolve().parent,
            object_id=object_id,
            oracle_masks=oracle_future,
            candidate_masks=candidate_future,
            output_dir=artifact_dir,
            report=report,
            candidate_label=candidate_label,
        )
    del target_state, target_predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_cross_model_direct_handoff(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible entry point for the Direct Copy handoff baseline."""

    return run_cross_model_translator_handoff(**kwargs)
