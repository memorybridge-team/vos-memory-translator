#!/usr/bin/env python
"""Run Tiny then Large with CPU offload, then compare their manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vos_memory_inspector.compatibility import (
    compare_manifests,
    write_compatibility_report,
)
from vos_memory_inspector.runner import run_video_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--tiny-checkpoint", required=True, type=Path)
    parser.add_argument("--large-checkpoint", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = (
        (
            "sam2.1-hiera-tiny",
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            args.tiny_checkpoint,
            args.output_dir / "tiny.jsonl",
            args.output_dir / "tiny.csv",
            args.output_dir / "tiny_state.pt",
        ),
        (
            "sam2.1-hiera-large",
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            args.large_checkpoint,
            args.output_dir / "large.jsonl",
            args.output_dir / "large.csv",
            args.output_dir / "large_state.pt",
        ),
    )
    summaries = []
    for model_id, config, checkpoint, jsonl, csv, canonical_state in runs:
        summaries.append(
            run_video_probe(
                sam2_repo=args.sam2_repo,
                config_file=config,
                checkpoint=checkpoint,
                model_id=model_id,
                video_dir=args.video_dir,
                prompt_mask=args.prompt_mask,
                object_id=args.object_id,
                switch_frame=args.switch_frame,
                jsonl_path=jsonl,
                csv_path=csv,
                device=args.device,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                seed=args.seed,
                canonical_state_path=canonical_state,
            )
        )
    report = compare_manifests(
        args.output_dir / "tiny.jsonl", args.output_dir / "large.jsonl"
    )
    write_compatibility_report(
        report,
        args.output_dir / "compatibility.json",
        args.output_dir / "compatibility.md",
    )
    print(json.dumps({"runs": summaries, "summary": report["disposition_counts"]}, indent=2))


if __name__ == "__main__":
    main()

