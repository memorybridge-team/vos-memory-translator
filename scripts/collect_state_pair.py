from __future__ import annotations

import argparse
import json
from pathlib import Path

from vos_memory_inspector.collection import ModelRunSpec, collect_sam2_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one sequential Small/Large SAM 2 paired memory shard"
    )
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--direction", choices=("small_to_large", "large_to_small"), required=True
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--prompt-mask", type=Path, required=True)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = collect_sam2_pair(
        sam2_repo=args.sam2_repo,
        source=ModelRunSpec(
            args.source_model_id, args.source_config, args.source_checkpoint
        ),
        target=ModelRunSpec(
            args.target_model_id, args.target_config, args.target_checkpoint
        ),
        direction=args.direction,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        output_dir=args.output_dir,
        device=args.device,
        allow_upstream_mismatch=args.allow_upstream_mismatch,
    )
    print(
        json.dumps(
            {
                "pair_id": pair.pair_id,
                "direction": pair.direction,
                "source_model": pair.source.model_id,
                "target_model": pair.target.model_id,
                "frames": pair.source.frame_indices,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

