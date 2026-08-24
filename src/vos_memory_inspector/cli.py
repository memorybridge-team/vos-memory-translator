from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compatibility import compare_manifests, write_compatibility_report
from .davis import download_davis_2017_trainval_480p, validate_davis_sequence
from .runner import run_video_probe


def _probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe stored and consumed SAM 2 temporal-memory tensors."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument(
        "--dump-tensor",
        action="append",
        default=[],
        help=(
            "Opt-in tensor name to save on CPU; repeat for multiple names. "
            "Without this option only statistics are written."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--allow-upstream-mismatch", action="store_true")
    return parser


def probe_main(argv: list[str] | None = None) -> None:
    args = _probe_parser().parse_args(argv)
    summary = run_video_probe(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        jsonl_path=args.jsonl,
        csv_path=args.csv,
        dump_dir=args.dump_dir,
        dump_tensors=tuple(args.dump_tensor),
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        allow_upstream_mismatch=args.allow_upstream_mismatch,
    )
    print(json.dumps(summary, indent=2))


def compare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare matching rows from sequential Tiny and Large manifests."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    report = compare_manifests(args.source, args.target)
    write_compatibility_report(report, args.json, args.markdown)
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))


def davis_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate a DAVIS 2017 sequence and resolve probe inputs."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--resolution", default="480p")
    args = parser.parse_args(argv)
    sequence = validate_davis_sequence(
        args.root, args.sequence, resolution=args.resolution
    )
    print(
        json.dumps(
            {
                "name": sequence.name,
                "frames_directory": str(sequence.frames_directory),
                "first_mask": str(sequence.first_mask),
                "frame_count": sequence.frame_count,
                "resolution": sequence.resolution,
            },
            indent=2,
        )
    )


def davis_download_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and safely extract official DAVIS 2017 trainval 480p."
    )
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--accept-dataset-terms", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args(argv)
    root = download_davis_2017_trainval_480p(
        args.destination,
        accept_dataset_terms=args.accept_dataset_terms,
        keep_archive=args.keep_archive,
    )
    print(json.dumps({"davis_root": str(root)}, indent=2))
