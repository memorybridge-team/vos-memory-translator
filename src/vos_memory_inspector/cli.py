from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compatibility import compare_manifests, write_compatibility_report
from .davis import download_davis_2017_trainval_480p, validate_davis_sequence
from .paired_experiment import (
    load_canonical_state,
    run_paired_experiment,
    run_synthetic_experiment,
)
from .runner import run_video_probe
from .roundtrip import (
    run_cross_model_direct_handoff,
    run_cross_model_translator_handoff,
    run_same_checkpoint_roundtrip,
)
from .state_inspector import inspect_state, write_inspection_report
from .translators import RidgeDirectPresenceTranslator, RidgeStateTranslator


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--canonical-state",
        type=Path,
        help="Opt-in .pt export of the continuation-oriented canonical state.",
    )
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
        seed=args.seed,
        canonical_state_path=args.canonical_state,
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
    parser.add_argument(
        "--split",
        choices=("train", "val", "none"),
        default="val",
        help="Dataset split membership to validate. Use 'none' to skip membership checking.",
    )
    args = parser.parse_args(argv)
    sequence = validate_davis_sequence(
        args.root,
        args.sequence,
        resolution=args.resolution,
        split=None if args.split == "none" else args.split,
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


def state_inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inspect tensors in a nested .pt state, HF cache, or canonical state."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--key", help="Optional top-level mapping key to inspect")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    # State files are pickle-backed. Only load files from a trusted source.
    value = __import__("torch").load(args.input, map_location="cpu", weights_only=False)
    if args.key is not None:
        value = value[args.key]
    report = inspect_state(value)
    write_inspection_report(report, args.json, args.markdown)
    print(json.dumps(report.to_dict(), indent=2))


def synthetic_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an explicitly synthetic paired-state translator smoke test."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_synthetic_experiment(
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(report, indent=2))


def paired_experiment_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit translators from paired canonical states and evaluate held-out pairs."
    )
    parser.add_argument("--train-source", action="append", type=Path, default=[])
    parser.add_argument("--train-target", action="append", type=Path, default=[])
    parser.add_argument("--test-source", action="append", type=Path, default=[])
    parser.add_argument("--test-target", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--ridge-lambda", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--translator",
        action="append",
        choices=("direct", "ridge", "linear", "residual_mlp"),
        default=[],
        help="Translator to run; repeat as needed. Default: all.",
    )
    args = parser.parse_args(argv)
    if len(args.train_source) != len(args.train_target) or not args.train_source:
        parser.error("provide the same non-zero number of --train-source/--train-target")
    if len(args.test_source) != len(args.test_target) or not args.test_source:
        parser.error("provide the same non-zero number of --test-source/--test-target")
    train_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.train_source, args.train_target, strict=True)
    ]
    test_pairs = [
        (load_canonical_state(source), load_canonical_state(target))
        for source, target in zip(args.test_source, args.test_target, strict=True)
    ]
    report = run_paired_experiment(
        train_pairs,
        test_pairs,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge_lambda=args.ridge_lambda,
        hidden_dim=args.hidden_dim,
        translator_names=tuple(args.translator) or None,
    )
    print(json.dumps(report, indent=2))


def roundtrip_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run same-checkpoint SAM 2 export/inject continuation validation."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run_same_checkpoint_roundtrip(
        sam2_repo=args.sam2_repo,
        config_file=args.config,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def direct_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a checkpoint-backed cross-model Direct Copy handoff."
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Write browsable mask PNGs plus report.json/report.md.",
    )
    args = parser.parse_args(argv)
    report = run_cross_model_direct_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def ridge_handoff_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint-backed cross-model handoff using a saved Ridge model."
        )
    )
    parser.add_argument("--sam2-repo", required=True, type=Path)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--target-model-id", required=True)
    parser.add_argument("--translator-artifact", required=True, type=Path)
    parser.add_argument(
        "--presence-policy",
        choices=("direct", "ridge"),
        default="direct",
        help="Use source presence logits directly or apply the fitted Ridge head.",
    )
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--prompt-mask", required=True, type=Path)
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--switch-frame", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-video-on-device", action="store_true")
    parser.add_argument("--keep-state-on-device", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)

    payload = __import__("torch").load(
        args.translator_artifact, map_location="cpu", weights_only=True
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("ridge"), dict):
        parser.error("translator artifact does not contain a Ridge payload")
    ridge = RidgeStateTranslator.from_payload(payload["ridge"])
    if args.presence_policy == "direct":
        translator = RidgeDirectPresenceTranslator(ridge)
        translator_name = translator.name
        candidate_label = "Ridge memory/pointer + Direct presence"
    else:
        translator = ridge
        translator_name = ridge.name
        candidate_label = "Ridge"
    report = run_cross_model_translator_handoff(
        sam2_repo=args.sam2_repo,
        source_config_file=args.source_config,
        source_checkpoint=args.source_checkpoint,
        source_model_id=args.source_model_id,
        target_config_file=args.target_config,
        target_checkpoint=args.target_checkpoint,
        target_model_id=args.target_model_id,
        video_dir=args.video_dir,
        prompt_mask=args.prompt_mask,
        object_id=args.object_id,
        switch_frame=args.switch_frame,
        device=args.device,
        offload_video_to_cpu=not args.keep_video_on_device,
        offload_state_to_cpu=not args.keep_state_on_device,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        translator=translator,
        translator_name=translator_name,
        candidate_label=candidate_label,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
