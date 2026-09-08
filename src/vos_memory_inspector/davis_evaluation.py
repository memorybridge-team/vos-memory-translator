"""Evaluate a partial DAVIS continuation with the official metric functions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np
from PIL import Image


Metric = Callable[[np.ndarray, np.ndarray], float]


def load_official_davis_metrics(
    evaluation_repo: str | Path,
) -> tuple[Metric, Metric, str]:
    repo = Path(evaluation_repo).resolve()
    metrics_file = repo / "davis2017" / "metrics.py"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Official DAVIS metrics file not found: {metrics_file}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from davis2017.metrics import db_eval_boundary, db_eval_iou

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return db_eval_iou, db_eval_boundary, commit


def _prediction_files(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.glob("*.png")):
        stem = path.stem
        if stem.startswith("frame_"):
            stem = stem.removeprefix("frame_")
        try:
            frame = int(stem)
        except ValueError:
            continue
        result[frame] = path
    if not result:
        raise ValueError(f"No frame PNG predictions found in {directory}")
    return result


def evaluate_davis_future_masks(
    *,
    prediction_directory: str | Path,
    annotation_directory: str | Path,
    object_id: int,
    start_frame: int,
    iou_metric: Metric,
    boundary_metric: Metric,
    metric_source: str,
    sequence: str,
) -> dict[str, Any]:
    prediction_directory = Path(prediction_directory).resolve()
    annotation_directory = Path(annotation_directory).resolve()
    predictions = _prediction_files(prediction_directory)
    frames = sorted(frame for frame in predictions if frame >= start_frame)
    if not frames:
        raise ValueError(f"No predictions at or after frame {start_frame}")
    expected = list(range(start_frame, frames[-1] + 1))
    if frames != expected:
        missing = sorted(set(expected) - set(frames))
        raise ValueError(f"Prediction frame sequence is not contiguous; missing={missing}")

    rows: list[dict[str, float | int]] = []
    for frame in frames:
        annotation_path = annotation_directory / f"{frame:05d}.png"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation not found: {annotation_path}")
        prediction = np.asarray(Image.open(predictions[frame])) > 0
        annotation_labels = np.asarray(Image.open(annotation_path))
        annotation = annotation_labels == object_id
        if prediction.shape != annotation.shape:
            raise ValueError(
                f"Frame {frame} shape mismatch: prediction={prediction.shape}, "
                f"annotation={annotation.shape}"
            )
        j = float(iou_metric(annotation, prediction))
        f = float(boundary_metric(annotation, prediction))
        rows.append({"frame": frame, "J": j, "F": f, "J_and_F": (j + f) / 2})

    return {
        "schema_version": "cmmt.davis_future_evaluation.v1",
        "scope": "partial_sequence_after_switch",
        "warning": (
            "This is a single-object, partial-sequence score from start_frame onward; "
            "it is not the full DAVIS benchmark score."
        ),
        "metric_source": metric_source,
        "sequence": sequence,
        "object_id": object_id,
        "start_frame": start_frame,
        "end_frame": frames[-1],
        "evaluated_frames": len(rows),
        "mean_J": mean(float(row["J"]) for row in rows),
        "mean_F": mean(float(row["F"]) for row in rows),
        "mean_J_and_F": mean(float(row["J_and_F"]) for row in rows),
        "frames": rows,
    }


def write_davis_future_report(report: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "\n".join(
            [
                "# Partial DAVIS continuation evaluation",
                "",
                f"> {report['warning']}",
                "",
                f"- Sequence: `{report['sequence']}`",
                f"- Object id: {report['object_id']}",
                f"- Frames: {report['start_frame']}–{report['end_frame']} "
                f"({report['evaluated_frames']} frames)",
                f"- Mean J: {report['mean_J']:.6f}",
                f"- Mean F: {report['mean_F']:.6f}",
                f"- Mean J&F: {report['mean_J_and_F']:.6f}",
                f"- Metric source: `{report['metric_source']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
