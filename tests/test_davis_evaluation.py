from __future__ import annotations

import numpy as np
from PIL import Image

from vos_memory_inspector.davis_evaluation import evaluate_davis_future_masks


def _iou(annotation: np.ndarray, prediction: np.ndarray) -> float:
    union = (annotation | prediction).sum()
    return 1.0 if union == 0 else float((annotation & prediction).sum() / union)


def _same_as_iou(annotation: np.ndarray, prediction: np.ndarray) -> float:
    return _iou(annotation, prediction)


def test_partial_davis_evaluation_uses_object_label_and_start_frame(tmp_path) -> None:
    predictions = tmp_path / "predictions"
    annotations = tmp_path / "annotations"
    predictions.mkdir()
    annotations.mkdir()
    for frame in range(1, 3):
        prediction = np.zeros((4, 4), dtype=np.uint8)
        prediction[1:3, 1:3] = 255
        annotation = np.zeros((4, 4), dtype=np.uint8)
        annotation[1:3, 1:3] = 2
        Image.fromarray(prediction).save(predictions / f"frame_{frame:05d}.png")
        Image.fromarray(annotation).save(annotations / f"{frame:05d}.png")

    report = evaluate_davis_future_masks(
        prediction_directory=predictions,
        annotation_directory=annotations,
        object_id=2,
        start_frame=1,
        iou_metric=_iou,
        boundary_metric=_same_as_iou,
        metric_source="test",
        sequence="demo",
    )

    assert report["scope"] == "partial_sequence_after_switch"
    assert report["evaluated_frames"] == 2
    assert report["mean_J_and_F"] == 1.0

    truncated = evaluate_davis_future_masks(
        prediction_directory=predictions,
        annotation_directory=annotations,
        object_id=2,
        start_frame=1,
        end_frame=1,
        iou_metric=_iou,
        boundary_metric=_same_as_iou,
        metric_source="test",
        sequence="demo",
    )
    assert truncated["end_frame"] == 1
    assert truncated["evaluated_frames"] == 1
