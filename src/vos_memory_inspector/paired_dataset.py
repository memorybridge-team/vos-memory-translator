from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import IterableDataset

from .state import MemoryState


PAIR_SCHEMA_VERSION = "sam2-paired-state-v1"


@dataclass
class PairedState:
    pair_id: str
    direction: str
    source: MemoryState
    target: MemoryState

    def validate(self) -> None:
        self.source.validate()
        self.target.validate()
        if self.direction not in {"small_to_large", "large_to_small"}:
            raise ValueError(f"Unsupported direction: {self.direction}")
        comparable = (
            "sequence_id",
            "switch_frame",
            "frame_indices",
            "object_ids",
        )
        mismatches = [
            name
            for name in comparable
            if getattr(self.source, name) != getattr(self.target, name)
        ]
        if mismatches:
            raise ValueError(f"Unpaired source/target state fields: {mismatches}")
        if self.source.prompt != self.target.prompt:
            raise ValueError("Source and target prompt metadata differ")

    def to_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": PAIR_SCHEMA_VERSION,
            "pair_id": self.pair_id,
            "direction": self.direction,
            "source": self.source.to_payload(),
            "target": self.target.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PairedState":
        if payload.get("schema_version") != PAIR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported pair schema: {payload.get('schema_version')}")
        pair = cls(
            pair_id=str(payload["pair_id"]),
            direction=str(payload["direction"]),
            source=MemoryState.from_payload(payload["source"]),
            target=MemoryState.from_payload(payload["target"]),
        )
        pair.validate()
        return pair


class PairShardWriter:
    """Write bounded lists of CPU state pairs instead of accumulating a corpus."""

    def __init__(self, output_dir: str | Path, *, pairs_per_shard: int = 1) -> None:
        if pairs_per_shard <= 0:
            raise ValueError("pairs_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_per_shard = pairs_per_shard
        self._buffer: list[dict[str, Any]] = []
        self._shard_index = 0
        self._manifest: list[dict[str, Any]] = []

    def add(self, pair: PairedState) -> None:
        pair.validate()
        self._buffer.append(pair.to_payload())
        if len(self._buffer) >= self.pairs_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        path = self.output_dir / f"pairs-{self._shard_index:05d}.pt"
        torch.save(self._buffer, path)
        self._manifest.append(
            {
                "path": path.name,
                "pair_ids": [str(item["pair_id"]) for item in self._buffer],
                "sequence_ids": [
                    str(item["source"]["sequence_id"]) for item in self._buffer
                ],
                "num_pairs": len(self._buffer),
                "num_bytes": path.stat().st_size,
            }
        )
        self._buffer = []
        self._shard_index += 1
        self._write_manifest()
        return path

    def close(self) -> None:
        self.flush()

    def _write_manifest(self) -> None:
        payload = {
            "schema_version": PAIR_SCHEMA_VERSION,
            "storage": "torch.save CPU tensors; one bounded list per shard",
            "shards": self._manifest,
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def __enter__(self) -> "PairShardWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()


class PairedStateDataset(IterableDataset[PairedState]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PAIR_SCHEMA_VERSION:
            raise ValueError("Unsupported paired-state manifest")
        self.shards = [self.manifest_path.parent / item["path"] for item in manifest["shards"]]

    def __iter__(self) -> Iterator[PairedState]:
        worker = torch.utils.data.get_worker_info()
        shards = self.shards if worker is None else self.shards[worker.id :: worker.num_workers]
        for shard in shards:
            payloads = torch.load(shard, map_location="cpu", weights_only=False)
            for payload in payloads:
                yield PairedState.from_payload(payload)


def iter_aligned_tensors(
    pair: PairedState, tensor_name: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    pair.validate()
    target_lookup = {
        (frame.object_id, frame.frame_idx, frame.storage_kind): frame
        for frame in pair.target.frames
    }
    for source_frame in pair.source.frames:
        key = (
            source_frame.object_id,
            source_frame.frame_idx,
            source_frame.storage_kind,
        )
        target_frame = target_lookup[key]
        if tensor_name not in source_frame.tensors or tensor_name not in target_frame.tensors:
            continue
        yield (
            source_frame.tensors[tensor_name],
            target_frame.tensors[tensor_name],
            {
                "pair_id": pair.pair_id,
                "sequence_id": pair.source.sequence_id,
                "frame_idx": source_frame.frame_idx,
                "object_id": source_frame.object_id,
                "storage_kind": source_frame.storage_kind,
                "direction": pair.direction,
            },
        )


def validate_video_splits(splits: dict[str, Sequence[str]]) -> None:
    owners: dict[str, str] = {}
    duplicates: list[str] = []
    for split, video_ids in splits.items():
        for video_id in video_ids:
            prior = owners.setdefault(str(video_id), split)
            if prior != split:
                duplicates.append(f"{video_id}:{prior}/{split}")
    if duplicates:
        raise ValueError("Video-level split leakage: " + ", ".join(sorted(duplicates)))


def write_video_splits(path: str | Path, splits: dict[str, Sequence[str]]) -> None:
    validate_video_splits(splits)
    Path(path).write_text(
        json.dumps({name: list(ids) for name, ids in splits.items()}, indent=2) + "\n",
        encoding="utf-8",
    )

