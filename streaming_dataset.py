from __future__ import annotations

import json
import math
import multiprocessing
import os
import sqlite3
import zipfile
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from paddle.io import Dataset, Sampler
from ppocr.data.imaug import create_operators, transform

from ocr_training.aihub import stable_number, valid_annotation_indices
from ocr_training.dataset import crop_annotation, make_variant, normalize_label
from ocr_training.streaming_index import sample_location

MAX_SAMPLE_ATTEMPTS = 32


def _load_npy(path: Path) -> np.ndarray[Any, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Streaming index array not found: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


class AIHubStreamDataSet(Dataset):
    """PaddleOCR dataset that crops AI-Hub ZIP pages in worker memory."""

    def __init__(
        self,
        config: dict[str, Any],
        mode: str,
        logger: Any,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        dataset_config = config[mode]["dataset"]
        split = str(dataset_config.get("split", mode.lower()))
        if split not in {"train", "dev"}:
            raise ValueError(f"Unsupported streaming dataset split: {split}")

        self.logger = logger
        self.mode = mode.lower()
        self.split = split
        self.index_dir = Path(dataset_config["index_dir"])
        self.data_root = Path(dataset_config["data_root"])
        self.padding_ratio = float(dataset_config.get("padding_ratio", 0.06))
        self.max_text_length = int(
            dataset_config.get("max_text_length", config["Global"]["max_text_length"])
        )
        self.cache_bytes = max(0, int(dataset_config.get("cache_mb", 128))) * 1024 * 1024
        self.seed = seed
        self.need_reset = False
        self.ds_width = False
        self.document_ids = _load_npy(self.index_dir / f"{split}_documents.npy")
        self.sample_ends = _load_npy(self.index_dir / f"{split}_sample_ends.npy")
        self.data_idx_order_list = range(len(self))
        self.ops = create_operators(dataset_config["transforms"], config["Global"])
        self._shared_epoch = multiprocessing.Value("q", int(seed or 0))
        self._process_id: int | None = None
        self._connection: sqlite3.Connection | None = None
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._cache: OrderedDict[
            int,
            tuple[np.ndarray[Any, Any], dict[str, Any], list[int], tuple[Any, ...], int],
        ] = OrderedDict()
        self._cache_size = 0

    def __len__(self) -> int:
        return int(self.sample_ends[-1]) if len(self.sample_ends) else 0

    def reset_data_lines(self, seed: int | None = None, epoch: int | None = None) -> None:
        self.seed = seed
        self._shared_epoch.value = int(epoch if epoch is not None else seed or 0)

    def _ensure_worker_resources(self) -> None:
        process_id = os.getpid()
        if self._process_id == process_id and self._connection is not None:
            return
        self._close_worker_resources()
        database = (self.index_dir / "documents.sqlite3").resolve()
        self._connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        self._process_id = process_id

    def _close_worker_resources(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._process_id = None
        self._cache.clear()
        self._cache_size = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_archives"] = {}
        state["_cache"] = OrderedDict()
        state["_cache_size"] = 0
        state["_process_id"] = None
        return state

    def _archive(self, relative_path: str) -> zipfile.ZipFile:
        path = (self.data_root / relative_path).resolve()
        archive = self._archives.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path)
            self._archives[path] = archive
        return archive

    def _load_document(
        self, document_key: int
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any], list[int], tuple[Any, ...]]:
        self._ensure_worker_resources()
        cached = self._cache.get(document_key)
        if cached is not None:
            self._cache.move_to_end(document_key)
            return cached[0], cached[1], cached[2], cached[3]
        assert self._connection is not None
        row = self._connection.execute(
            """
            SELECT document_id, image_zip, image_member, label_zip, label_member
            FROM documents WHERE document_key = ?
            """,
            (document_key,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Document not found in streaming index: {document_key}")
        image_payload = self._archive(str(row[1])).read(str(row[2]))
        image = cv2.imdecode(np.frombuffer(image_payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode AI-Hub image: {row[1]}::{row[2]}")
        label_payload = self._archive(str(row[3])).read(str(row[4]))
        label = json.loads(label_payload)
        if not isinstance(label, dict):
            raise ValueError(f"AI-Hub label root is not an object: {row[3]}::{row[4]}")
        valid_indices = valid_annotation_indices(
            label.get("annotations"), self.max_text_length
        )

        cache_size = int(image.nbytes + len(label_payload))
        if self.cache_bytes and cache_size <= self.cache_bytes:
            while self._cache and self._cache_size + cache_size > self.cache_bytes:
                _, (_, _, _, _, removed_size) = self._cache.popitem(last=False)
                self._cache_size -= removed_size
            self._cache[document_key] = (image, label, valid_indices, row, cache_size)
            self._cache_size += cache_size
        return image, label, valid_indices, row

    def _one_sample(self, sample_index: int) -> Any:
        document_key, local_position = sample_location(
            self.document_ids, self.sample_ends, sample_index
        )
        image, label, valid_indices, row = self._load_document(document_key)
        annotation_index = int(valid_indices[local_position])
        annotations = label.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("annotations is not a list")
        annotation = annotations[annotation_index]
        if not isinstance(annotation, dict):
            raise ValueError("annotation is not an object")

        text = normalize_label(annotation.get("annotation.text"))
        crop = crop_annotation(image, annotation.get("annotation.bbox", []), self.padding_ratio)
        if not text or crop is None:
            raise ValueError(
                "indexed annotation is no longer a valid crop: "
                f"document={row[0]}, annotation={annotation_index}"
            )

        document_id = str(row[0])
        annotation_id = str(annotation.get("id", annotation_index))
        if self.mode == "train":
            variant = cast(
                Any,
                "camera"
                if stable_number(f"variant:{document_id}:{annotation_id}") % 100 < 45
                else "clean",
            )
            variant_seed = stable_number(
                f"{document_id}:{annotation_id}:{variant}:{self._shared_epoch.value}"
            )
            crop = make_variant(crop, variant, variant_seed)

        return transform({"image": crop, "label": text}, self.ops)

    def _replacement_index(
        self,
        original_index: int,
        attempt: int,
        attempted: set[int],
    ) -> int:
        total = len(self)
        if total <= 0:
            raise IndexError("streaming dataset is empty")

        candidate = stable_number(
            f"stream-retry:{self.split}:{original_index}:{attempt}"
        ) % total
        while candidate in attempted and len(attempted) < total:
            candidate = (candidate + 1) % total
        return candidate

    def __getitem__(self, sample_index: int) -> Any:
        original = int(sample_index)
        current = original
        attempted: set[int] = set()
        failures: list[str] = []

        for attempt in range(MAX_SAMPLE_ATTEMPTS):
            attempted.add(current)
            try:
                result = self._one_sample(current)
                if result is not None:
                    # Recoverable invalid samples are expected in this corpus.
                    # Keep the hot path silent; only a full retry exhaustion
                    # raises below with representative failure examples.
                    return result
                failure = f"{current}: transform returned None"
            except (
                IndexError,
                KeyError,
                OSError,
                ValueError,
                zipfile.BadZipFile,
                cv2.error,
            ) as error:
                failure = f"{current}: {type(error).__name__}: {error}"

            failures.append(failure)
            current = self._replacement_index(
                original,
                attempt + 1,
                attempted,
            )

        examples = " | ".join(failures[:5])
        raise RuntimeError(
            "Could not load a valid streaming sample after "
            f"{MAX_SAMPLE_ATTEMPTS} dispersed attempts; "
            f"original={original}; examples={examples}"
        )

    def __del__(self) -> None:
        with suppress(Exception):
            self._close_worker_resources()


class DocumentBatchSampler(Sampler):
    """Shuffle documents deterministically while keeping their crops adjacent."""

    def __init__(
        self,
        data_source: AIHubStreamDataSet,
        batch_size: int,
        drop_last: bool = True,
        seed: int = 2026,
        resume_epoch: int = 0,
        resume_batch: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.streaming_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.resume_epoch = int(resume_epoch)
        self.resume_batch = int(resume_batch)
        self.batch_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Any:
        import paddle.distributed as dist

        document_order = np.arange(
            len(self.streaming_source.document_ids), dtype=np.int64
        )
        np.random.default_rng(self.seed + self.epoch).shuffle(document_order)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        rank_order = document_order[rank::world_size]
        self.batch_offset = self.resume_batch if self.epoch == self.resume_epoch else 0
        batch_number = 0
        batch: list[int] = []
        for document_position in rank_order:
            position = int(document_position)
            start = (
                0
                if position == 0
                else int(self.streaming_source.sample_ends[position - 1])
            )
            end = int(self.streaming_source.sample_ends[position])
            for sample_index in range(start, end):
                batch.append(sample_index)
                if len(batch) == self.batch_size:
                    if batch_number >= self.batch_offset:
                        yield batch
                    batch_number += 1
                    batch = []
        if batch and not self.drop_last and batch_number >= self.batch_offset:
            yield batch

    def __len__(self) -> int:
        import paddle.distributed as dist

        samples = len(self.streaming_source) // max(1, dist.get_world_size())
        batches = samples // self.batch_size if self.drop_last else math.ceil(
            samples / self.batch_size
        )
        offset = self.resume_batch if self.epoch == self.resume_epoch else 0
        return max(0, batches - offset)
