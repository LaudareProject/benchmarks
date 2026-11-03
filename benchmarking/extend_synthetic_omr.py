"""Create or extend synthetic OMR data with unique Gregorian chant annotations."""

import argparse
import atexit
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
import random
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
from .annotations.ann_handler import create_all_pagexml_files
from .annotations.data_synthesis import generate_single_sample_batch, update_json_file
from .annotations.gregobase import generate_batch
from .annotations.music_offsets import iter_music_offsets
from .extend_synthetic_ocr import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LINES_PER_IMAGE,
    MAX_WORKERS,
    _LazyTexturePool,
    _initialize_splits,
    _texture_pool,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "pretrain_data" / "omr"
TEXTURE_CACHE_SIZE = 16
RENDER_WIDTH = 1200
RENDER_HEIGHT = 600

_SHARED_TEXTURES: _LazyTexturePool | None = None
_WORKER_TEMP: tempfile.TemporaryDirectory | None = None


def _initialize_render_worker(seed: int, texture_paths: list[str]) -> None:
    global _SHARED_TEXTURES, _WORKER_TEMP
    worker_ids = multiprocessing.current_process()._identity
    worker_index = worker_ids[0] if worker_ids else 0
    worker_seed = seed + worker_index
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    from .annotations import data_synthesis, gregobase

    gregobase.DEFAULT_CONFIG["font_path"] = str(
        Path(gregobase.DEFAULT_CONFIG["font_path"]).resolve()
    )
    _WORKER_TEMP = tempfile.TemporaryDirectory(prefix="omr_worker_")
    atexit.register(_WORKER_TEMP.cleanup)
    os.chdir(_WORKER_TEMP.name)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass
    data_synthesis.rg.seed(worker_seed)
    _SHARED_TEXTURES = _LazyTexturePool(texture_paths, TEXTURE_CACHE_SIZE)

def _render_music_batch(
    candidates: list[dict], first_image_id: int, output_dir: Path
) -> list[dict]:
    """Render a chant batch once, then composite successful scores into pages."""
    if _SHARED_TEXTURES is None:
        raise RuntimeError("Shared textures were not initialized in the OMR worker")

    rendered = generate_batch(
        [candidate["gabc"] for candidate in candidates],
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        dpi=300,
        debug=False,
    )
    if len(rendered) != len(candidates):
        raise RuntimeError(
            f"Gregobase returned {len(rendered)} scores for {len(candidates)} candidates"
        )

    valid = []
    missed = []
    for candidate, image in zip(candidates, rendered):
        if image is None:
            missed.append(candidate["source_id"])
        else:
            valid.append((image, candidate["gabc"], candidate["label"], candidate))
    if missed:
        print(
            f"WARNING: Gregobase did not render {len(missed)} chant candidate(s): "
            + ", ".join(missed),
            flush=True,
        )

    pages = []
    for offset in range(0, len(valid), LINES_PER_IMAGE):
        entries = valid[offset : offset + LINES_PER_IMAGE]
        image_id = first_image_id + len(pages)
        result = generate_single_sample_batch(
            image_id,
            [(image, gabc, label) for image, gabc, label, _ in entries],
            "music",
            [],
            None,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            output_dir,
            _SHARED_TEXTURES,
            LINES_PER_IMAGE,
            debug=False,
        )
        if result is None:
            print(
                f"WARNING: rejecting OMR image {image_id}: compositor returned no page "
                f"for {len(entries)} chants",
                flush=True,
            )
            (Path(output_dir) / f"music{image_id + 1:04d}.jpg").unlink(
                missing_ok=True
            )
            continue
        image_record, annotations = result
        if len(annotations) != len(entries):
            print(
                f"WARNING: rejecting OMR image {image_id}: compositor retained "
                f"{len(annotations)} of {len(entries)} chants",
                flush=True,
            )
            (Path(output_dir) / image_record["file_name"]).unlink(missing_ok=True)
            continue
        accepted_annotations = []
        provenance = []
        invalid_box = False
        width, height = image_record["width"], image_record["height"]
        for annotation, entry in zip(annotations, entries):
            bbox = annotation.get("bbox", [])
            if (
                len(bbox) != 4
                or bbox[2] <= 0
                or bbox[3] <= 0
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[0] + bbox[2] > width
                or bbox[1] + bbox[3] > height
            ):
                print(
                    f"WARNING: discarding invalid OMR box {bbox!r}; "
                    f"rejecting image {image_id}",
                    flush=True,
                )
                invalid_box = True
                break
            source = entry[3]
            annotation.update(
                {
                    "source_id": source["source_id"],
                    "section": source["section"],
                    "start": source["start"],
                    "length": source["length"],
                }
            )
            accepted_annotations.append(annotation)
            provenance.append(source)
        if invalid_box:
            (Path(output_dir) / image_record["file_name"]).unlink(missing_ok=True)
            continue
        if not accepted_annotations:
            (Path(output_dir) / image_record["file_name"]).unlink(missing_ok=True)
            continue
        image_record["source_ids"] = list(
            dict.fromkeys(item["source_id"] for item in provenance)
        )
        image_record["source_spans"] = [
            {
                "source_id": item["source_id"],
                "start": item["start"],
                "length": item["length"],
                "section": item["section"],
            }
            for item in provenance
        ]
        pages.append({"image": image_record, "annotations": accepted_annotations})
    return pages


def _load_split(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _line_key(label: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", label).split()).casefold()

def _write_pagexml(data_dir: Path, split_paths: dict[str, Path]) -> None:
    processed_dir = data_dir / "annotations-diplomatic" / "processed_splits"
    for path in split_paths.values():
        create_all_pagexml_files(
            processed_dir,
            _load_split(path),
            "diplomatic",
            image_base_path=data_dir,
            target_tasks="omr",
        )


def extend_dataset(
    data_dir: Path,
    target_lines: int = 900_000,
    workers: int = MAX_WORKERS,
    flush_images: int = 20,
    batch_size: int = 30,
    seed: int = 2025,
    debug: bool = False,
) -> int:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if target_lines < 0:
        raise ValueError("target-lines must be non-negative")
    if flush_images < 1 or batch_size < 1:
        raise ValueError("flush-images and batch-size must be positive")
    data_dir = data_dir.resolve()

    split_paths, splits = _initialize_splits(data_dir, "omr")
    existing_annotations = [
        annotation
        for split in splits.values()
        for annotation in split.get("annotations", [])
    ]
    line_count = len(existing_annotations)
    if target_lines < line_count:
        raise ValueError(
            f"Target {target_lines:,} is below the existing {line_count:,} OMR lines"
        )
    if debug:
        target_lines = min(target_lines, line_count + 50)
        workers = 1
    if target_lines == line_count:
        print(f"Already at target: {line_count:,} OMR lines")
        _write_pagexml(data_dir, split_paths)
        return line_count

    seen = {
        _line_key(annotation.get("description", ""))
        for annotation in existing_annotations
    }
    image_ids = [
        int(image["id"])
        for split in splits.values()
        for image in split.get("images", [])
    ]
    next_image_id = max(image_ids, default=-1) + 1
    for split in splits.values():
        if not split.get("categories"):
            split["categories"] = [
                {"id": 5, "name": "music_line", "supercategory": "recognition"}
            ]

    pending = {name: {"images": [], "annotations": []} for name in split_paths}
    pending_images = 0
    added_images = 0
    added_lines = 0
    reserved_lines = 0
    source = iter_music_offsets(seen)
    futures = deque()
    buffered_candidates = deque()
    exhausted = False
    started = time.monotonic()
    def flush() -> None:
        nonlocal pending_images
        for name, records in pending.items():
            if records["images"]:
                update_json_file(split_paths[name], records)
                records["images"].clear()
                records["annotations"].clear()
        pending_images = 0
        print(
            f"Checkpoint saved: {line_count:,}/{target_lines:,} OMR lines "
            f"({added_lines:,} new annotations)",
            flush=True,
        )


    texture_directory = tempfile.TemporaryDirectory(prefix="synthetic_omr_textures_")
    try:
        # Share on-disk textures; workers load only a small LRU subset instead of
        # serializing a hundred full-resolution PIL images into each process.
        texture_paths = []
        textures = _texture_pool(seed, debug=debug)
        for index, texture in enumerate(textures[:TEXTURE_CACHE_SIZE]):
            texture_path = Path(texture_directory.name) / f"texture_{index:03}.png"
            texture.save(texture_path, format="PNG")
            texture_paths.append(str(texture_path))
        for texture in textures:
            texture.close()
        del textures
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"

        print(f"Rendering OMR with {workers} worker processes (maximum {MAX_WORKERS})")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_render_worker,
            initargs=(seed, texture_paths),
        ) as executor:
            while line_count < target_lines or futures:
                while not exhausted and len(futures) < workers:
                    available = target_lines - line_count - reserved_lines
                    if available <= 0:
                        break
                    candidates = []
                    batch_split = None
                    while len(candidates) < min(batch_size, available):
                        try:
                            candidate = buffered_candidates.popleft() if buffered_candidates else next(source)
                        except StopIteration:
                            exhausted = True
                            break
                        if not isinstance(candidate, dict) or not {
                            "gabc", "label", "source_id", "start", "length", "split"
                        }.issubset(candidate):
                            raise RuntimeError(f"Invalid music-offset candidate: {candidate!r}")
                        if candidate["split"] not in split_paths:
                            raise RuntimeError(
                                f"Music-offset source returned invalid split: {candidate['split']!r}"
                            )
                        if batch_split is None:
                            batch_split = candidate["split"]
                        elif candidate["split"] != batch_split:
                            buffered_candidates.appendleft(candidate)
                            break
                        key = _line_key(candidate["label"])
                        if not key or key in seen:
                            raise RuntimeError(
                                "Music-offset source returned an empty or duplicate normalized label"
                            )
                        seen.add(key)
                        candidates.append(candidate)
                    if not candidates:
                        break
                    # Each worker emits its pages consecutively; allocate enough IDs
                    # up front to avoid collisions between in-flight render jobs.
                    image_slots = (len(candidates) + LINES_PER_IMAGE - 1) // LINES_PER_IMAGE
                    first_id = next_image_id
                    next_image_id += image_slots
                    reserved_lines += len(candidates)
                    futures.append(
                        (
                            executor.submit(
                                _render_music_batch, candidates, first_id, data_dir
                            ),
                            candidates,
                        )
                    )

                if not futures:
                    if exhausted or line_count >= target_lines:
                        break
                    continue

                future, candidates = futures.popleft()
                reserved_lines -= len(candidates)
                result = future.result()
                completed = []
                for record in result:
                    image_record = record["image"]
                    annotations = record["annotations"]
                    image_record["split"] = candidates[0]["split"]
                    completed.append((image_record, annotations))
                for image_record, annotations in completed:
                    split_name = image_record["split"]
                    pending[split_name]["images"].append(image_record)
                    pending[split_name]["annotations"].extend(annotations)
                    added_images += 1
                    pending_images += 1
                    added_lines += len(annotations)
                    line_count += len(annotations)
                if line_count > target_lines:
                    raise RuntimeError(
                        f"Generated {line_count:,} OMR lines, exceeding target {target_lines:,}"
                    )
                if result:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"Generated {added_images:,} images; total OMR lines "
                        f"{line_count:,}/{target_lines:,}; "
                        f"{added_lines / elapsed:,.1f} lines/s",
                        flush=True,
                    )
                if pending_images >= flush_images:
                    flush()
    finally:
        source.close()
        flush()
        texture_directory.cleanup()

    _write_pagexml(data_dir, split_paths)
    if line_count < target_lines:
        raise RuntimeError(
            f"Only {line_count:,} of {target_lines:,} requested OMR lines are available "
            "from the current music-offset source and renderable candidates."
        )
    print(f"Reached {line_count:,} OMR lines; added {added_lines:,} in {added_images:,} images.")
    return line_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or extend synthetic OMR data with unique Gregorian chants."
    )
    parser.add_argument(
        "--target-lines", type=int, default=900_000,
        help="Total line annotations across the training and validation splits.",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--flush-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--debug", action="store_true",
        help="Add at most 50 lines with one worker and a smaller texture pool.",
    )
    args = parser.parse_args()
    extend_dataset(
        args.data_dir,
        args.target_lines,
        args.workers,
        args.flush_images,
        args.batch_size,
        args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
