"""Extend synthetic OCR data without reusing previously stored text lines."""

import argparse
from collections import OrderedDict, deque
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
import tempfile
import time
import unicodedata
from pathlib import Path

import augraphy as au
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .annotations.ann_handler import create_all_pagexml_files
from .annotations.unique_text import iter_unique_text
from .annotations.data_synthesis import generate_single_sample_batch, get_fonts
from .annotations.data_synthesis import update_json_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "pretrain_data" / "ocr"
DEFAULT_FONT_DIR = PROJECT_ROOT / "fonts"
DEFAULT_CAPITALS_DIR = PROJECT_ROOT / "fonts" / "capitals"
LINES_PER_IMAGE = 15
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1600

MAX_WORKERS = 10


class _LazyTexturePool:
    def __init__(self, paths: list[str], cache_size: int = 16):
        self._paths = paths
        self._cache_size = cache_size
        self._cache: OrderedDict[int, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Image.Image:
        image = self._cache.pop(index, None)
        if image is None:
            with Image.open(self._paths[index]) as source:
                image = source.convert("RGB")
        self._cache[index] = image
        if len(self._cache) > self._cache_size:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        return image


_SHARED_TEXTURES: _LazyTexturePool | None = None


def _initialize_render_worker(seed: int, texture_paths: list[str]) -> None:
    global _SHARED_TEXTURES
    worker_ids = multiprocessing.current_process()._identity
    worker_index = worker_ids[0] if worker_ids else 0
    worker_seed = seed + worker_index
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    from .annotations import data_synthesis

    data_synthesis.rg.seed(worker_seed)
    _SHARED_TEXTURES = _LazyTexturePool(texture_paths)

def _render_text_sample(
    sample_index: int,
    batch: list[str],
    fonts: list[str],
    capitals_dir: str | None,
    output_dir: Path,
    lines_per_image: int,
    font_path: str,
    font_size: int,
):
    if _SHARED_TEXTURES is None:
        raise RuntimeError("Shared textures were not initialized in the worker")
    return generate_single_sample_batch(
        sample_index,
        batch,
        "text",
        fonts,
        capitals_dir,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        output_dir,
        _SHARED_TEXTURES,
        lines_per_image,
        debug=False,
        selected_font_path=font_path,
        font_size=font_size,
    )


def _line_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _load_split(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_pagexml(data_dir: Path, split_paths: dict[str, Path]) -> None:
    processed_dir = data_dir / "annotations-diplomatic" / "processed_splits"
    for path in split_paths.values():
        create_all_pagexml_files(
            processed_dir,
            _load_split(path),
            "diplomatic",
            image_base_path=data_dir,
            target_tasks="ocr",
        )

def _texture_pool(seed: int) -> list[Image.Image]:
    rng = random.Random(seed)
    pool = []
    tessellation = au.VoronoiTessellation(
        mult_range=(50, 80),
        seed=1994,
        num_cells_range=(5000, 6000),
        noise_type="random",
        background_value=(200, 256),
    )
    for _ in range(100):
        width = round((rng.random() + 0.5) * IMAGE_WIDTH)
        height = round((rng.random() + 0.5) * IMAGE_HEIGHT)
        background = (
            91 + rng.randint(-5, 5),
            82 + rng.randint(-5, 5),
            65 + rng.randint(-5, 5),
        )
        image = Image.new("RGB", (width, height), color=background)
        pool.append(Image.fromarray(tessellation(np.asarray(image))))
    return pool


def extend_dataset(
    data_dir: Path,
    font_dir: Path,
    capitals_dir: Path | None,
    target_lines: int,
    lines_per_image: int = LINES_PER_IMAGE,
    seed: int = 2025,
    flush_images: int = 200,
    workers: int = MAX_WORKERS,
) -> int:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    annotations_dir = data_dir / "annotations-diplomatic"
    split_paths = {
        "train": annotations_dir / "ocr_train.json",
        "val": annotations_dir / "ocr_val.json",
    }
    splits = {name: _load_split(path) for name, path in split_paths.items()}
    existing_annotations = [
        annotation
        for split in splits.values()
        for annotation in split.get("annotations", [])
    ]
    line_count = len(existing_annotations)
    if target_lines < line_count:
        raise ValueError(
            f"Target {target_lines:,} is below the existing {line_count:,} OCR lines"
        )
    if target_lines == line_count:
        print(f"Already at target: {line_count:,} OCR text lines")
        _write_pagexml(data_dir, split_paths)
        return line_count

    fonts = get_fonts(font_dir)
    if not fonts:
        raise ValueError(f"No OCR fonts found in {font_dir}")
    if lines_per_image < 1 or flush_images < 1:
        raise ValueError("lines-per-image and flush-images must be positive")

    seen = {
        _line_key(annotation.get("description", ""))
        for annotation in existing_annotations
    }
    selection_rng = random.Random(seed)
    measurement_canvas = Image.new("RGB", (1, 1))
    measurement_draw = ImageDraw.Draw(measurement_canvas)
    font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
    image_ids = [
        int(image["id"])
        for split in splits.values()
        for image in split.get("images", [])
    ]
    next_image_id = max(image_ids, default=-1) + 1
    output_dir = data_dir
    for split in splits.values():
        if not split.get("categories"):
            split["categories"] = [
                {"id": 6, "name": "line", "supercategory": "recognition"}
            ]

    pending = {
        name: {"images": [], "annotations": []}
        for name in split_paths
    }
    pending_images = 0
    started = time.monotonic()
    added_images = 0
    added_lines = 0
    textures = _texture_pool(seed)
    source = iter_unique_text(seed, seen)
    stopped_on_exhaustion = False

    def flush() -> None:
        nonlocal pending_images
        for name, records in pending.items():
            if records["images"]:
                update_json_file(split_paths[name], records)
                records["images"].clear()
                records["annotations"].clear()
        pending_images = 0

    def next_batch(wanted: int) -> tuple[list[str], str, int]:
        nonlocal stopped_on_exhaustion
        font_path = selection_rng.choice(fonts)
        font_size = selection_rng.randint(32, 45)
        font_key = (font_path, font_size)
        font = font_cache.get(font_key)
        if font is None:
            try:
                font = ImageFont.truetype(font_path, size=font_size)
            except OSError:
                font = ImageFont.load_default()
            font_cache[font_key] = font

        batch = []
        while len(batch) < wanted:
            try:
                text = next(source)
            except StopIteration:
                stopped_on_exhaustion = True
                break

            key = _line_key(text)
            if not key or key in seen:
                continue
            text_bbox = measurement_draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x_position = max(20, (IMAGE_WIDTH - text_width) // 2)
            if x_position + text_width <= IMAGE_WIDTH - 20:
                batch.append(text)
        return batch, font_path, font_size

    texture_directory = tempfile.TemporaryDirectory(prefix="synthetic_ocr_textures_")
    texture_paths = []
    futures = deque()
    reserved_lines = 0
    print(f"Rendering with {workers} worker processes (maximum {MAX_WORKERS})")
    try:
        for index, texture in enumerate(textures):
            texture_path = Path(texture_directory.name) / f"texture_{index:03}.png"
            texture.save(texture_path, format="PNG")
            texture_paths.append(str(texture_path))
        textures.clear()

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_render_worker,
            initargs=(seed, texture_paths),
        ) as executor:
            while line_count < target_lines or futures:
                while not stopped_on_exhaustion and len(futures) < workers:
                    available = target_lines - line_count - reserved_lines
                    if available <= 0:
                        break
                    batch, font_path, font_size = next_batch(
                        min(lines_per_image, available)
                    )
                    if not batch:
                        if stopped_on_exhaustion:
                            break
                        continue
                    futures.append(
                        (
                            executor.submit(
                                _render_text_sample,
                                next_image_id,
                                batch,
                                fonts,
                                str(capitals_dir) if capitals_dir else None,
                                output_dir,
                                lines_per_image,
                                font_path,
                                font_size,
                            ),
                            len(batch),
                        )
                    )
                    reserved_lines += len(batch)
                    next_image_id += 1

                if not futures:
                    if stopped_on_exhaustion or line_count >= target_lines:
                        break
                    continue

                future, reserved = futures.popleft()
                reserved_lines -= reserved
                result = future.result()
                if result:
                    image_record, annotations = result
                    for annotation in annotations:
                        key = _line_key(annotation["description"])
                        if key in seen:
                            raise RuntimeError(
                                "A generated OCR line duplicates existing normalized text"
                            )
                        seen.add(key)
                    destination = "val" if added_images % 5 == 4 else "train"
                    pending[destination]["images"].append(image_record)
                    pending[destination]["annotations"].extend(annotations)
                    added_images += 1
                    pending_images += 1
                    added_lines += len(annotations)
                    line_count += len(annotations)

                if result and added_images % 200 == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(
                        f"Generated {added_images:,} images; total OCR lines "
                        f"{line_count:,}/{target_lines:,}; "
                        f"{added_lines / elapsed:,.1f} lines/s; "
                        f"{added_lines / added_images:.1f} lines/image",
                        flush=True,
                    )
                if pending_images >= flush_images:
                    flush()
    finally:
        source.close()
        flush()
        texture_directory.cleanup()

    if line_count < target_lines:
        print(
            f"Text source exhausted after {line_count:,}/{target_lines:,} OCR lines; "
            f"added {added_lines:,} unique lines in {added_images:,} images."
        )
    else:
        print(
            f"Reached {line_count:,} OCR lines; added {added_lines:,} unique lines "
            f"in {added_images:,} images."
        )

    _write_pagexml(data_dir, split_paths)
    if line_count < target_lines:
        raise RuntimeError(
            f"Only {line_count:,} of {target_lines:,} requested OCR lines are available "
            "from the current source and abbreviation variants."
        )
    return line_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append unique historical-text OCR lines to synthetic pretrain data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--font-dir", type=Path, default=DEFAULT_FONT_DIR)
    parser.add_argument("--capitals-dir", type=Path, default=DEFAULT_CAPITALS_DIR)
    parser.add_argument("--target-lines", type=int, default=1_000_000)
    parser.add_argument("--lines-per-image", type=int, default=LINES_PER_IMAGE)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--flush-images", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Image-rendering worker processes (1-{MAX_WORKERS}; default: {MAX_WORKERS})",
    )
    args = parser.parse_args()
    if not args.capitals_dir.exists():
        args.capitals_dir = None
    extend_dataset(
        args.data_dir,
        args.font_dir,
        args.capitals_dir,
        args.target_lines,
        args.lines_per_image,
        args.seed,
        args.flush_images,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
