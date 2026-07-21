from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from ..ultralytics_monkey_patch import apply_ultralytics_monkey_patch
from ..utils import load_image_stems_from_json, save_text_predictions

apply_ultralytics_monkey_patch()

SYMBOL_CATEGORY_NAMES = {"neume", "clef", "custos", "musicDelimiter"}
BASE_LABELS = ["background", "clef_c", "clef_f", "custos", "delimiter_1", "delimiter_2"]
FIXED_IMAGE_HEIGHT = 80
NOTE_ORDER = "CDEFGAB"
PITCH_RE = re.compile(r"([A-G])(\d+)")
CLEF_RE = re.compile(r"^K([CF])(\d+)$")
N_STAFF_LINES = 5


@dataclass
class SymbolAnnotation:
    label: str
    description: str
    bbox: Tuple[float, float, float, float]


@dataclass
class StaffSample:
    image_path: Path
    image_stem: str
    staff_index: int
    crop_box: Tuple[int, int, int, int]
    staff_box: Tuple[float, float, float, float]
    dsl: float
    symbols: List[SymbolAnnotation]


@dataclass
class PredictedSymbol:
    label: str
    description: str
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    confidence: float
    source: str


@dataclass
class ClefState:
    description: str
    center_y: float



def _load_coco(json_path: Path) -> Dict:
    return json.loads(json_path.read_text(encoding="utf-8"))



def _pitch_to_value(token: str) -> int:
    match = PITCH_RE.fullmatch(token)
    if match is None:
        raise ValueError(f"Invalid pitch token: {token}")
    note, octave = match.groups()
    return int(octave) * 7 + NOTE_ORDER.index(note)


def _value_to_pitch(value: int) -> str:
    octave, note_index = divmod(value, len(NOTE_ORDER))
    return f"{NOTE_ORDER[note_index]}{octave}"


def _extract_pitch_tokens(description: str) -> List[str]:
    return [match.group(0) for match in PITCH_RE.finditer(description)]


def _contour_from_pitches(pitches: Sequence[str]) -> str:
    if len(pitches) <= 1:
        return ""
    values = [_pitch_to_value(token) for token in pitches]
    contour = []
    for current, nxt in zip(values, values[1:]):
        if nxt > current:
            contour.append("u")
        elif nxt < current:
            contour.append("d")
        else:
            contour.append("s")
    return "".join(contour)


def _neume_label(count: int, contour: str) -> str:
    return f"neume_{count}" if not contour else f"neume_{count}_{contour}"


def _neume_size(label: str) -> int:
    parts = label.split("_")
    if len(parts) < 2 or parts[0] != "neume":
        raise ValueError(f"Invalid neume label: {label}")
    return int(parts[1])


def _neume_contour(label: str) -> str:
    parts = label.split("_", 2)
    return parts[2] if len(parts) == 3 else ""


def _label_sort_key(label: str) -> Tuple[int, int, str]:
    if label in BASE_LABELS:
        return (0, BASE_LABELS.index(label), "")
    if label.startswith("neume_"):
        return (1, _neume_size(label), _neume_contour(label))
    return (2, 0, label)


def _label_family(label: str) -> str:
    if label.startswith("neume_"):
        return "neume"
    if label.startswith("delimiter_"):
        return "delimiter"
    return label


def _label_from_annotation(category_name: str, description: str) -> Optional[str]:
    if category_name == "neume":
        pitches = _extract_pitch_tokens(description)
        if not pitches:
            return None
        return _neume_label(len(pitches), _contour_from_pitches(pitches))
    if category_name == "custos":
        return "custos"
    if category_name == "musicDelimiter":
        return "delimiter_2" if description.count("/") >= 2 else "delimiter_1"
    if category_name == "clef":
        if description.startswith("KC"):
            return "clef_c"
        if description.startswith("KF"):
            return "clef_f"
        return None
    return None


def _center_in_bbox(center: Tuple[float, float], bbox: Sequence[float]) -> bool:
    x, y, w, h = bbox
    cx, cy = center
    return x <= cx <= x + w and y <= cy <= y + h


def _annotation_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0



def build_staff_samples(split_json: Path, data_root: Path) -> List[StaffSample]:
    data = _load_coco(split_json)
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    anns_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    samples: List[StaffSample] = []
    for image_id, image in images.items():
        image_path = data_root / image["file_name"]
        image_stem = Path(image["file_name"]).stem
        staff_anns = [ann for ann in anns_by_image[image_id] if categories.get(ann["category_id"]) == "staff"]
        staff_anns.sort(key=lambda ann: (ann["bbox"][1], ann["bbox"][0]))
        page_anns = anns_by_image[image_id]
        for staff_index, staff_ann in enumerate(staff_anns):
            x, y, w, h = staff_ann["bbox"]
            dsl = max(8.0, h / 4.0)
            margin_y = int(max(1, round(dsl)))
            margin_right = int(max(1, round(dsl)))
            margin_left = int(max(1, round(4 * dsl)))
            left = max(0, int(round(x - margin_left)))
            top = max(0, int(round(y - margin_y)))
            right = int(round(x + w + margin_right))
            bottom = int(round(y + h + margin_y))
            symbols: List[SymbolAnnotation] = []
            for ann in page_anns:
                category_name = categories.get(ann["category_id"])
                if category_name not in SYMBOL_CATEGORY_NAMES:
                    continue
                description = (ann.get("description") or ann.get("text") or "").strip()
                center = _annotation_center(ann["bbox"])
                if not _center_in_bbox(center, staff_ann["bbox"]):
                    continue
                label = _label_from_annotation(category_name, description)
                if label is None:
                    continue
                ax, ay, aw, ah = ann["bbox"]
                symbols.append(
                    SymbolAnnotation(
                        label=label,
                        description=description,
                        bbox=(ax - left, ay - top, aw, ah),
                    )
                )
            samples.append(
                    StaffSample(
                        image_path=image_path,
                        image_stem=image_stem,
                        staff_index=staff_index,
                        crop_box=(left, top, right, bottom),
                        staff_box=(x - left, y - top, w, h),
                        dsl=dsl,
                        symbols=symbols,
                    )
                )
    return samples


def build_label_space(samples: Sequence[StaffSample]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    neume_labels = sorted(
        {
            symbol.label
            for sample in samples
            for symbol in sample.symbols
            if symbol.label.startswith("neume_")
        },
        key=_label_sort_key,
    )
    labels = [*BASE_LABELS, *neume_labels]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return labels, label_to_index, index_to_label


def _load_crop(sample: StaffSample) -> Image.Image:
    image = Image.open(sample.image_path).convert("L")
    return image.crop(sample.crop_box)



def _symbol_center_from_annotation(symbol: SymbolAnnotation) -> Tuple[float, float]:
    x, y, w, h = symbol.bbox
    return x + w / 2.0, y + h / 2.0


def _staff_pitch_step(sample: StaffSample) -> float:
    return max(sample.staff_box[3] / (N_STAFF_LINES * 2.0), 1e-6)


def _clef_pitch_value(description: str) -> int:
    match = CLEF_RE.fullmatch(description)
    if match is None:
        raise ValueError(f"Invalid clef token: {description}")
    kind, _ = match.groups()
    return _pitch_to_value("C4" if kind == "C" else "F3")


def build_neume_templates(samples: Sequence[StaffSample]) -> Dict[str, Tuple[float, ...]]:
    offsets_by_label: Dict[str, List[Tuple[float, ...]]] = defaultdict(list)
    previous_clef_by_page: Dict[str, Optional[ClefState]] = defaultdict(lambda: None)
    ordered_samples = sorted(samples, key=lambda sample: (sample.image_stem, sample.staff_index))
    for sample in ordered_samples:
        current_clef = previous_clef_by_page[sample.image_stem]
        for symbol in sorted(sample.symbols, key=lambda item: _symbol_center_from_annotation(item)[0]):
            _, center_y = _symbol_center_from_annotation(symbol)
            if symbol.label.startswith("clef"):
                current_clef = ClefState(description=symbol.description, center_y=center_y)
                continue
            if current_clef is None or not symbol.label.startswith("neume_"):
                continue
            pitches = _extract_pitch_tokens(symbol.description)
            if len(pitches) != _neume_size(symbol.label):
                continue
            anchor = _clef_pitch_value(current_clef.description) + (current_clef.center_y - center_y) / _staff_pitch_step(sample)
            offsets_by_label[symbol.label].append(tuple(_pitch_to_value(pitch) - anchor for pitch in pitches))
        previous_clef_by_page[sample.image_stem] = current_clef

    templates: Dict[str, Tuple[float, ...]] = {}
    for label, examples in offsets_by_label.items():
        count = _neume_size(label)
        templates[label] = tuple(
            float(np.mean([example[idx] for example in examples]))
            for idx in range(count)
        )
    return templates


def _normalized_overlap(a: PredictedSymbol, b: PredictedSymbol, dsl: float) -> bool:
    scale = {
        "neume": (0.4, 0.4),
        "custos": (0.35, 0.35),
        "delimiter": (0.12, 1.0),
        "clef_c": (0.6, 1.6),
        "clef_f": (0.8, 1.6),
    }
    ax, ay = a.center
    bx, by = b.center
    aw, ah = scale.get(_label_family(a.label), (0.4, 0.4))
    bw, bh = scale.get(_label_family(b.label), (0.4, 0.4))
    a_box = (ax - dsl * aw, ay - dsl * ah, ax + dsl * aw, ay + dsl * ah)
    b_box = (bx - dsl * bw, by - dsl * bh, bx + dsl * bw, by + dsl * bh)
    return not (
        a_box[2] < b_box[0]
        or b_box[2] < a_box[0]
        or a_box[3] < b_box[1]
        or b_box[3] < a_box[1]
    )


def _remove_overlaps(symbols: List[PredictedSymbol], dsl: float) -> List[PredictedSymbol]:
    result: List[PredictedSymbol] = []
    for symbol in sorted(symbols, key=lambda item: (-item.confidence, item.center[0])):
        keep = True
        for kept in result:
            if _normalized_overlap(symbol, kept, dsl):
                if symbol.label.startswith("clef") and not kept.label.startswith("clef"):
                    result.remove(kept)
                    break
                keep = False
                break
        if keep:
            result.append(symbol)
    return sorted(result, key=lambda item: item.center[0])



def _postprocess_staff(
    symbols: List[PredictedSymbol],
    dsl: float,
) -> List[PredictedSymbol]:
    cleaned = _remove_overlaps(symbols, dsl)
    tokens: List[PredictedSymbol] = []
    last_delimiter = False
    for symbol in cleaned:
        if _label_family(symbol.label) == "delimiter":
            if last_delimiter and tokens:
                tokens[-1] = symbol
            else:
                tokens.append(symbol)
            last_delimiter = True
            continue
        last_delimiter = False
        tokens.append(symbol)
    return tokens


def _estimate_clef_line(symbol: PredictedSymbol, sample: StaffSample) -> int:
    _, staff_top, _, staff_height = sample.staff_box
    line_spacing = max(staff_height / float(N_STAFF_LINES), 1e-6)
    k = min(range(N_STAFF_LINES), key=lambda i: abs(symbol.center[1] - (staff_top + i * line_spacing)))
    return max(1, min(4, N_STAFF_LINES - k))


def _symbol_anchor_value(symbol: PredictedSymbol, clef: ClefState, sample: StaffSample) -> float:
    return _clef_pitch_value(clef.description) + (clef.center_y - symbol.center[1]) / _staff_pitch_step(sample)


def _decode_symbol_description(
    symbol: PredictedSymbol,
    sample: StaffSample,
    current_clef: Optional[ClefState],
    neume_templates: Dict[str, Tuple[float, ...]],
) -> str:
    if symbol.label == "delimiter_1":
        return "/"
    if symbol.label == "delimiter_2":
        return "//"
    if symbol.label.startswith("clef"):
        line = _estimate_clef_line(symbol, sample)
        return f"K{'C' if symbol.label == 'clef_c' else 'F'}{line}"
    if current_clef is None:
        return ""
    anchor = _symbol_anchor_value(symbol, current_clef, sample)
    if symbol.label == "custos":
        return _value_to_pitch(int(round(anchor)))
    if symbol.label.startswith("neume_"):
        notes = [_value_to_pitch(int(round(anchor + offset))) for offset in neume_templates[symbol.label]]
        return f"({' '.join(notes)})"
    return ""


def _decode_staff(
    symbols: Sequence[PredictedSymbol],
    sample: StaffSample,
    previous_clef: Optional[ClefState],
    neume_templates: Dict[str, Tuple[float, ...]],
) -> Tuple[List[PredictedSymbol], Optional[ClefState]]:
    decoded: List[PredictedSymbol] = []
    current_clef = previous_clef
    for symbol in sorted(symbols, key=lambda item: item.center[0]):
        description = _decode_symbol_description(symbol, sample, current_clef, neume_templates)
        if not description:
            continue
        decoded_symbol = PredictedSymbol(
            label=symbol.label,
            description=description,
            bbox=symbol.bbox,
            center=symbol.center,
            confidence=symbol.confidence,
            source=symbol.source,
        )
        decoded.append(decoded_symbol)
        if decoded_symbol.label.startswith("clef"):
            current_clef = ClefState(description=decoded_symbol.description, center_y=decoded_symbol.center[1])
    return decoded, current_clef



def _write_predictions(expected_stems: Sequence[str], texts: Dict[str, Dict[int, str]], output_dir: Path) -> None:
    seen = set()
    ordered_stems = []
    for stem in expected_stems:
        if stem not in seen:
            ordered_stems.append(stem)
            seen.add(stem)
    for stem in texts:
        if stem not in seen:
            ordered_stems.append(stem)
            seen.add(stem)

    for image_stem in ordered_stems:
        staff_texts = texts.get(image_stem, {})
        text = " ".join(
            value.strip()
            for _, value in sorted(staff_texts.items())
            if value and value.strip()
        ).strip()
        save_text_predictions(image_stem, text, output_dir)



def _save_yolo_crop(sample: StaffSample, image_dir: Path) -> Tuple[Path, Image.Image]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample.image_stem}_staff{sample.staff_index:02d}.png"
    src = sample.image_path
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")
    crop = _load_crop(sample).convert("RGB")
    if crop.width == 0 or crop.height == 0:
        raise ValueError(f"Degenerate crop (0-size) for {sample.image_stem} staff {sample.staff_index}: box={sample.crop_box}")
    crop.save(image_path)
    return image_path, crop


def _write_yolo_label(sample: StaffSample, label_dir: Path, label_to_index: Dict[str, int], crop_size: Tuple[int, int]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{sample.image_stem}_staff{sample.staff_index:02d}.txt"
    width, height = crop_size
    lines: List[str] = []
    max_width = max(width, 1)
    max_height = max(height, 1)
    for symbol in sample.symbols:
        if symbol.label not in label_to_index:
            continue
        x, y, w, h = symbol.bbox
        x1 = min(max(x, 0.0), float(width))
        y1 = min(max(y, 0.0), float(height))
        x2 = min(max(x + w, 0.0), float(width))
        y2 = min(max(y + h, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = ((x1 + x2) / 2.0) / max_width
        cy = ((y1 + y2) / 2.0) / max_height
        nw = (x2 - x1) / max_width
        nh = (y2 - y1) / max_height
        lines.append(f"{label_to_index[symbol.label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _write_label_distribution_plot(samples: Sequence[StaffSample], labels: Sequence[str], plot_path: Path) -> None:
    counts = Counter(symbol.label for sample in samples for symbol in sample.symbols if symbol.label in labels and symbol.label != "background")
    xs = [label for label in labels if label != "background"]
    ys = [counts.get(label, 0) for label in xs]
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(xs) * 0.45), 4.5))
    sns.barplot(x=xs, y=ys, ax=ax, color="#4C78A8")
    ax.set_title("BGK YOLO label distribution")
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    for tick, value in zip(ax.patches, ys):
        ax.text(tick.get_x() + tick.get_width() / 2.0, tick.get_height(), str(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _prepare_yolo_split(samples: Sequence[StaffSample], images_dir: Path, labels_dir: Path, label_to_index: Dict[str, int]) -> Dict[str, StaffSample]:
    image_dir = images_dir
    label_dir = labels_dir
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, StaffSample] = {}

    def _process(sample: StaffSample) -> Tuple[str, StaffSample]:
        image_path, crop = _save_yolo_crop(sample, image_dir)
        _write_yolo_label(sample, label_dir, label_to_index, crop.size)
        return image_path.name, sample

    max_workers = min(8, max(1, len(samples)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for image_name, sample in executor.map(_process, samples):
            mapping[image_name] = sample
    return mapping


def _prepare_yolo_dataset(
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    test_samples: Sequence[StaffSample],
    labels: Sequence[str],
    cache_root: Path,
    cache_key: str,
) -> Tuple[Path, Dict[str, Dict[str, StaffSample]]]:
    dataset_dir = cache_root / cache_key
    detector_labels = [label for label in labels if label != "background"]
    detector_label_to_index = {label: idx for idx, label in enumerate(detector_labels)}
    val_split_samples = val_samples
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        _prepare_yolo_split(train_samples, dataset_dir / "images" / "train", dataset_dir / "labels" / "train", detector_label_to_index)
        _prepare_yolo_split(val_split_samples, dataset_dir / "images" / "val", dataset_dir / "labels" / "val", detector_label_to_index)
        _prepare_yolo_split(test_samples, dataset_dir / "images" / "test", dataset_dir / "labels" / "test", detector_label_to_index)
        _write_label_distribution_plot(train_samples + list(val_split_samples) + list(test_samples), detector_labels, dataset_dir.parent / "label_distribution.png")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(dataset_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {idx: label for idx, label in enumerate(detector_labels)},
                }
            ),
            encoding="utf-8",
        )
    mappings = {
        "train": {f"{s.image_stem}_staff{s.staff_index:02d}.png": s for s in train_samples},
        "val": {f"{s.image_stem}_staff{s.staff_index:02d}.png": s for s in val_split_samples},
        "test": {f"{s.image_stem}_staff{s.staff_index:02d}.png": s for s in test_samples},
    }
    return dataset_dir, mappings



def _train_yolo_model(
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    test_samples: Sequence[StaffSample],
    labels: Sequence[str],
    cache_key: str,
    save_model_path: Path,
    load_model_path: Optional[Path],
    model_identifier: str,
    debug: bool,
) -> Tuple[Path, Path, Dict[str, Dict[str, StaffSample]]]:
    dataset_dir, mappings = _prepare_yolo_dataset(train_samples, val_samples, test_samples, labels, save_model_path.parent / "bgk_yolo_cache", cache_key)
    model_source = str(load_model_path) if load_model_path else model_identifier
    model = YOLO(model_source)
    epochs = 1 if debug else 30
    model.train(
        data=str(dataset_dir / "data.yaml"),
        epochs=epochs,
        patience=10,
        imgsz=960,
        project=str(save_model_path.parent),
        name="train",
        exist_ok=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch=8 if not debug else 4,
        workers=2,
        lr0=0.01,
    )
    best_model_path = save_model_path.parent / "train" / "weights" / "best.pt"
    save_model_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, save_model_path / "model.pt")
    return best_model_path, dataset_dir, mappings


def _predict_symbols_yolo(
    model: YOLO,
    detector_labels: Sequence[str],
    image_path: Path,
) -> List[PredictedSymbol]:
    results = model.predict(
        source=str(image_path),
        conf=0.05,
        iou=0.5,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    predictions: List[PredictedSymbol] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].item())
        class_idx = int(box.cls[0].item())
        if class_idx < 0 or class_idx >= len(detector_labels):
            continue
        label = detector_labels[class_idx]
        pred = PredictedSymbol(
            label=label,
            description="",
            bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            center=(float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)),
            confidence=conf,
            source="yolo",
        )
        predictions.append(pred)
    return predictions



def train_test_bgk(
    args,
    is_train_test_mode,
    is_sequential,
    output_dir: Path,
    train_json: Optional[Path],
    val_json: Optional[Path],
    test_json: Optional[Path],
    save_model_path: Path,
    load_model_path: Optional[Path],
    model_identifier: str = "yolov8n.pt",
) -> None:
    train_root = Path(args.data_dir or args.train_dir)
    test_root = Path(args.data_dir or args.test_dir)

    if train_json is None or val_json is None:
        raise ValueError("bgk requires train_json and val_json")

    train_samples = build_staff_samples(Path(train_json), train_root)
    val_samples = build_staff_samples(Path(val_json), train_root)
    test_samples = build_staff_samples(Path(test_json), test_root) if test_json else []
    expected_test_stems = load_image_stems_from_json(Path(test_json)) if test_json else []
    labels, _, _ = build_label_space([*train_samples, *val_samples])
    print(f"🧾 BGK staff samples train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} labels={len(labels)}")

    detector_labels = [label for label in labels if label != "background"]
    cache_key = hashlib.sha1("|".join(map(str, [train_json, val_json, test_json, train_root, test_root, *labels])).encode("utf-8")).hexdigest()[:16]
    best_model_path, dataset_dir, mappings = _train_yolo_model(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        labels=labels,
        cache_key=cache_key,
        save_model_path=save_model_path,
        load_model_path=Path(load_model_path) if load_model_path else None,
        model_identifier=model_identifier,
        debug=getattr(args, "debug", False),
    )
    yolo_model = YOLO(str(best_model_path))

    neume_templates = build_neume_templates(train_samples)
    page_texts: Dict[str, Dict[int, str]] = defaultdict(dict)
    previous_clef_by_page: Dict[str, Optional[ClefState]] = defaultdict(lambda: None)
    test_image_map = mappings["test"]
    for image_name, sample in sorted(test_image_map.items(), key=lambda item: (item[1].image_stem, item[1].staff_index)):
        image_path = dataset_dir / "images" / "test" / image_name
        predictions = _predict_symbols_yolo(yolo_model, detector_labels, image_path)
        tokens = _postprocess_staff(predictions, sample.dsl)
        decoded_tokens, current_clef = _decode_staff(tokens, sample, previous_clef_by_page[sample.image_stem], neume_templates)
        previous_clef_by_page[sample.image_stem] = current_clef
        page_texts[sample.image_stem][sample.staff_index] = " ".join(
            token.description for token in decoded_tokens if token.description
        )
    _write_predictions(expected_test_stems, page_texts, output_dir)
