from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from .ultralytics_monkey_patch import apply_ultralytics_monkey_patch
from .utils import load_image_stems_from_json, save_text_predictions

apply_ultralytics_monkey_patch()

SYMBOL_CATEGORY_NAMES = {"neume", "clef", "custos", "musicDelimiter"}
BASE_LABELS = ["background", "clef_c", "clef_f", "custos", "delimiter_1", "delimiter_2"]
FIXED_IMAGE_HEIGHT = 80
PAPER_BASE_CHANNELS = 64
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


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = len(BASE_LABELS), base_channels: int = PAPER_BASE_CHANNELS):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


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


def _default_neume_offsets(label: str) -> Tuple[float, ...]:
    count = _neume_size(label)
    contour = _neume_contour(label)
    steps = [0.0]
    for token in contour:
        delta = 1.0 if token == "u" else -1.0 if token == "d" else 0.0
        steps.append(steps[-1] + delta)
    if len(steps) != count:
        steps = list(range(count))
    mean_step = sum(steps) / len(steps)
    return tuple(step - mean_step for step in steps)


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


def _resolve_annotation_source(split_json: Path) -> Dict:
    data = _load_coco(split_json)
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    present = {categories.get(ann["category_id"]) for ann in data.get("annotations", [])}
    if {"staff", "neume", "clef", "custos", "musicDelimiter"}.issubset(present):
        return data
    master_json = None
    for parent in split_json.parents:
        candidate = parent / "gt.json"
        if candidate.exists():
            master_json = candidate
            break
    if master_json is None:
        return data
    master = _load_coco(master_json)
    image_ids = {img["id"] for img in data.get("images", [])}
    image_ids.update(ann["image_id"] for ann in data.get("annotations", []))
    file_names = {img["file_name"] for img in data.get("images", [])}
    master["images"] = [
        img for img in master.get("images", [])
        if img["id"] in image_ids or img.get("file_name") in file_names
    ]
    master_ids = {img["id"] for img in master["images"]}
    master["annotations"] = [ann for ann in master.get("annotations", []) if ann["image_id"] in master_ids]
    return master


def build_staff_samples(split_json: Path, data_root: Path) -> List[StaffSample]:
    data = _resolve_annotation_source(split_json)
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
                description = (ann.get("description") or "").strip()
                if not description:
                    continue
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
            if symbols:
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


def _resize_image(image: Image.Image, target_h: int = FIXED_IMAGE_HEIGHT) -> Tuple[np.ndarray, float, float]:
    target_w = max(1, int(round(image.width * (target_h / max(1, image.height)))))
    resized = image.resize((target_w, target_h), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    sx = target_w / image.width
    sy = target_h / image.height
    return arr, sx, sy


def _draw_disk(mask: np.ndarray, cx: float, cy: float, radius: int, value: int) -> None:
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(mask.shape[1], int(math.ceil(cx + radius + 1)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(mask.shape[0], int(math.ceil(cy + radius + 1)))
    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                mask[y, x] = value


class BGKStaffDataset(Dataset):
    def __init__(self, samples: Sequence[StaffSample], label_to_index: Dict[str, int], image_height: int = FIXED_IMAGE_HEIGHT):
        self.samples = list(samples)
        self.label_to_index = label_to_index
        self.image_height = image_height

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        crop = _load_crop(sample)
        image_arr, sx, sy = _resize_image(crop, self.image_height)
        mask = np.zeros(image_arr.shape, dtype=np.int64)
        radius = max(2, int(round(sample.dsl * sy / 8.0)))
        for symbol in sample.symbols:
            if symbol.label not in self.label_to_index:
                continue
            x, y, w, h = symbol.bbox
            cx = (x + w / 2.0) * sx
            cy = (y + h / 2.0) * sy
            _draw_disk(mask, cx, cy, radius, self.label_to_index[symbol.label])
        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask),
        }


def _pad_to_width(tensor: torch.Tensor, width: int) -> torch.Tensor:
    pad_width = width - tensor.shape[-1]
    if pad_width <= 0:
        return tensor
    return F.pad(tensor, (0, pad_width))


def _collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_width = max(item["image"].shape[-1] for item in batch)
    return {
        "image": torch.stack([_pad_to_width(item["image"], max_width) for item in batch]),
        "mask": torch.stack([_pad_to_width(item["mask"], max_width) for item in batch]),
    }


def _component_bbox(points: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return min_x, min_y, max_x - min_x + 1, max_y - min_y + 1


def _connected_components(binary_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    visited = np.zeros_like(binary_mask, dtype=bool)
    components: List[List[Tuple[int, int]]] = []
    height, width = binary_mask.shape
    for y in range(height):
        for x in range(width):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            components.append(comp)
    return components


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


def _ensure_leading_clef(
    baseline: List[PredictedSymbol],
    uncertain: List[PredictedSymbol],
    previous_clef: Optional[ClefState],
    dsl: float,
) -> List[PredictedSymbol]:
    result = list(baseline)
    if any(symbol.label.startswith("clef") for symbol in result[:2]):
        return result
    clef_candidates = [symbol for symbol in uncertain if symbol.label.startswith("clef")]
    if clef_candidates:
        chosen = max(clef_candidates, key=lambda item: item.confidence)
        result.insert(0, chosen)
        return sorted(result, key=lambda item: item.center[0])
    if previous_clef:
        result.insert(
            0,
            PredictedSymbol(
                label="clef_c" if previous_clef.description.startswith("KC") else "clef_f",
                description=previous_clef.description,
                bbox=(0.0, max(0.0, previous_clef.center_y - dsl), dsl, dsl * 2),
                center=(dsl, previous_clef.center_y),
                confidence=0.0,
                source="prior",
            ),
        )
    return result


def _postprocess_staff(
    baseline: List[PredictedSymbol],
    uncertain: List[PredictedSymbol],
    previous_clef: Optional[ClefState],
    dsl: float,
) -> List[PredictedSymbol]:
    cleaned = _remove_overlaps(baseline, dsl)
    cleaned = _ensure_leading_clef(cleaned, uncertain, previous_clef, dsl)
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
        if symbol.source == "prior" and symbol.description:
            return symbol.description
        line = _estimate_clef_line(symbol, sample)
        return f"K{'C' if symbol.label == 'clef_c' else 'F'}{line}"
    if current_clef is None:
        return ""
    anchor = _symbol_anchor_value(symbol, current_clef, sample)
    if symbol.label == "custos":
        return _value_to_pitch(int(round(anchor)))
    if symbol.label.startswith("neume_"):
        offsets = neume_templates.get(symbol.label, _default_neume_offsets(symbol.label))
        notes = [_value_to_pitch(int(round(anchor + offset))) for offset in offsets]
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


def _predict_symbols(
    model: nn.Module,
    sample: StaffSample,
    index_to_label: Dict[int, str],
    device: torch.device,
    image_height: int = FIXED_IMAGE_HEIGHT,
) -> Tuple[List[PredictedSymbol], List[PredictedSymbol]]:
    crop = _load_crop(sample)
    image_arr, sx, sy = _resize_image(crop, image_height)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    bg = probs[0]
    best_non_bg = np.argmax(probs[1:], axis=0) + 1
    confidence = np.max(probs[1:], axis=0)

    predictions: Dict[str, List[PredictedSymbol]] = {"baseline": [], "uncertain": []}
    for source, threshold in (("baseline", 0.5), ("uncertain", 0.9)):
        label_map = best_non_bg.copy()
        label_map[bg >= threshold] = 0
        if source == "uncertain":
            baseline_mask = best_non_bg.copy()
            baseline_mask[bg >= 0.5] = 0
            label_map[baseline_mask > 0] = 0
        for label_idx in range(1, len(index_to_label)):
            class_mask = label_map == label_idx
            components = _connected_components(class_mask)
            for component in components:
                if len(component) < 3:
                    continue
                min_x, min_y, width, height = _component_bbox(component)
                conf = float(np.mean([confidence[y, x] for y, x in component]))
                orig_bbox = (
                    min_x / sx,
                    min_y / sy,
                    max(1.0, width / sx),
                    max(1.0, height / sy),
                )
                predictions[source].append(
                    PredictedSymbol(
                        label=index_to_label[label_idx],
                        description="",
                        bbox=orig_bbox,
                        center=(orig_bbox[0] + orig_bbox[2] / 2.0, orig_bbox[1] + orig_bbox[3] / 2.0),
                        confidence=conf,
                        source=source,
                    )
                )
    return predictions["baseline"], predictions["uncertain"]


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


def _model_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.is_dir():
        candidate = path / "model.pt"
        return candidate if candidate.exists() else None
    return path if path.exists() else None


def _save_yolo_crop(sample: StaffSample, image_dir: Path) -> Tuple[Path, Image.Image]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample.image_stem}_staff{sample.staff_index:02d}.png"
    crop = _load_crop(sample).convert("RGB")
    crop.save(image_path)
    return image_path, crop


def _write_yolo_label(sample: StaffSample, label_dir: Path, label_to_index: Dict[str, int], crop_size: Tuple[int, int]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{sample.image_stem}_staff{sample.staff_index:02d}.txt"
    width, height = crop_size
    lines: List[str] = []
    for symbol in sample.symbols:
        if symbol.label not in label_to_index:
            continue
        x, y, w, h = symbol.bbox
        cx = (x + w / 2.0) / max(width, 1)
        cy = (y + h / 2.0) / max(height, 1)
        nw = w / max(width, 1)
        nh = h / max(height, 1)
        lines.append(f"{label_to_index[symbol.label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_yolo_split(samples: Sequence[StaffSample], split_dir: Path, label_to_index: Dict[str, int]) -> Dict[str, StaffSample]:
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, StaffSample] = {}
    for sample in samples:
        image_path, crop = _save_yolo_crop(sample, image_dir)
        _write_yolo_label(sample, label_dir, label_to_index, crop.size)
        mapping[image_path.name] = sample
    return mapping


def _prepare_yolo_dataset(
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    test_samples: Sequence[StaffSample],
    labels: Sequence[str],
) -> Tuple[Path, Dict[str, Dict[str, StaffSample]]]:
    artifacts_dir = Path(tempfile.mkdtemp(prefix="bgk_yolo_artifacts_"))
    dataset_dir = artifacts_dir / "dataset"
    detector_labels = [label for label in labels if label != "background"]
    detector_label_to_index = {label: idx for idx, label in enumerate(detector_labels)}
    val_split_samples = val_samples if val_samples else train_samples
    mappings = {
        "train": _prepare_yolo_split(train_samples, dataset_dir / "train", detector_label_to_index),
        "val": _prepare_yolo_split(val_split_samples, dataset_dir / "val", detector_label_to_index),
        "test": _prepare_yolo_split(test_samples, dataset_dir / "test", detector_label_to_index),
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir.resolve()),
                "train": "train/images",
                "val": "val/images",
                "names": {idx: label for idx, label in enumerate(detector_labels)},
            }
        ),
        encoding="utf-8",
    )
    return artifacts_dir, mappings


def _train_yolo_model(
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    test_samples: Sequence[StaffSample],
    labels: Sequence[str],
    save_model_path: Path,
    load_model_path: Optional[Path],
    model_identifier: str,
    debug: bool,
) -> Tuple[Path, Path, Dict[str, Dict[str, StaffSample]]]:
    artifacts_dir, mappings = _prepare_yolo_dataset(train_samples, val_samples, test_samples, labels)
    existing_model = _model_path(load_model_path)
    model_source = str(existing_model) if existing_model else model_identifier
    model = YOLO(model_source)
    epochs = 1 if debug else 30
    model.train(
        data=str(artifacts_dir / "dataset" / "data.yaml"),
        epochs=epochs,
        patience=10,
        imgsz=960,
        project=str(artifacts_dir),
        name="train",
        exist_ok=True,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch=8 if not debug else 4,
        workers=2,
        lr0=0.01,
    )
    best_model_path = artifacts_dir / "train" / "weights" / "best.pt"
    if not best_model_path.exists():
        best_model_path = artifacts_dir / "train" / "weights" / "last.pt"
    save_model_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_path, save_model_path / "model.pt")
    return best_model_path, artifacts_dir, mappings


def _predict_symbols_yolo(
    model: YOLO,
    detector_labels: Sequence[str],
    image_path: Path,
) -> Tuple[List[PredictedSymbol], List[PredictedSymbol]]:
    results = model.predict(
        source=str(image_path),
        conf=0.05,
        iou=0.5,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    baseline: List[PredictedSymbol] = []
    uncertain: List[PredictedSymbol] = []
    if not results:
        return baseline, uncertain
    res = results[0]
    if res.boxes is None:
        return baseline, uncertain
    for box in res.boxes:
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
            source="baseline" if conf >= 0.25 else "uncertain",
        )
        if conf >= 0.25:
            baseline.append(pred)
        else:
            uncertain.append(pred)
    return baseline, uncertain


def _train_model(
    model: nn.Module,
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    label_to_index: Dict[str, int],
    save_model_path: Path,
    device: torch.device,
    debug: bool,
) -> None:
    train_loader = DataLoader(
        BGKStaffDataset(train_samples, label_to_index),
        batch_size=2 if not debug else 1,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        BGKStaffDataset(val_samples, label_to_index),
        batch_size=2 if not debug else 1,
        shuffle=False,
        collate_fn=_collate,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val = float("inf")
    epochs = 1 if debug else 30
    save_model_path.mkdir(parents=True, exist_ok=True)
    out_path = save_model_path / "model.pt"
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                losses.append(float(F.cross_entropy(logits, masks).item()))
        val_loss = sum(losses) / max(1, len(losses))
        print(f"   epoch={epoch + 1} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)
    if out_path.exists():
        model.load_state_dict(torch.load(out_path, map_location=device))


def run_bgk_omr_pipeline(
    args,
    train_json: Optional[Path],
    val_json: Optional[Path],
    test_json: Optional[Path],
    output_dir: Path,
    save_model_path: Path,
    load_model_path: Optional[Path],
    model_identifier: str = "yolov8n.pt",
) -> None:
    data_root = Path(args.data_dir) if getattr(args, "data_dir", None) else Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_json is None or val_json is None:
        raise ValueError("bgk requires train_json and val_json")

    train_samples = build_staff_samples(Path(train_json), data_root)
    val_samples = build_staff_samples(Path(val_json), data_root)
    test_samples = build_staff_samples(Path(test_json), data_root) if test_json else []
    expected_test_stems = load_image_stems_from_json(Path(test_json)) if test_json else []
    labels, label_to_index, index_to_label = build_label_space([*train_samples, *val_samples])
    print(f"🧾 BGK staff samples train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} labels={len(labels)}")

    detector_labels = [label for label in labels if label != "background"]
    best_model_path, artifacts_dir, mappings = _train_yolo_model(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        labels=labels,
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
        image_path = artifacts_dir / "dataset" / "test" / "images" / image_name
        baseline, uncertain = _predict_symbols_yolo(yolo_model, detector_labels, image_path)
        tokens = _postprocess_staff(baseline, uncertain, previous_clef_by_page[sample.image_stem], sample.dsl)
        decoded_tokens, current_clef = _decode_staff(tokens, sample, previous_clef_by_page[sample.image_stem], neume_templates)
        previous_clef_by_page[sample.image_stem] = current_clef
        page_texts[sample.image_stem][sample.staff_index] = " ".join(
            token.description for token in decoded_tokens if token.description
        )
    _write_predictions(expected_test_stems, page_texts, output_dir)
    shutil.rmtree(artifacts_dir, ignore_errors=True)
