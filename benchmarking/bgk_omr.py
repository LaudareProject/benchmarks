from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SYMBOL_CATEGORY_NAMES = {"neume", "clef", "custos", "musicDelimiter"}
LABELS = ["background", "neume", "clef_c", "clef_f", "custos", "delimiter"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}
FIXED_IMAGE_SIZE = (256, 1024)
PROTOTYPE_SIZE = (48, 48)


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
    def __init__(self, in_channels: int = 1, out_channels: int = len(LABELS), base_channels: int = 32):
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


def _label_from_annotation(category_name: str, description: str) -> Optional[str]:
    if category_name == "neume":
        return "neume"
    if category_name == "custos":
        return "custos"
    if category_name == "musicDelimiter":
        return "delimiter"
    if category_name == "clef":
        if description.startswith("KC"):
            return "clef_c"
        return "clef_f"
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


def build_staff_samples(split_json: Path, data_root: Path, margin_ratio: float = 0.25) -> List[StaffSample]:
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
            margin_x = int(max(8, w * margin_ratio * 0.1))
            margin_y = int(max(8, dsl * 0.8))
            left = max(0, int(x - margin_x))
            top = max(0, int(y - margin_y))
            right = int(x + w + margin_x)
            bottom = int(y + h + margin_y)
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
                        dsl=dsl,
                        symbols=symbols,
                    )
                )
    return samples


def _load_crop(sample: StaffSample) -> Image.Image:
    image = Image.open(sample.image_path).convert("L")
    return image.crop(sample.crop_box)


def _resize_image(image: Image.Image, size: Tuple[int, int]) -> Tuple[np.ndarray, float, float]:
    target_h, target_w = size
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
    def __init__(self, samples: Sequence[StaffSample], image_size: Tuple[int, int] = FIXED_IMAGE_SIZE):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        crop = _load_crop(sample)
        image_arr, sx, sy = _resize_image(crop, self.image_size)
        mask = np.zeros(self.image_size, dtype=np.int64)
        radius = max(2, int(round(sample.dsl * sy / 8.0)))
        for symbol in sample.symbols:
            x, y, w, h = symbol.bbox
            cx = (x + w / 2.0) * sx
            cy = (y + h / 2.0) * sy
            _draw_disk(mask, cx, cy, radius, LABEL_TO_INDEX[symbol.label])
        return {
            "image": torch.from_numpy(image_arr).unsqueeze(0),
            "mask": torch.from_numpy(mask),
        }


def _collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
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


def _prototype_vector(image: Image.Image) -> np.ndarray:
    resized = image.resize(PROTOTYPE_SIZE[::-1], Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    vec = arr.flatten()
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def build_prototype_bank(samples: Sequence[StaffSample]) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    bank: Dict[str, List[Tuple[np.ndarray, str]]] = defaultdict(list)
    cache: Dict[Path, Image.Image] = {}
    for sample in samples:
        if sample.image_path not in cache:
            cache[sample.image_path] = Image.open(sample.image_path).convert("L")
        page = cache[sample.image_path]
        left, top, _, _ = sample.crop_box
        for symbol in sample.symbols:
            x, y, w, h = symbol.bbox
            gx0 = int(left + x)
            gy0 = int(top + y)
            gx1 = int(gx0 + w)
            gy1 = int(gy0 + h)
            patch = page.crop((gx0, gy0, gx1, gy1))
            bank[symbol.label].append((_prototype_vector(patch), symbol.description))
    return bank


def _classify_crop(patch: Image.Image, label: str, bank: Dict[str, List[Tuple[np.ndarray, str]]]) -> Tuple[str, float]:
    candidates = bank.get(label, [])
    if not candidates:
        return "", 0.0
    vec = _prototype_vector(patch)
    best_description = ""
    best_score = -1.0
    for prototype, description in candidates:
        score = float(np.dot(vec, prototype))
        if score > best_score:
            best_score = score
            best_description = description
    return best_description, best_score


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
    aw, ah = scale.get(a.label, (0.4, 0.4))
    bw, bh = scale.get(b.label, (0.4, 0.4))
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
    previous_clef: Optional[str],
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
                label="clef_c" if previous_clef.startswith("KC") else "clef_f",
                description=previous_clef,
                bbox=(0.0, 0.0, dsl, dsl * 2),
                center=(dsl, dsl),
                confidence=0.0,
                source="prior",
            ),
        )
    return result


def _postprocess_staff(
    baseline: List[PredictedSymbol],
    uncertain: List[PredictedSymbol],
    previous_clef: Optional[str],
    dsl: float,
) -> Tuple[List[PredictedSymbol], Optional[str]]:
    cleaned = _remove_overlaps(baseline, dsl)
    cleaned = _ensure_leading_clef(cleaned, uncertain, previous_clef, dsl)
    tokens: List[PredictedSymbol] = []
    last_delimiter = False
    current_clef = previous_clef
    for symbol in cleaned:
        if not symbol.description:
            continue
        if symbol.label == "delimiter":
            if last_delimiter and tokens:
                tokens[-1] = symbol
            else:
                tokens.append(symbol)
            last_delimiter = True
            continue
        if symbol.label.startswith("clef"):
            current_clef = symbol.description
        last_delimiter = False
        tokens.append(symbol)
    return tokens, current_clef


def _predict_symbols(
    model: nn.Module,
    sample: StaffSample,
    bank: Dict[str, List[Tuple[np.ndarray, str]]],
    device: torch.device,
    image_size: Tuple[int, int] = FIXED_IMAGE_SIZE,
) -> Tuple[List[PredictedSymbol], List[PredictedSymbol]]:
    crop = _load_crop(sample)
    image_arr, sx, sy = _resize_image(crop, image_size)
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
        for label_idx in range(1, len(LABELS)):
            class_mask = label_map == label_idx
            components = _connected_components(class_mask)
            for component in components:
                if len(component) < 3:
                    continue
                min_x, min_y, width, height = _component_bbox(component)
                conf = float(np.mean([confidence[y, x] for y, x in component]))
                cx = min_x + width / 2.0
                cy = min_y + height / 2.0
                orig_bbox = (
                    min_x / sx,
                    min_y / sy,
                    max(1.0, width / sx),
                    max(1.0, height / sy),
                )
                patch = crop.crop(
                    (
                        int(orig_bbox[0]),
                        int(orig_bbox[1]),
                        int(orig_bbox[0] + orig_bbox[2]),
                        int(orig_bbox[1] + orig_bbox[3]),
                    )
                )
                label = INDEX_TO_LABEL[label_idx]
                description, score = _classify_crop(patch, label, bank)
                predictions[source].append(
                    PredictedSymbol(
                        label=label,
                        description=description,
                        bbox=orig_bbox,
                        center=(orig_bbox[0] + orig_bbox[2] / 2.0, orig_bbox[1] + orig_bbox[3] / 2.0),
                        confidence=(conf + score) / 2.0,
                        source=source,
                    )
                )
    return predictions["baseline"], predictions["uncertain"]


def _write_predictions(samples: Sequence[StaffSample], texts: Dict[str, List[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        texts.setdefault(sample.image_stem, [])
    for image_stem, parts in texts.items():
        text = " ".join(part for part in parts if part).strip()
        (output_dir / f"{image_stem}.pred.txt").write_text(text, encoding="utf-8")


def _model_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.is_dir():
        candidate = path / "model.pt"
        return candidate if candidate.exists() else None
    return path if path.exists() else None


def _train_model(
    model: nn.Module,
    train_samples: Sequence[StaffSample],
    val_samples: Sequence[StaffSample],
    save_model_path: Path,
    device: torch.device,
    debug: bool,
) -> None:
    train_loader = DataLoader(BGKStaffDataset(train_samples), batch_size=2 if not debug else 1, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(BGKStaffDataset(val_samples), batch_size=2 if not debug else 1, shuffle=False, collate_fn=_collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")
    epochs = 1 if debug else 8
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
) -> None:
    data_root = Path(args.data_dir) if getattr(args, "data_dir", None) else Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)

    if train_json is None or val_json is None:
        raise ValueError("bgk requires train_json and val_json")

    train_samples = build_staff_samples(Path(train_json), data_root)
    val_samples = build_staff_samples(Path(val_json), data_root)
    test_samples = build_staff_samples(Path(test_json), data_root) if test_json else []
    print(f"🧾 BGK staff samples train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    existing_model = _model_path(Path(load_model_path) if load_model_path else None)
    if existing_model:
        print(f"📦 Loading BGK model from {existing_model}")
        model.load_state_dict(torch.load(existing_model, map_location=device))
    else:
        _train_model(model, train_samples, val_samples, save_model_path, device, getattr(args, "debug", False))

    bank = build_prototype_bank(train_samples)
    page_texts: Dict[str, List[str]] = defaultdict(list)
    previous_clef_by_page: Dict[str, Optional[str]] = defaultdict(lambda: None)
    for sample in test_samples:
        baseline, uncertain = _predict_symbols(model, sample, bank, device)
        tokens, current_clef = _postprocess_staff(baseline, uncertain, previous_clef_by_page[sample.image_stem], sample.dsl)
        previous_clef_by_page[sample.image_stem] = current_clef
        page_texts[sample.image_stem].append(" ".join(token.description for token in tokens if token.description))
    _write_predictions(test_samples, page_texts, output_dir)
