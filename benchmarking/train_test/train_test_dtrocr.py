from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..evaluation import calculate_wer_cer
from ..utils import load_image_stems_from_json, save_text_predictions

try:
    from dtrocr.config import DTrOCRConfig
    from dtrocr.data import DTrOCRProcessorOutput
    from dtrocr.model import DTrOCRLMHeadModel
    from dtrocr.processor import DTrOCRProcessor
except ImportError as exc:
    raise ImportError(
        "DTrOCR is not installed. Run `uv sync` and execute benchmarks via `uv run` so the pinned git dependency is available."
    ) from exc


MAX_TARGET_LENGTH = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 10
DEFAULT_LR = 5.0e-5
DEFAULT_WEIGHT_DECAY = 1.0e-4


def _encode_sample(processor, image: Image.Image, text: str, max_target_length: int) -> dict[str, torch.Tensor]:
    encoded = processor.encode_sample(image, text, max_target_length)
    return {
        "pixel_values": encoded.pixel_values,
        "input_ids": encoded.input_ids.long(),
        "attention_mask": encoded.attention_mask.long(),
        "labels": encoded.labels.long(),
    }


class DTrOCRDataset(Dataset):
    def __init__(self, json_file, data_dir, processor, max_target_length=MAX_TARGET_LENGTH, debug=False):
        self.json_file = Path(json_file) if json_file else None
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_target_length = max_target_length
        self.debug = debug
        if self.json_file is not None:
            data = json.loads(self.json_file.read_text(encoding="utf-8"))
            self.annotations = data["annotations"]
            self.image_id_to_info = {img["id"]: img for img in data["images"]}
        else:
            self.annotations = []
            self.image_id_to_info = {}
        if self.debug:
            self.annotations = self.annotations[:5]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        text = ann.get("description") or ann.get("text") or ""
        image_info = self.image_id_to_info[ann["image_id"]]
        image_path = self.data_dir / image_info["file_name"]
        full_image = cv2.imread(str(image_path))
        if full_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        x, y, w, h = [int(c) for c in ann["bbox"]]
        margin = 8
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(full_image.shape[1] - x, w + 2 * margin)
        h = min(full_image.shape[0] - y, h + 2 * margin)
        line_image = cv2.cvtColor(full_image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
        return _encode_sample(
            self.processor,
            Image.fromarray(line_image),
            text,
            self.max_target_length,
        )


class DTrOCRTrainer:
    def __init__(self, model):
        self.model = model

    def save_model(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        (path / "model_config.json").write_text(
            json.dumps(self.model.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _collate_batch(features):
    return {key: torch.stack([feature[key] for feature in features]) for key in features[0]}


def _resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_batch_size(args) -> int:
    if getattr(args, "debug", False):
        return 2
    return DEFAULT_BATCH_SIZE if torch.cuda.is_available() else 1


def _resolve_epochs(args, adaptation_mode: bool) -> int:
    base = 1 if getattr(args, "debug", False) else DEFAULT_EPOCHS
    return max(1, base // 2) if adaptation_mode else base


def _generation_inputs(moved: dict[str, torch.Tensor], processor) -> DTrOCRProcessorOutput:
    return processor.build_generation_inputs(moved["pixel_values"])


def _evaluate(model, processor, dataset, device, batch_size) -> tuple[float, float, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_batch)
    losses: list[float] = []
    references: list[str] = []
    predictions: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            moved = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**moved)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu()))
            generated = model.generate(
                inputs=_generation_inputs(moved, processor),
                processor=processor,
                num_beams=4,
            )
            predictions.extend(processor.tokeniser.batch_decode(generated.detach().cpu(), skip_special_tokens=True))
            references.extend(processor.tokeniser.batch_decode(batch["labels"], skip_special_tokens=True))
    if not references:
        return float("inf"), float("inf"), float("inf")
    wer, cer = calculate_wer_cer(predictions, references)
    avg_loss = sum(losses) / max(len(losses), 1)
    return avg_loss, wer, cer


def load_model(model_identifier, load_model_path):
    device = _resolve_device()
    if load_model_path and Path(load_model_path).exists():
        model_to_load = Path(load_model_path)
        print(f"   Fine-tuning from checkpoint: {model_to_load}")
        payload = json.loads((model_to_load / "model_config.json").read_text(encoding="utf-8"))
        config = DTrOCRConfig.from_dict(payload)
        processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)
        model = DTrOCRLMHeadModel(config)
        model.load_state_dict(torch.load(model_to_load / "model.pt", map_location="cpu"))
    else:
        print(f"   Loading base model preset: {model_identifier}")
        config = DTrOCRConfig()
        processor = DTrOCRProcessor(config, add_bos_token=True, add_eos_token=True)
        model = DTrOCRLMHeadModel(config)
    model.to(device)
    return model, processor, device


def train(args, model, processor, train_dataset, val_dataset, adaptation_mode: bool):
    artifacts_path = Path(tempfile.mkdtemp(prefix="dtrocr_artifacts_"))
    device = next(model.parameters()).device
    batch_size = _resolve_batch_size(args)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate_batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY)
    best_state_path = artifacts_path / "best_model.pt"
    best_metric = float("inf")

    for epoch in range(_resolve_epochs(args, adaptation_mode)):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            moved = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**moved)
            if outputs.loss is None:
                continue
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(outputs.loss.detach().cpu())
        val_loss, wer, cer = _evaluate(model, processor, val_dataset, device, batch_size)
        metric = cer if cer != float("inf") else val_loss
        if metric < best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_state_path)
        print(f"   Epoch {epoch + 1}/{_resolve_epochs(args, adaptation_mode)} - train_loss={running_loss / max(len(dataloader), 1):.4f} val_loss={val_loss:.4f} val_wer={wer:.4f} val_cer={cer:.4f}")

    if best_state_path.exists():
        model.load_state_dict(torch.load(best_state_path, map_location=device))
    return DTrOCRTrainer(model), artifacts_path


def save_model(trainer, save_model_path):
    if not save_model_path:
        return
    try:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(save_model_path)
        print(f"   💾 Saved DTrOCR checkpoint to standardized path: {save_model_path}")
    except Exception as exc:
        print(f"   ⚠️  Failed to save model to standardized path: {exc}")


def predict(args, model, processor, output_dir: Path, test_json):
    dataset = DTrOCRDataset(test_json, args.data_dir or args.test_dir, processor, debug=args.debug)
    dataloader = DataLoader(dataset, batch_size=_resolve_batch_size(args), shuffle=False, num_workers=0, collate_fn=_collate_batch)
    expected_stems = load_image_stems_from_json(Path(test_json))
    page_predictions: dict[str, list[str]] = {stem: [] for stem in expected_stems}
    model.eval()
    device = next(model.parameters()).device
    offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            moved = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(
                inputs=_generation_inputs(moved, processor),
                processor=processor,
                num_beams=4,
            )
            pred_str = processor.tokeniser.batch_decode(generated.detach().cpu(), skip_special_tokens=True)
            for i, prediction in enumerate(pred_str):
                ann = dataset.annotations[offset + i]
                image_info = dataset.image_id_to_info[ann["image_id"]]
                image_stem = Path(image_info["file_name"]).stem
                page_predictions.setdefault(image_stem, []).append(prediction.strip())
            offset += len(pred_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_stem in expected_stems:
        save_text_predictions(image_stem, " ".join(token for token in page_predictions.get(image_stem, []) if token), output_dir)
    print(f"✅ Predictions saved to {output_dir}")


def train_test_dtrocr(args, is_train_test_mode, is_sequential, output_dir, train_json, val_json, test_json, save_model_path, load_model_path, model_identifier):
    if args.task not in {"ocr", "omr"}:
        raise ValueError("dtrocr supports only task=ocr|omr")
    adaptation_mode = bool(load_model_path and Path(load_model_path).exists())
    print(f"📦 Loading DTrOCR model ({model_identifier})...")
    model, processor, device = load_model(model_identifier, load_model_path)
    train_dataset = DTrOCRDataset(train_json, args.data_dir or args.train_dir, processor, debug=args.debug)
    val_dataset = DTrOCRDataset(val_json, args.data_dir or args.train_dir, processor, debug=args.debug)
    print(f"🚀 Training on {device}...")
    trainer, artifacts_path = train(args, model, processor, train_dataset, val_dataset, adaptation_mode=adaptation_mode)
    save_model(trainer, save_model_path)
    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(args, trainer.model, processor, output_dir, test_json)
    else:
        print("⏭️  No test split provided; skipping prediction.")
    shutil.rmtree(artifacts_path, ignore_errors=True)
