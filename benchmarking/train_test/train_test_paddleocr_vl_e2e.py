"""Full-page PaddleOCR-VL adapter for end-to-end OCR and OMR benchmarks.

Unlike ``paddleocr_vl``, this adapter presents each complete page to the model.
Ground-truth region transcriptions are joined in annotation order, matching the
page-level contract used by ``evaluation.load_ground_truth_from_json``.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils import load_image_stems_from_json, save_text_predictions
from .train_test_paddleocr_vl import (
    PaddleOCRVLCollator,
    PROMPTS,
    _load_annotations,
    _move_batch_to_device,
    _prepare_inputs,
    load_model,
    save_model,
    train,
)


MAX_PAGE_NEW_TOKENS = 2048


class PaddleOCRVLE2ETrainDataset(Dataset):
    """One full-page supervised sample per page with usable transcription."""

    def __init__(self, json_file, data_dir, processor, task: str, debug: bool = False):
        self.json_file = Path(json_file)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.prompt = PROMPTS[task]

        annotations, image_map = _load_annotations(self.json_file, debug)
        texts_by_image = defaultdict(list)
        for annotation in annotations:
            text = (annotation.get("description") or annotation.get("text") or "").strip()
            image_id = annotation.get("image_id")
            if image_id is None:
                continue
            image_info = image_map.get(image_id)
            if text and image_info is not None:
                texts_by_image[image_id].append(text)

        self.samples = [
            (image_map[image_id], " ".join(texts))
            for image_id, texts in texts_by_image.items()
        ]
        if debug:
            self.samples = self.samples[:5]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_info, target_text = self.samples[idx]
        image_path = self.data_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        prompt_inputs = _prepare_inputs(self.processor, image, self.prompt)
        full_inputs = _prepare_inputs(
            self.processor, image, self.prompt, target_text=target_text
        )

        prompt_length = prompt_inputs["input_ids"].shape[-1]
        labels = full_inputs["input_ids"].squeeze(0).clone().long()
        labels[:prompt_length] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return {
            "input_ids": full_inputs["input_ids"].squeeze(0).long(),
            "attention_mask": full_inputs["attention_mask"].squeeze(0).long(),
            "mm_token_type_ids": full_inputs["mm_token_type_ids"].squeeze(0).long(),
            "pixel_values": full_inputs["pixel_values"].float(),
            "image_grid_thw": full_inputs["image_grid_thw"].reshape(-1, 3).long(),
            "labels": labels,
        }


class PaddleOCRVLE2EPredictDataset(Dataset):
    """Full-page inputs, including pages without task annotations."""

    def __init__(self, json_file, data_dir, processor, task: str, debug: bool = False):
        self.json_file = Path(json_file)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.prompt = PROMPTS[task]
        with self.json_file.open(encoding="utf-8") as source:
            data = json.load(source)
        self.images = data.get("images", [])
        if debug:
            self.images = self.images[:5]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = self.data_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        inputs = _prepare_inputs(self.processor, image, self.prompt)
        return {
            "input_ids": inputs["input_ids"].squeeze(0).long(),
            "attention_mask": inputs["attention_mask"].squeeze(0).long(),
            "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0).long(),
            "pixel_values": inputs["pixel_values"].float(),
            "image_grid_thw": inputs["image_grid_thw"].reshape(-1, 3).long(),
            "image_stem": Path(image_info["file_name"]).stem,
        }


def predict(args, model, processor, device, output_dir, test_json):
    dataset = PaddleOCRVLE2EPredictDataset(
        test_json,
        args.data_dir or args.test_dir,
        processor,
        task=args.task,
        debug=args.debug,
    )
    collator = PaddleOCRVLCollator(
        pad_token_id=processor.tokenizer.pad_token_id,
        with_labels=False,
        pad_side="left",
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    predictions = {}

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader, start=1):
            image_stems = batch.pop("image_stems")
            batch = _move_batch_to_device(batch, device)
            context_length = batch["input_ids"].shape[1]
            outputs = model.generate(**batch, max_new_tokens=MAX_PAGE_NEW_TOKENS)
            for image_stem, output_ids in zip(image_stems, outputs):
                generated = output_ids[context_length:]
                predictions[image_stem] = processor.decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if args.debug:
                    print(f"   [page {batch_idx}] {image_stem}: {predictions[image_stem][:80]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_stems = load_image_stems_from_json(Path(test_json))
    if args.debug:
        expected_stems = expected_stems[:5]
    for image_stem in expected_stems:
        save_text_predictions(image_stem, predictions.get(image_stem, ""), output_dir)
    print(f"✅ Full-page predictions saved to {output_dir}")


def train_test_paddleocr_vl_e2e(
    args,
    is_train_test_mode,
    is_sequential,
    output_dir,
    train_json,
    val_json,
    test_json,
    save_model_path,
    load_model_path,
    model_identifier,
):
    if args.task not in {"ocr", "omr"}:
        raise ValueError("paddleocr_vl_e2e supports only task=ocr|omr")
    if model_identifier is None:
        raise ValueError("paddleocr_vl_e2e requires a configured base model")

    print(f"📦 Loading PaddleOCR-VL full-page model ({model_identifier})...")
    model, processor, device = load_model(model_identifier, load_model_path)
    data_root = args.data_dir or args.train_dir
    train_dataset = PaddleOCRVLE2ETrainDataset(
        train_json, data_root, processor, args.task, args.debug
    )
    val_dataset = PaddleOCRVLE2ETrainDataset(
        val_json, data_root, processor, args.task, args.debug
    )
    if not train_dataset:
        raise ValueError(f"No usable page-level training samples in {train_json}")

    trainer, artifacts_path = train(
        args, model, processor, train_dataset, val_dataset
    )
    save_model(trainer, processor, save_model_path)
    if test_json is not None:
        predict(args, trainer.model, processor, device, output_dir, test_json)
    else:
        print("⏭️  No test split provided; skipping prediction.")

    if artifacts_path.exists():
        import shutil

        shutil.rmtree(artifacts_path)
