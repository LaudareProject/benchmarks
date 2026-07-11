"""
Zero-shot PaddleOCR-VL adapter for OCR and OMR benchmarks.
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPTS = {
    "ocr": "OCR:",
    "omr": "OMR:",
}
MAX_PIXELS = 1280 * 28 * 28
UPSCALE_THRESHOLD = 1500


def _get_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load_model(model_identifier: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype()

    print(f"   Loading model: {model_identifier}")
    print(f"   Device: {device}, dtype: {dtype}")

    processor = AutoProcessor.from_pretrained(model_identifier, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_identifier,
        dtype=dtype,
    ).to(device)
    model.eval()
    return model, processor, device


def _load_annotations(json_path: Path, debug: bool):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"][:5] if debug else data["annotations"]
    image_map = {image["id"]: image for image in data["images"]}
    return annotations, image_map


def _crop_line_image(data_dir: Path, image_info: dict, ann: dict) -> Image.Image:
    image_path = data_dir / image_info["file_name"]
    full_image = cv2.imread(str(image_path))
    if full_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x, y, w, h = [int(value) for value in ann["bbox"]]
    margin = 8
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(full_image.shape[1] - x, w + 2 * margin)
    h = min(full_image.shape[0] - y, h + 2 * margin)

    line_image = full_image[y : y + h, x : x + w]
    line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(line_image)

    if image.width < UPSCALE_THRESHOLD and image.height < UPSCALE_THRESHOLD:
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)

    return image


def _prepare_inputs(processor, image: Image.Image, device: torch.device, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": MAX_PIXELS,
            }
        },
    )

    for key, value in list(inputs.items()):
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        inputs[key] = value.to(device)

    return inputs


def _decode_output(processor, outputs: torch.Tensor, prompt_length: int) -> str:
    generated = outputs[0][prompt_length:]
    if len(generated) > 0 and generated[-1].item() == processor.tokenizer.eos_token_id:
        generated = generated[:-1]
    return processor.decode(generated, skip_special_tokens=True).strip()


def predict(args, model, processor, device, output_dir, test_json):
    annotations, image_map = _load_annotations(Path(test_json), args.debug)
    image_predictions = defaultdict(list)
    data_dir = args.data_dir or args.test_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPTS.get(args.task, f"{args.task.upper()}:")

    for index, ann in enumerate(annotations, start=1):
        image_info = image_map[ann["image_id"]]
        image = _crop_line_image(data_dir, image_info, ann)
        inputs = _prepare_inputs(processor, image, device, prompt)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)

        prediction = _decode_output(processor, outputs, inputs["input_ids"].shape[-1])
        image_name = Path(image_info["file_name"]).stem
        image_predictions[image_name].append(prediction)

        if args.debug:
            print(f"   [{index}/{len(annotations)}] {image_name}: {prediction[:80]}")

    for image_name, lines in image_predictions.items():
        output_file = output_dir / f"{image_name}.pred.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(lines))

    print(f"✅ Predictions saved to {output_dir}")


def train_test_paddleocr_vl(
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
        raise ValueError("paddleocr_vl supports only task=ocr|omr")

    print(f"⚠️  Zero-shot adapter: training is skipped for task={args.task}.")

    if test_json is None:
        print("⏭️  No test split provided; nothing to run.")
        return

    model, processor, device = _load_model(model_identifier)
    predict(args, model, processor, device, output_dir, test_json)
