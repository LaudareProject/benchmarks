"""Training and prediction script for a faithful VLT OCR/OMR recognizer."""

from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein as _Lev
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..evaluation import calculate_wer_cer
from ..utils import load_image_stems_from_json, save_text_predictions

IMAGE_HEIGHT = 128
MAX_TARGET_LENGTH = 128
MAX_NEW_TOKENS = 128
TTA_ROUNDS = 60
PREDICTION_FILTER_THRESHOLD = 0.75
AUGMENT_PROBABILITY = 0.2
LOSS_LAMBDA = 0.5
WARMUP_STEPS = 4000
MAX_LR = 0.01
GENERIC_MAX_EPOCHS = 200
STEP2_EPOCHS = 10
ADAPTATION_MAX_EPOCHS = 50
DEFAULT_CONV_CHANNELS = (32, 64, 96, 128, 256)
DEFAULT_NUM_ENCODER_LAYERS = 4
DEFAULT_NUM_DECODER_LAYERS = 2
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_CONV_DROPOUT = 0.1
DEFAULT_FF_DIM = 1024
BLANK_TOKEN = "<ctc_blank>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass
class VLTConfig:
    image_height: int = IMAGE_HEIGHT
    max_target_length: int = MAX_TARGET_LENGTH
    max_new_tokens: int = MAX_NEW_TOKENS
    conv_channels: tuple[int, ...] = DEFAULT_CONV_CHANNELS
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_encoder_layers: int = DEFAULT_NUM_ENCODER_LAYERS
    num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    transformer_dropout: float = DEFAULT_DROPOUT
    conv_dropout: float = DEFAULT_CONV_DROPOUT
    ff_dim: int = DEFAULT_FF_DIM
    loss_lambda: float = LOSS_LAMBDA
    warmup_steps: int = WARMUP_STEPS
    max_lr: float = MAX_LR

    @classmethod
    def from_dict(cls, data: dict) -> "VLTConfig":
        payload = dict(data)
        if "conv_channels" in payload:
            payload["conv_channels"] = tuple(payload["conv_channels"])
        return cls(**payload)


class VLTTokenizer:
    """
    - it defines two vocabularies: seq_tokens and ctc_tokens
    - it uses special IDs for seq decoding: pad/bos/eos/unk
    - it uses a different CTC blank inventory
    - vocab is built directly from dataset characters
   """
    def __init__(self, characters: Iterable[str]):
        chars = sorted({c for c in characters if c})
        self.characters = chars
        self.seq_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + chars
        self.seq_token_to_id = {t: i for i, t in enumerate(self.seq_tokens)}
        self.seq_id_to_token = {i: t for t, i in self.seq_token_to_id.items()}
        self.ctc_tokens = [BLANK_TOKEN, UNK_TOKEN] + chars
        self.ctc_token_to_id = {t: i for i, t in enumerate(self.ctc_tokens)}
        self.pad_token_id = self.seq_token_to_id[PAD_TOKEN]
        self.bos_token_id = self.seq_token_to_id[BOS_TOKEN]
        self.eos_token_id = self.seq_token_to_id[EOS_TOKEN]
        self.unk_token_id = self.seq_token_to_id[UNK_TOKEN]
        self.ctc_blank_id = self.ctc_token_to_id[BLANK_TOKEN]
        self.ctc_unk_id = self.ctc_token_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.seq_tokens)

    @property
    def ctc_vocab_size(self) -> int:
        return len(self.ctc_tokens)

    def encode_seq(self, text: str, max_length: int) -> torch.Tensor:
        ids = [self.bos_token_id] + [self.seq_token_to_id.get(c, self.unk_token_id) for c in text] + [self.eos_token_id]
        ids = ids[:max_length]
        ids += [self.pad_token_id] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def encode_ctc(self, text: str) -> torch.Tensor:
        ids = [self.ctc_token_to_id.get(c, self.ctc_unk_id) for c in text] or [self.ctc_unk_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for idx in ids:
            t = self.seq_id_to_token.get(int(idx), UNK_TOKEN)
            if t in {PAD_TOKEN, BOS_TOKEN}:
                continue
            if t == EOS_TOKEN:
                break
            if t != UNK_TOKEN:
                chars.append(t)
        return "".join(chars)

    def batch_decode(self, batch_ids: Iterable[Iterable[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]

    def save_pretrained(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_text(json.dumps({"characters": self.characters}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "VLTTokenizer":
        return cls(json.loads((Path(path) / "tokenizer.json").read_text(encoding="utf-8"))["characters"])


class VLTProcessor:
    def __init__(self, tokenizer: VLTTokenizer, image_height: int = IMAGE_HEIGHT):
        self.tokenizer = tokenizer
        self.image_height = int(image_height)

    def save_pretrained(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(p)
        (p / "processor_config.json").write_text(json.dumps({"image_height": self.image_height}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "VLTProcessor":
        p = Path(path)
        tokenizer = VLTTokenizer.from_pretrained(p)
        cfg = json.loads((p / "processor_config.json").read_text(encoding="utf-8"))
        return cls(tokenizer=tokenizer, image_height=cfg.get("image_height", IMAGE_HEIGHT))

    @classmethod
    def build_from_jsons(cls, json_paths: Iterable[str | Path], image_height: int = IMAGE_HEIGHT) -> "VLTProcessor":
        characters: set[str] = set()
        for json_path in json_paths:
            if not json_path:
                continue
            p = Path(json_path)
            if not p.exists():
                continue
            for ann in json.loads(p.read_text(encoding="utf-8")).get("annotations", []):
                characters.update((ann.get("description") or ann.get("text") or "").strip())
        return cls(VLTTokenizer(characters), image_height=image_height)


def _random_pad(image: np.ndarray, p: float = AUGMENT_PROBABILITY) -> np.ndarray:
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    return cv2.copyMakeBorder(
        image,
        random.randint(0, max(1, int(0.05 * h))), random.randint(0, max(1, int(0.05 * h))),
        random.randint(0, max(1, int(0.08 * w))), random.randint(0, max(1, int(0.08 * w))),
        borderType=cv2.BORDER_REPLICATE,
    )


class PaperVLTAugment:
    def __init__(self, p: float = AUGMENT_PROBABILITY):
        self._p = p
        self._tf = A.Compose([
            A.OneOf([
                A.Morphological(scale=(2, 3), operation="dilation"),
                A.Morphological(scale=(2, 3), operation="erosion"),
            ], p=p),
            A.ElasticTransform(alpha=4.0, sigma=2.0, p=p),
            A.Perspective(scale=(0.05, 0.1), p=p),
            A.GaussNoise(std_range=(8 / 255, 8 / 255), mean_range=(0, 0), p=p),
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return _random_pad(self._tf(image=image)["image"], p=self._p)


class VLTDataset(Dataset):
    def __init__(
        self,
        json_file: str | Path | None,
        data_dir: str | Path,
        processor: VLTProcessor,
        debug: bool = False,
        augment_transform: PaperVLTAugment | None = None,
        tta_rounds: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.augment_transform = augment_transform
        self.tta_rounds = max(1, int(tta_rounds))

        annotations, image_map = [], {}
        if json_file is not None:
            data = json.loads(Path(json_file).read_text(encoding="utf-8"))
            annotations = data.get("annotations", [])
            image_map = {img["id"]: img for img in data.get("images", [])}
        if debug:
            annotations = annotations[:5]

        self.base_samples: list[dict] = []
        for sid, ann in enumerate(annotations):
            img_info = image_map.get(ann.get("image_id"))
            if img_info is None:
                continue
            self.base_samples.append({
                "sample_id": sid, "ann": ann, "image_info": img_info,
                "image_stem": Path(img_info["file_name"]).stem,
                "text": (ann.get("description") or ann.get("text") or "").strip(),
            })

        if augment_transform is not None and self.tta_rounds > 1:
            self.samples = []
            for s in self.base_samples:
                self.samples.append({**s, "apply_augment": False})
                for _ in range(self.tta_rounds - 1):
                    self.samples.append({**s, "apply_augment": True})
        else:
            flag = augment_transform is not None
            self.samples = [{**s, "apply_augment": flag} for s in self.base_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_crop(self, sample: dict) -> np.ndarray:
        ann, img_info = sample["ann"], sample["image_info"]
        img = cv2.imread(str(self.data_dir / img_info["file_name"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read: {self.data_dir / img_info['file_name']}")
        x, y, w, h = [int(v) for v in ann["bbox"]]
        m = 8
        x, y = max(0, x - m), max(0, y - m)
        w, h = min(img.shape[1] - x, w + 2 * m), min(img.shape[0] - y, h + 2 * m)
        return cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)

    def _resize(self, image: np.ndarray) -> tuple[torch.Tensor, int]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Invalid crop dimensions")
        th = self.processor.image_height
        tw = max(4, int(round(w * th / h)))
        resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0, tw

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = self._load_and_crop(s)
        if self.augment_transform is not None and s.get("apply_augment", False):
            image = self.augment_transform(image)
        pixel_values, image_width = self._resize(image)
        ctc_ids = self.processor.tokenizer.encode_ctc(s["text"])
        return {
            "pixel_values": pixel_values,
            "image_width": image_width,
            "labels": self.processor.tokenizer.encode_seq(s["text"], MAX_TARGET_LENGTH),
            "ctc_targets": ctc_ids,
            "ctc_target_length": len(ctc_ids),
            "sample_id": s["sample_id"],
            "image_stem": s["image_stem"],
            "text": s["text"],
        }


def _pad_image_batch(features: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    widths = torch.tensor([f["pixel_values"].shape[-1] for f in features], dtype=torch.long)
    width_major = [f["pixel_values"].permute(2, 0, 1) for f in features]
    padded = pad_sequence(width_major, batch_first=True)
    return padded.permute(0, 2, 3, 1).contiguous(), widths


class VLTTrainCollator:
    def __init__(self, processor: VLTProcessor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        images, widths = _pad_image_batch(features)
        labels = torch.stack([f["labels"] for f in features]).long()
        decoder_input_ids = labels.roll(1, dims=1)
        decoder_input_ids[:, 0] = self.processor.tokenizer.bos_token_id
        return {
            "pixel_values": images,
            "image_widths": widths,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "ctc_targets": torch.cat([f["ctc_targets"] for f in features]).long(),
            "ctc_target_lengths": torch.tensor([f["ctc_target_length"] for f in features], dtype=torch.long),
        }


class VLTPredictCollator:
    def __call__(self, features: list[dict]) -> dict:
        images, widths = _pad_image_batch(features)
        sample_ids = [f["sample_id"] for f in features]
        stems = [f["image_stem"] for f in features]
        return {
            "pixel_values": images,
            "image_widths": widths,
            "sample_ids": sample_ids,
            "image_stems": stems,
        }

class SpatialLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PaperVLTModel(nn.Module):
    def __init__(self, config: VLTConfig, seq_vocab_size: int, ctc_vocab_size: int,
                 pad_token_id: int, bos_token_id: int, eos_token_id: int, ctc_blank_id: int):
        super().__init__()
        self.config = config
        self.seq_vocab_size = seq_vocab_size
        self.ctc_vocab_size = ctc_vocab_size
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.ctc_blank_id = int(ctc_blank_id)

        conv_layers: list[nn.Module] = []
        in_ch = 3
        for i, out_ch in enumerate(config.conv_channels):
            conv_layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), SpatialLayerNorm(out_ch)]
            if i < 3:
                conv_layers.append(nn.MaxPool2d(2, 2))
            conv_layers.append(nn.Dropout(config.conv_dropout))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*conv_layers)

        enc_layer = nn.TransformerEncoderLayer(config.hidden_size, config.num_heads, config.ff_dim,
                                               config.transformer_dropout, "relu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(config.hidden_size, config.num_heads, config.ff_dim,
                                               config.transformer_dropout, "relu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=config.num_decoder_layers)
        self.token_embedding = nn.Embedding(seq_vocab_size, config.hidden_size)
        self.encoder_positional = SinusoidalPositionalEncoding(config.hidden_size)
        self.decoder_positional = SinusoidalPositionalEncoding(config.hidden_size)
        self.output_dropout = nn.Dropout(config.transformer_dropout)
        self.seq_head = nn.Linear(config.hidden_size, seq_vocab_size)
        self.ctc_head = nn.Linear(config.hidden_size, ctc_vocab_size)

    def save_pretrained(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "model.pt")
        payload = {
            "config": asdict(self.config),
            "seq_vocab_size": self.seq_vocab_size, "ctc_vocab_size": self.ctc_vocab_size,
            "pad_token_id": self.pad_token_id, "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id, "ctc_blank_id": self.ctc_blank_id,
        }
        (p / "model_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "PaperVLTModel":
        p = Path(path)
        d = json.loads((p / "model_config.json").read_text(encoding="utf-8"))
        model = cls(VLTConfig.from_dict(d["config"]), d["seq_vocab_size"], d["ctc_vocab_size"],
                    d["pad_token_id"], d["bos_token_id"], d["eos_token_id"], d["ctc_blank_id"])
        model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
        return model

    @staticmethod
    def reduced_lengths(image_widths: torch.Tensor) -> torch.Tensor:
        return torch.div(image_widths, 8, rounding_mode="floor").clamp_min(1)

    def _encode(self, pixel_values: torch.Tensor, image_widths: torch.Tensor):
        features = self.conv_backbone(pixel_values).mean(dim=2).transpose(1, 2)
        features = self.encoder_positional(features)
        reduced = self.reduced_lengths(image_widths).to(features.device)
        mask = torch.arange(features.size(1), device=features.device).unsqueeze(0) >= reduced.unsqueeze(1)
        return self.encoder(features, src_key_padding_mask=mask), mask, reduced

    def _decode(self, memory: torch.Tensor, enc_mask: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        t = self.output_dropout(self.decoder_positional(
            self.token_embedding(decoder_input_ids) * math.sqrt(self.config.hidden_size)
        ))
        L = decoder_input_ids.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=decoder_input_ids.device), diagonal=1)
        out = self.decoder(t, memory, tgt_mask=causal,
                           tgt_key_padding_mask=decoder_input_ids.eq(self.pad_token_id),
                           memory_key_padding_mask=enc_mask)
        return self.seq_head(out)

    def forward(self, pixel_values, image_widths, decoder_input_ids=None,
                labels=None, ctc_targets=None, ctc_target_lengths=None) -> dict:
        memory, enc_mask, reduced = self._encode(pixel_values, image_widths)
        ctc_logits = self.ctc_head(memory)
        seq_logits = self._decode(memory, enc_mask, decoder_input_ids) if decoder_input_ids is not None else None

        ce_loss = F.cross_entropy(seq_logits.reshape(-1, seq_logits.size(-1)), labels.reshape(-1),
                                  ignore_index=self.pad_token_id) if (labels is not None and seq_logits is not None) else None
        ctc_loss = F.ctc_loss(F.log_softmax(ctc_logits, -1).transpose(0, 1), ctc_targets,
                              reduced.cpu(), ctc_target_lengths.cpu(), blank=self.ctc_blank_id,
                              zero_infinity=True) if ctc_targets is not None else None

        if ce_loss is not None and ctc_loss is not None:
            loss = self.config.loss_lambda * ctc_loss + (1.0 - self.config.loss_lambda) * ce_loss
        else:
            loss = ce_loss if ce_loss is not None else ctc_loss

        return {"loss": loss, "ce_loss": ce_loss, "ctc_loss": ctc_loss,
                "seq_logits": seq_logits, "ctc_logits": ctc_logits}

    def generate(self, pixel_values: torch.Tensor, image_widths: torch.Tensor,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> torch.Tensor:
        memory, enc_mask, _ = self._encode(pixel_values, image_widths)
        generated = torch.full((pixel_values.size(0), 1), self.bos_token_id, dtype=torch.long, device=pixel_values.device)
        finished = torch.zeros(pixel_values.size(0), dtype=torch.bool, device=pixel_values.device)
        for _ in range(max_new_tokens):
            next_tok = self._decode(memory, enc_mask, generated)[:, -1, :].argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.eos_token_id), next_tok)
            generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok.eq(self.eos_token_id)
            if finished.all():
                break
        return generated



MODEL_INPUT_KEYS = ("pixel_values", "image_widths", "decoder_input_ids", "labels", "ctc_targets", "ctc_target_lengths")


def _model_inputs(batch: dict) -> dict:
    return {k: batch[k] for k in MODEL_INPUT_KEYS}

def _filter_predictions(predictions: list[str], τ: float = PREDICTION_FILTER_THRESHOLD) -> list[str]:
    if len(predictions) <= 1:
        return predictions
    avgs = [
        sum(_Lev.normalized_distance(p, o) for j, o in enumerate(predictions) if j != i) / (len(predictions) - 1)
        for i, p in enumerate(predictions)
    ]
    filtered = [(p, d) for p, d in zip(predictions, avgs) if d <= τ] or list(zip(predictions, avgs))
    return [p for p, _ in sorted(filtered, key=lambda x: x[1])]


def _align_pair(ref: str, cand: str) -> tuple[list, list]:
    aligned_ref, aligned_cand, ri, ci = [], [], 0, 0
    for op, src, _ in _Lev.editops(ref, cand):
        while ri < src:
            aligned_ref.append(ref[ri]); aligned_cand.append(cand[ci]); ri += 1; ci += 1
        if op == "replace":
            aligned_ref.append(ref[ri]); aligned_cand.append(cand[ci]); ri += 1; ci += 1
        elif op == "delete":
            aligned_ref.append(ref[ri]); aligned_cand.append(None); ri += 1
        else:
            aligned_ref.append(None); aligned_cand.append(cand[ci]); ci += 1
    while ri < len(ref):
        aligned_ref.append(ref[ri]); aligned_cand.append(cand[ci]); ri += 1; ci += 1
    return aligned_ref, aligned_cand


def _align_predictions(predictions: list[str]) -> list[list[str | None]]:
    if not predictions:
        return []
    center = predictions[0]
    center_length = len(center)
    payloads = []
    max_insertions = [0] * (center_length + 1)

    for pred in predictions:
        a_center, a_pred = _align_pair(center, pred)
        insertions: list[list] = [[] for _ in range(center_length + 1)]
        aligned_chars: list = []
        cpos = 0
        for ct, pt in zip(a_center, a_pred):
            if ct is None:
                insertions[cpos].append(pt)
            else:
                aligned_chars.append(pt); cpos += 1
        payloads.append((insertions, aligned_chars))
        for pos, bucket in enumerate(insertions):
            max_insertions[pos] = max(max_insertions[pos], len(bucket))

    result = []
    for insertions, aligned_chars in payloads:
        expanded: list = []
        for pos in range(center_length + 1):
            expanded.extend(insertions[pos])
            expanded.extend([None] * (max_insertions[pos] - len(insertions[pos])))
            if pos < center_length:
                expanded.append(aligned_chars[pos])
        result.append(expanded)
    return result


def _vote_aligned_predictions(aligned: list[list[str | None]]) -> str:
    if not aligned:
        return ""
    return "".join(
        Counter(t for t in col if t is not None).most_common(1)[0][0]
        for col in zip(*aligned)
        if any(t is not None for t in col)
    )


def _consensus_prediction(candidates: list[str]) -> str:
    predictions = [c.strip() for c in candidates if c.strip()]
    if not predictions:
        return ""
    return _vote_aligned_predictions(_align_predictions(_filter_predictions(predictions)))


def _freeze_decoder(model: PaperVLTModel) -> None:
    for module in (model.decoder, model.token_embedding, model.seq_head):
        for p in module.parameters():
            p.requires_grad = False


def _resolve_batch_size(args) -> int:
    if args.debug or not torch.cuda.is_available():
        return 1
    gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 16 if gb >= 40 else 8 if gb >= 24 else 4 if gb >= 16 else 2


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _evaluate(model: PaperVLTModel, processor: VLTProcessor, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds, refs, total_loss, n = [], [], 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            mb = _move_batch(batch, device)
            out = model(**_model_inputs(mb))
            if out["loss"] is not None:
                total_loss += float(out["loss"]); n += 1
            gen = model.generate(mb["pixel_values"], mb["image_widths"], max_new_tokens=MAX_NEW_TOKENS)
            preds.extend(processor.tokenizer.batch_decode(gen.cpu().tolist()))
            refs.extend(processor.tokenizer.batch_decode(batch["labels"].tolist()))
    wer, cer = calculate_wer_cer(preds, refs)
    return {"loss": total_loss / max(1, n), "wer": wer, "cer": cer}


def _make_loader(args, processor, dataset, shuffle=True, num_workers=None):
    nw = (0 if args.debug else min(4, os.cpu_count() or 0)) if num_workers is None else num_workers
    collator = VLTTrainCollator(processor)
    return DataLoader(dataset, batch_size=_resolve_batch_size(args), shuffle=shuffle, num_workers=nw, collate_fn=collator)


def _run_training_stage(args, model, processor, train_dataset, val_dataset, stage_name: str, max_epochs: int) -> PaperVLTModel:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = _make_loader(args, processor, train_dataset, shuffle=True)
    eval_loader = _make_loader(args, processor, val_dataset, shuffle=False, num_workers=0) if val_dataset and len(val_dataset) else None
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=model.config.max_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: (step + 1) / model.config.warmup_steps if (step + 1) <= model.config.warmup_steps
        else math.exp(-((step + 1) - model.config.warmup_steps) / model.config.warmup_steps),
    )
    best_state, best_cer = None, float("inf")

    print(f"   📘 {stage_name}: {1 if args.debug else max_epochs} epoch(s)")
    for epoch in range(1, (2 if args.debug else max_epochs + 1)):
        model.train()
        total_loss = ce_total = ctc_total = lr = 0.0
        n = 0
        for batch in train_loader:
            mb = _move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(**_model_inputs(mb))
            loss = out["loss"]
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step(); lr = scheduler.get_last_lr()[0]
            total_loss += float(loss)
            ce_total += float(out["ce_loss"]) if out["ce_loss"] is not None else 0.0
            ctc_total += float(out["ctc_loss"]) if out["ctc_loss"] is not None else 0.0
            n += 1
        if n == 0:
            raise RuntimeError(f"VLT {stage_name}: no valid batches")
        print(f"   [{stage_name} {epoch:03d}] loss={total_loss/n:.4f} ce={ce_total/n:.4f} ctc={ctc_total/n:.4f} lr={lr:.6f}")

        if eval_loader is not None:
            m = _evaluate(model, processor, eval_loader, device)
            print(f"   [{stage_name} {epoch:03d}] val_loss={m['loss']:.4f} wer={m['wer']:.4f} cer={m['cer']:.4f}")
            if m["cer"] < best_cer:
                best_cer = m["cer"]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def load_model(model_identifier, load_model_path, processor: VLTProcessor | None = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = Path(load_model_path) if load_model_path and Path(load_model_path).exists() else (
        Path(str(model_identifier)) if model_identifier and Path(str(model_identifier)).exists() else None
    )
    adaptation = bool(load_model_path and Path(load_model_path).exists())
    if ckpt is not None:
        if processor is None:
            processor = VLTProcessor.from_pretrained(ckpt)
        model = PaperVLTModel.from_pretrained(ckpt)
        if adaptation:
            _freeze_decoder(model)
            print(f"   🔒 Decoder frozen for adaptation: {ckpt}")
        else:
            print(f"   Loading checkpoint: {ckpt}")
    else:
        if processor is None:
            raise ValueError("processor required for fresh VLT init")
        cfg = VLTConfig()
        model = PaperVLTModel(cfg, processor.tokenizer.vocab_size, processor.tokenizer.ctc_vocab_size,
                              processor.tokenizer.pad_token_id, processor.tokenizer.bos_token_id,
                              processor.tokenizer.eos_token_id, processor.tokenizer.ctc_blank_id)
        print(f"   Fresh VLT model ({model_identifier})")
    return model.to(device), processor, device


def save_model(model: PaperVLTModel, processor: VLTProcessor, save_model_path) -> None:
    if not save_model_path:
        return
    try:
        p = Path(save_model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(p)
        processor.save_pretrained(p)
        print(f"   💾 Saved: {p}")
    except Exception as exc:
        print(f"   ⚠️  Save failed: {exc}")


def predict(args, model: PaperVLTModel, processor: VLTProcessor, output_dir: Path, test_json) -> None:
    tta = 1 if args.debug else TTA_ROUNDS
    dataset = VLTDataset(test_json, args.data_dir or args.test_dir, processor, debug=args.debug,
                         augment_transform=PaperVLTAugment() if tta > 1 else None, tta_rounds=tta)
    loader = DataLoader(dataset, batch_size=_resolve_batch_size(args), shuffle=False,
                        num_workers=0, collate_fn=VLTPredictCollator())
    expected_stems = load_image_stems_from_json(Path(test_json))
    line_votes: dict[int, list[str]] = defaultdict(list)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            sids, stems = batch.pop("sample_ids"), batch.pop("image_stems")
            gen = model.generate(**_move_batch(batch, device), max_new_tokens=MAX_NEW_TOKENS)
            for sid, stem, pred in zip(sids, stems, processor.tokenizer.batch_decode(gen.cpu().tolist())):
                line_votes[sid].append(pred.strip())
                if args.debug:
                    print(f"   {stem}: {pred[:80]}")
    line_preds = {s["sample_id"]: _consensus_prediction(line_votes.get(s["sample_id"], [])) for s in dataset.base_samples}
    page_preds: dict[str, list[str]] = defaultdict(list)
    for s in dataset.base_samples:
        page_preds[s["image_stem"]].append(line_preds.get(s["sample_id"], ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem in expected_stems:
        save_text_predictions(stem, " ".join(t for t in page_preds.get(stem, []) if t), output_dir)
    print(f"✅ Predictions saved to {output_dir}")


def train_test_vlt(args, is_train_test_mode, is_sequential, output_dir,
                   train_json, val_json, test_json, save_model_path, load_model_path, model_identifier):
    if args.task not in {"ocr", "omr"}:
        raise ValueError("vlt supports only task=ocr|omr")

    adaptation_mode = bool(load_model_path and Path(load_model_path).exists())
    processor = None
    if not adaptation_mode and not (model_identifier and Path(str(model_identifier)).exists()):
        processor = VLTProcessor.build_from_jsons([train_json, val_json])

    print(f"📦 Loading VLT ({model_identifier})...")
    model, processor, device = load_model(model_identifier, load_model_path, processor=processor)
    aug = PaperVLTAugment()
    train_ds = VLTDataset(train_json, args.data_dir or args.train_dir, processor, debug=args.debug, augment_transform=aug)
    val_ds = VLTDataset(val_json, args.data_dir or args.train_dir, processor, debug=args.debug)

    print(f"🚀 Training on {device}...")
    if adaptation_mode:
        model = _run_training_stage(args, model, processor, train_ds, val_ds, "step3_adaptation", ADAPTATION_MAX_EPOCHS)
    else:
        model = _run_training_stage(args, model, processor, train_ds, val_ds, "step1_generic", GENERIC_MAX_EPOCHS)
        step2_ds = ConcatDataset([train_ds, VLTDataset(val_json, args.data_dir or args.train_dir, processor,
                                                        debug=args.debug, augment_transform=aug)])
        model = _run_training_stage(args, model, processor, step2_ds, None, "step2_train+val", STEP2_EPOCHS)

    save_model(model, processor, save_model_path)
    if test_json is not None:
        print("✍️  Generating predictions...")
        predict(args, model, processor, output_dir, test_json)
    print(f"✅ VLT complete: {output_dir}")
