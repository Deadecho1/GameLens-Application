"""
trainer.py — LoRA fine-tuning logic for game-specific X-CLIP adaptation.

Adapts the base roguelike event detector to a new game using a small set of
labelled 2-second clips (~4–8 per class).  Only the LoRA adapters on the
vision encoder attention layers and the MIT (Multiframe Integration
Transformer) are trained; the text encoder is kept fully frozen.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image, ImageEnhance
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from transformers import XCLIPModel, XCLIPProcessor

from app_core.logging import get_logger

logger = get_logger(__name__)

# ── Labels (mirrors scripts/event_detector/labels.py) ────────────────────────

LABEL_START  = "the roguelike gameplay or run starts after menu or loading screen"
LABEL_END    = "the moment the player character dies and the roguelike run ends"
LABEL_DROP   = "a reward screen shows an item drop or chest reward"
LABEL_CHOICE = "the moment the player selects a choice from a choice UI screen that shows multiple upgrade or item options"
NONE_LABEL   = "regular gameplay with no reward screen, no choice screen, no defeat screen, not player death, and no run start transition"

LABEL_TEXTS = [LABEL_START, LABEL_END, LABEL_DROP, LABEL_CHOICE, NONE_LABEL]
LABEL_NAMES = ["start", "end", "drop", "choice", "none"]
FOLDER_TO_IDX = {"start": 0, "end": 1, "drop": 2, "choice": 3, "none": 4}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Clip:
    path: Path
    label_idx: int
    label_text: str


def discover_clips(clips_dir: Path) -> list[Clip]:
    """Scan clips_dir for MP4 files in named subfolders."""
    clips: list[Clip] = []
    for folder_name, idx in FOLDER_TO_IDX.items():
        folder = clips_dir / folder_name
        if not folder.is_dir():
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            clips.append(Clip(path=mp4, label_idx=idx, label_text=LABEL_TEXTS[idx]))
    if not clips:
        raise ValueError(
            f"No clips found in {clips_dir}. "
            f"Expected subfolders: start/, end/, drop/, choice/, none/"
        )
    return clips


# ── Video loading + augmentation ──────────────────────────────────────────────

def load_video_frames(path: Path, num_frames: int, offset_ratio: float = 0.0) -> list:
    """
    Load num_frames uniformly sampled frames from a video.
    offset_ratio shifts the sampling window by that fraction of total frames
    (use small values like ±0.15 for temporal augmentation).
    """
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Video has no frames: {path}")

    shift = int(total * offset_ratio)
    start_idx = max(0, shift)
    end_idx = min(total - 1, total - 1 + shift)

    indices = torch.linspace(start_idx, end_idx, steps=num_frames)
    indices = indices.round().long().clamp(0, total - 1)

    frames = vr.get_batch(indices.tolist()).asnumpy()
    return [frames[i] for i in range(frames.shape[0])]


def augment_frames(frames: list) -> list:
    """
    Apply consistent colour jitter and random horizontal flip across all frames.
    Factors are sampled once per clip to preserve temporal coherence.
    """
    b = 1.0 + random.uniform(-0.2, 0.2)
    c = 1.0 + random.uniform(-0.2, 0.2)
    s = 1.0 + random.uniform(-0.2, 0.2)
    do_flip = random.random() < 0.5

    result = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Color(img).enhance(s)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        result.append(np.array(img))
    return result


# ── Dataset ───────────────────────────────────────────────────────────────────

class FewShotVideoDataset(Dataset):
    """
    Few-shot video dataset with optional augmentation.
    aug_multiplier controls how many versions are generated per clip.
    Version 0 is always the clean centre crop; versions 1+ apply jitter + temporal shift.
    """

    def __init__(
        self,
        clips: list[Clip],
        processor: XCLIPProcessor,
        num_frames: int,
        augment: bool = True,
        aug_multiplier: int = 4,
    ):
        self.clips = clips
        self.processor = processor
        self.num_frames = num_frames
        self.augment = augment
        self.aug_multiplier = aug_multiplier if augment else 1

    def __len__(self) -> int:
        return len(self.clips) * self.aug_multiplier

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx // self.aug_multiplier]
        aug_version = idx % self.aug_multiplier

        offset = random.uniform(-0.15, 0.15) if (self.augment and aug_version > 0) else 0.0
        frames = load_video_frames(clip.path, self.num_frames, offset_ratio=offset)

        if self.augment and aug_version > 0:
            frames = augment_frames(frames)

        pixel_values = self.processor.image_processor(
            frames, return_tensors="pt"
        )["pixel_values"][0]

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(clip.label_idx, dtype=torch.long),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prepare_text_inputs(processor: XCLIPProcessor, device: str) -> dict:
    text_inputs = processor.tokenizer(
        LABEL_TEXTS,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in text_inputs.items()}


def _score_batch(model, pixel_values: torch.Tensor, text_inputs: dict) -> torch.Tensor:
    """Return (B, 5) classification logits via full X-CLIP forward pass."""
    outputs = model(
        pixel_values=pixel_values,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        return_loss=False,
    )
    return outputs.logits_per_video


@torch.no_grad()
def _compute_val_loss(
    model,
    processor: XCLIPProcessor,
    val_clips: list[Clip],
    text_inputs: dict,
    device: str,
    num_frames: int,
) -> float:
    model.eval()
    total = 0.0
    for clip in val_clips:
        frames = load_video_frames(clip.path, num_frames)
        pixel_values = processor.image_processor(
            frames, return_tensors="pt"
        )["pixel_values"].to(device)
        label = torch.tensor([clip.label_idx], dtype=torch.long, device=device)
        logits = _score_batch(model, pixel_values, text_inputs)
        total += F.cross_entropy(logits, label).item()
    return total / len(val_clips)


# ── Public API ────────────────────────────────────────────────────────────────

def run_finetuning(
    clips_dir: Path,
    base_model_dir: Path,
    output_dir: Path,
    lora_rank: int = 2,
    epochs: int = 50,
    seed: int = 42,
) -> None:
    """
    Fine-tune the X-CLIP event detector on a small set of game-specific clips
    and save the merged model to output_dir.

    The caller is responsible for setting up logging before calling this function.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError(
            "'peft' is required for fine-tuning. Install it with: pip install 'peft>=0.13'"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Discover clips ────────────────────────────────────────────────────────
    logger.info("Stage: discovering clips")
    clips = discover_clips(clips_dir)
    per_class = {name: sum(1 for c in clips if c.label_idx == i) for i, name in enumerate(LABEL_NAMES)}
    counts_str = "  ".join(f"{n}={v}" for n, v in per_class.items() if v > 0)
    logger.info("Found %d clips: %s", len(clips), counts_str)

    # ── Load base model ───────────────────────────────────────────────────────
    logger.info("Stage: loading base model from %s", base_model_dir)
    processor = XCLIPProcessor.from_pretrained(str(base_model_dir))
    model = XCLIPModel.from_pretrained(str(base_model_dir))
    num_frames = model.config.vision_config.num_frames

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Freeze LoRA params in the text encoder; unfreeze MIT and projections
    for name, param in model.named_parameters():
        if "lora_" in name and "text_model" in name:
            param.requires_grad = False
    for name, param in model.named_parameters():
        if ".mit." in name:
            param.requires_grad = True
    for name, param in model.named_parameters():
        if "visual_projection" in name or "text_projection" in name or "logit_scale" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.debug("Trainable parameters: %s / %s (%.2f%%)", f"{trainable:,}", f"{total_p:,}", 100 * trainable / total_p)

    model = model.to(device)

    # ── Train / val split ─────────────────────────────────────────────────────
    val_clips: list[Clip] = []
    train_clips: list[Clip] = []
    for label_idx in range(5):
        class_clips = [c for c in clips if c.label_idx == label_idx]
        if len(class_clips) >= 4:
            val_clips.append(class_clips[0])
            train_clips.extend(class_clips[1:])
        else:
            train_clips.extend(class_clips)

    aug_multiplier = max(1, min(8, (32 + len(train_clips) - 1) // max(len(train_clips), 1)))
    logger.debug(
        "Train clips: %d × %d aug = %d  |  Val clips: %d",
        len(train_clips), aug_multiplier, len(train_clips) * aug_multiplier, len(val_clips),
    )

    train_dataset = FewShotVideoDataset(
        train_clips, processor, num_frames, augment=True, aug_multiplier=aug_multiplier
    )
    batch_size = min(8, max(1, len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    text_inputs = _prepare_text_inputs(processor, device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.1,
    )
    total_steps = epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience = 10
    patience_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Stage: training")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            label_indices = batch["label"].to(device)

            optimizer.zero_grad()
            logits = _score_batch(model, pixel_values, text_inputs)
            loss = F.cross_entropy(logits, label_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if val_clips:
            val_loss = _compute_val_loss(model, processor, val_clips, text_inputs, device, num_frames)
            logger.info("Epoch %3d/%d  train=%.4f  val=%.4f", epoch, epochs, avg_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                    break
        else:
            logger.info("Epoch %3d/%d  train=%.4f", epoch, epochs, avg_loss)
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ── Merge LoRA and save ───────────────────────────────────────────────────
    logger.info("Stage: saving model")
    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Done. Model saved to %s", output_dir)
