# ultralytics_monkey_patch.py
"""
Monkey patch for ultralytics to add TrivialAugment support to classify_augmentations function.
"""

import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import (
    TORCHVISION_0_10,
    TORCHVISION_0_11,
    TORCHVISION_0_13,
)


# this function is a copy of the original classify_augmentations function
# just with the addition of TrivialAugment support
def patched_classify_augmentations(
    size: int = 224,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scale: tuple[float, float] | None = None,
    ratio: tuple[float, float] | None = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    auto_augment: str | None = None,
    hsv_h: float = 0.015,
    hsv_s: float = 0.4,
    hsv_v: float = 0.4,
    force_color_jitter: bool = False,
    erasing: float = 0.0,
    interpolation: str = "BILINEAR",
):
    """
    Patched version of classify_augmentations that includes TrivialAugment support.
    """
    import torchvision.transforms as T

    if not isinstance(size, int):
        raise TypeError(
            f"classify_augmentations() size {size} must be integer, not (list, tuple)"
        )
    scale = tuple(scale or (0.08, 1.0))
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))
    interpolation = getattr(T.InterpolationMode, interpolation)
    primary_tfl = [
        T.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation)
    ]
    if hflip > 0.0:
        primary_tfl.append(T.RandomHorizontalFlip(p=hflip))
    if vflip > 0.0:
        primary_tfl.append(T.RandomVerticalFlip(p=vflip))

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str), (
            f"Provided argument should be string, but got type {type(auto_augment)}"
        )
        disable_color_jitter = not force_color_jitter

        if auto_augment == "randaugment":
            if TORCHVISION_0_11:
                secondary_tfl.append(T.RandAugment(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=randaugment" requires torchvision >= 0.11.0. Disabling it.'
                )

        elif auto_augment == "augmix":
            if TORCHVISION_0_13:
                secondary_tfl.append(T.AugMix(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=augmix" requires torchvision >= 0.13.0. Disabling it.'
                )

        elif auto_augment == "autoaugment":
            if TORCHVISION_0_10:
                secondary_tfl.append(T.AutoAugment(interpolation=interpolation))
            else:
                LOGGER.warning(
                    '"auto_augment=autoaugment" requires torchvision >= 0.10.0. Disabling it.'
                )

        elif auto_augment == "trivialaugment":  # Add TrivialAugment support
            if TORCHVISION_0_11:  # TrivialAugment was introduced in torchvision 0.12.0
                try:
                    secondary_tfl.append(
                        T.TrivialAugmentWide(interpolation=interpolation)
                    )
                except AttributeError:
                    LOGGER.warning(
                        '"auto_augment=trivialaugment" requires torchvision >= 0.12.0. Disabling it.'
                    )
            else:
                LOGGER.warning(
                    '"auto_augment=trivialaugment" requires torchvision >= 0.12.0. Disabling it.'
                )

        else:
            raise ValueError(
                f'Invalid auto_augment policy: {auto_augment}. Should be one of "randaugment", '
                f'"augmix", "autoaugment", "trivialaugment" or None'
            )

    if not disable_color_jitter:
        secondary_tfl.append(
            T.ColorJitter(brightness=hsv_v, contrast=hsv_v, saturation=hsv_s, hue=hsv_h)
        )

    final_tfl = [
        T.ToTensor(),
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        T.RandomErasing(p=erasing, inplace=True),
    ]

    return T.Compose(primary_tfl + secondary_tfl + final_tfl)


def apply_ultralytics_monkey_patch():
    """Apply the monkey patch to ultralytics."""
    try:
        from ultralytics.data import augment

        augment.classify_augmentations = patched_classify_augmentations
        print(
            "✅ Successfully applied ultralytics monkey patch for TrivialAugment support"
        )
    except ImportError as e:
        print(f"❌ Failed to apply ultralytics monkey patch: {e}")

