"""Reusable rendering and annotation helpers for synthetic OCR/OMR samples."""

import json
import os
import random as rg
from pathlib import Path

import cv2
import augraphy as au
import numpy as np
from PIL import Image, ImageDraw, ImageFont

rg = rg.Random(2025)


def get_texture(width, height, texture_pool):
    """Get a random texture from the pre-generated pool and resize it if needed"""
    texture = rg.choice(texture_pool)

    # Callers draw directly on the returned texture; don't mutate the shared pool.
    if texture.width != width or texture.height != height:
        texture = texture.resize((width, height), Image.Resampling.LANCZOS)
    else:
        texture = texture.copy()

    return texture


def _render_text_ink(text, font, capital_font=None):
    """Render tightly bounded black ink, including an optional decorative capital."""
    parts = [(text, font)]
    if capital_font is not None and len(text) > 1:
        parts = [(text[0].upper(), capital_font), (text[1:], font)]
    bounds = [part_font.getbbox(part) for part, part_font in parts]
    widths = [box[2] - box[0] for box in bounds]
    heights = [box[3] - box[1] for box in bounds]
    gap = 5 if len(parts) > 1 else 0
    image = Image.new("RGBA", (max(1, sum(widths) + gap), max(1, max(heights))))
    draw = ImageDraw.Draw(image)
    x = 0
    for (part, part_font), box, part_width, part_height in zip(
        parts, bounds, widths, heights
    ):
        draw.text(
            (x - box[0], (image.height - part_height) // 2 - box[1]),
            part, font=part_font, fill=(0, 0, 0, 255),
        )
        x += part_width + gap
    return image


def _prepare_text_line(text, font, ink_color, max_width, max_height, capital_font=None):
    """Apply the shared line pipeline before tinting and pasting text ink."""
    ink = _render_text_ink(text, font, capital_font)
    x_bounds = font.getbbox("x")
    spacing = max(4.0, (x_bounds[3] - x_bounds[1]) / 2)
    ink = _prepare_ink_line(ink, max_width, max_height, spacing=spacing)
    tinted = Image.new("RGBA", ink.size, (*ink_color, 0))
    tinted.putalpha(ink.getchannel("A"))
    return tinted


def synthesize_text_batch(
    texts,
    font_path,
    capitals_dir,
    width,
    height,
    texture_pool,
    lines_per_image=20,
    font_size=None,
):
    """Composite pre-distorted text lines, returning COCO xywh boxes."""
    img = get_texture(width, height, texture_pool)
    try:
        font = ImageFont.truetype(
            font_path, size=rg.randint(32, 45) if font_size is None else font_size
        )
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()
    ink_color = (
        57 + rg.randint(-10, 10),
        41 + rg.randint(-10, 10),
        25 + rg.randint(-10, 10),
    )
    sample_bbox = font.getbbox("Sample Text")
    line_spacing = (sample_bbox[3] - sample_bbox[1]) * 2
    start_y = max(20, (height - len(texts) * line_spacing) // 2)
    bboxes_and_texts = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        y_position = start_y + i * line_spacing
        slot_height = min(line_spacing - 10, height - 20 - y_position)
        if slot_height < 1:
            break
        try:
            ink = _prepare_text_line(text, font, ink_color, width - 40, slot_height)
        except ValueError:
            # Some fonts lack the abbreviation glyph a line reduces to; the
            # blank render carries no visible ink, so it cannot carry a label.
            continue
        x = _line_x_offset(width, ink.width)
        y = y_position + (slot_height - ink.height) // 2
        img.paste(ink, (x, y), ink)
        bboxes_and_texts.append(((x, y, ink.width, ink.height), text))
    return img, bboxes_and_texts


def synthesize_text(text, font_path, capitals_dir, width, height, texture_pool):
    """Composite a pre-distorted text line and return its xyxy bounds."""
    img = get_texture(width, height, texture_pool)
    if not text.strip():
        return img, None
    try:
        font = ImageFont.truetype(font_path, size=rg.randint(52, 65))
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()
    ink_color = (
        57 + rg.randint(-10, 10),
        41 + rg.randint(-10, 10),
        25 + rg.randint(-10, 10),
    )
    capital_font = None
    if capitals_dir and len(text) > 1 and Path(capitals_dir).exists():
        capital_fonts = [
            path for path in Path(capitals_dir).iterdir()
            if path.suffix.lower() in (".ttf", ".otf")
        ]
        if capital_fonts:
            try:
                rest_bbox = font.getbbox(text[1:])
                size = max(int((rest_bbox[3] - rest_bbox[1]) * (1.5 * rg.random() + 1)), 35)
                capital_font = ImageFont.truetype(str(rg.choice(capital_fonts)), size=size)
            except Exception as e:
                print(f"Error applying capital letter: {e}")
    ink = _prepare_text_line(
        text, font, ink_color, width - 40, height - 40, capital_font
    )
    x = _line_x_offset(width, ink.width)
    y = (height - ink.height) // 2
    img.paste(ink, (x, y), ink)
    return img, (x, y, x + ink.width, y + ink.height)


def _music_staff_spacing(image):
    """Estimate staff spacing from colored horizontal rules before deformation."""
    pixels = np.asarray(image)
    red = (
        (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 140)
        & (pixels[:, :, 2] < 140) & (pixels[:, :, 3] > 0)
    )
    counts = red.sum(axis=1)
    rows = np.flatnonzero(counts > max(1, counts.max() * 0.6))
    if rows.size:
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        centers = np.array([group.mean() for group in groups])
        if centers.size >= 2:
            return float(np.median(np.diff(centers)))
    return max(4.0, image.height / 8)


def _distort_music_ink(image, spacing=None):
    """Deform dark notation independently of the colored staff and paper."""
    pixels = np.array(image)
    rgb = pixels[:, :, :3]
    black = (rgb.max(axis=2) < 200) & (
        rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2) < 40
    )
    ink = np.where(black, pixels[:, :, 3], 0).astype(np.uint8)
    if not ink.any():
        return image

    pixels[black, 3] = 0
    h, w = ink.shape
    # Glyph-scale fields reshape individual contours instead of merely shifting
    # whole neume groups. Keep vertical motion within a quarter staff spacing.
    spacing = _music_staff_spacing(image) if spacing is None else spacing
    amplitude = rg.uniform(0.09, 0.25) * spacing
    cell = max(6, int(spacing * rg.uniform(0.6, 1.1)))
    fields = []
    for _ in range(2):
        grid = np.array(
            [[rg.uniform(-amplitude, amplitude) for _ in range(max(2, w // cell))]
             for _ in range(max(2, h // cell))],
            dtype=np.float32,
        )
        fields.append(cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR))
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ink = cv2.remap(
        ink, x + fields[0], y + fields[1], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Fractional dilation varies stroke weight without erasing thin stems.
    if rg.random() < 0.5:
        thick = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ink = cv2.addWeighted(ink, 1 - (weight := rg.uniform(0.15, 0.55)), thick, weight, 0)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.putalpha(Image.fromarray(ink))
    return Image.alpha_composite(Image.fromarray(pixels), overlay)


def _warp_music_staff(image, spacing=None):
    """Apply shared waves, slant, and varying staff spacing to all notation."""
    spacing = _music_staff_spacing(image) if spacing is None else spacing
    pixels = np.array(image, dtype=np.float32)
    h, w = pixels.shape[:2]
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    phase = rg.uniform(0, 2 * np.pi)
    wave = np.sin(2 * np.pi * rg.uniform(1.0, 2.5) * t + phase)
    ripple = np.sin(2 * np.pi * rg.uniform(3.0, 6.0) * t + rg.uniform(0, 2 * np.pi))
    displacement = spacing * (
        rg.uniform(0.23, 0.52) * wave
        + rg.uniform(0.05, 0.13) * ripple
        + rg.uniform(-0.33, 0.33) * (t - 0.5)
    )
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # Positive vertical scale preserves staff order while varying line spacing.
    stretch = 1 + rg.uniform(0.05, 0.12) * wave
    map_y = (y - h / 2) / stretch + h / 2 + displacement
    map_x = x + spacing * rg.uniform(0.065, 0.195) * ripple
    # Interpolate premultiplied color to avoid fringes from transparent pixels.
    pixels[:, :, :3] *= pixels[:, :, 3:4] / 255
    warped = cv2.remap(
        pixels, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    alpha = warped[:, :, 3:4]
    np.divide(warped[:, :, :3] * 255, alpha, out=warped[:, :, :3], where=alpha > 0)
    return Image.fromarray(np.clip(np.rint(warped), 0, 255).astype(np.uint8))


def _weather_music_ink(image):
    """Partially thin strokes and vary ink coverage without deleting symbols."""
    pixels = np.array(image)
    alpha = pixels[:, :, 3]
    if rg.random() < 0.5:
        thin = cv2.erode(
            alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Keep at least 65% of even a one-pixel stroke.
        weight = rg.uniform(0.1, 0.35)
        alpha = cv2.addWeighted(alpha, 1 - weight, thin, weight, 0)
    h, w = alpha.shape
    coverage = np.array(
        [[rg.uniform(0.55, 1.0) for _ in range(max(2, w // 12))]
         for _ in range(max(2, h // 12))],
        dtype=np.float32,
    )
    coverage = cv2.resize(coverage, (w, h), interpolation=cv2.INTER_LINEAR)
    pixels[:, :, 3] = np.rint(alpha * coverage).astype(np.uint8)
    return Image.fromarray(pixels)


def _prepare_ink_line(image, max_width, max_height, spacing=None):
    """Crop, distort, and scale transparent music or text ink before compositing."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGBA")
    bounds = image.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Line image has no visible content")
    image = image.crop(bounds)
    spacing = _music_staff_spacing(image) if spacing is None else spacing
    # Cover combined staff waves, vertical stretch, and local ink displacement.
    guard = int(np.ceil(2 * spacing + image.height * 0.15 + 4))
    padded = Image.new("RGBA", (image.width + 2 * guard, image.height + 2 * guard))
    padded.paste(image, (guard, guard))
    image = padded
    if rg.random() < 0.35:
        image = _warp_music_staff(image, spacing)
    if rg.random() < 0.9:
        image = _distort_music_ink(image, spacing)
    if rg.random() < 0.3:
        image = _weather_music_ink(image)
    image = image.crop(image.getchannel("A").getbbox())
    fit = min(max_width / image.width, max_height / image.height)
    scale = min(1.0, fit) if rg.random() < 0.5 else fit * rg.uniform(0.65, 1.0)
    size = (
        max(1, min(max_width, int(image.width * scale))),
        max(1, min(max_height, int(image.height * scale))),
    )
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _line_x_offset(page_width, line_width, margin=20):
    """Choose centered placement or independently sampled left/right padding."""
    free = page_width - line_width - 2 * margin
    if rg.random() < 0.5:
        return (page_width - line_width) // 2
    return margin + rg.randint(0, max(0, free))


def synthesize_music_batch(
    chant_images, width, height, texture_pool, lines_per_image=20
):
    """
    Generate an image with multiple music notation lines arranged vertically.

    Args:
        chant_images: List of (chant_image, gabc, custom) tuples
        width: Target width of the output image
        height: Target height of the output image
        texture_pool: Pool of background textures
        lines_per_image: Number of music lines to put in each image

    Returns:
        PIL Image and list of bounding boxes with corresponding custom notations
    """

    # Create background
    background = get_texture(width, height, texture_pool)

    # Calculate spacing for music lines
    available_height = height - 40  # Leave margin
    line_height = available_height // max(len(chant_images), 1)

    bboxes_and_texts = []

    for i, (chant_image, _, custom) in enumerate(chant_images):
        if chant_image is None or i >= lines_per_image:
            continue

        try:
            pil_image = _prepare_ink_line(chant_image, width - 40, line_height - 10)
            new_width, new_height = pil_image.size
            x_offset = _line_x_offset(width, new_width)
            y_offset = 20 + i * line_height + (line_height - new_height) // 2

            # Skip if would go outside bounds
            if y_offset + new_height > height - 20:
                continue

            # Paste the chant onto the background
            background.paste(pil_image, (x_offset, y_offset), pil_image.convert("RGBA"))

            # Store the ACTUAL paste position as bounding box
            bbox = (x_offset, y_offset, new_width, new_height)

            bboxes_and_texts.append((bbox, custom))

        except Exception as e:
            print(f"Error processing chant {i}: {e}")
            continue

    return background, bboxes_and_texts


def synthesize_mixed_batch(
    content_batch,
    font_path,
    width,
    height,
    texture_pool,
):
    """
    Generate an image with multiple text and music lines arranged vertically.
    """
    background = get_texture(width, height, texture_pool)
    draw = ImageDraw.Draw(background)

    try:
        font = ImageFont.truetype(font_path, size=rg.randint(32, 45))
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()

    ink_color = (
        57 + rg.randint(-10, 10),
        41 + rg.randint(-10, 10),
        25 + rg.randint(-10, 10),
    )

    # Estimate line height based on text font
    sample_bbox = draw.textbbox((0, 0), "Sample Text", font=font)
    text_line_height = sample_bbox[3] - sample_bbox[1]
    line_spacing = int(text_line_height * 1.5)  # Increased spacing for mixed content

    total_height = len(content_batch) * line_spacing
    start_y = max(20, (height - total_height) // 2)

    bboxes_and_texts = []

    for i, (content_type, content) in enumerate(content_batch):
        y_position = start_y + i * line_spacing

        if content_type == "text":
            text = content
            if not text.strip():
                continue

            if y_position + text_line_height > height - 20:
                break

            slot_height = min(line_spacing - 10, height - 20 - y_position)
            if slot_height < 1:
                break
            ink = _prepare_text_line(text, font, ink_color, width - 40, slot_height)
            x_position = _line_x_offset(width, ink.width)
            y_position += (slot_height - ink.height) // 2
            background.paste(ink, (x_position, y_position), ink)
            bbox = (x_position, y_position, ink.width, ink.height)
            bboxes_and_texts.append((bbox, text, 6))  # Label 6 for text

        elif content_type == "music":
            chant_image, _, custom = content
            if chant_image is None:
                continue

            try:
                pil_image = _prepare_ink_line(
                    chant_image, width - 40, line_spacing - 10
                )
                new_width, new_height = pil_image.size
                x_offset = _line_x_offset(width, new_width)
                # Center vertically within the allocated line space
                y_offset = y_position + (line_spacing - new_height) // 2

                if y_offset + new_height > height - 20:
                    continue

                background.paste(
                    pil_image, (x_offset, y_offset), pil_image.convert("RGBA")
                )
                bbox = (
                    x_offset,
                    y_offset,
                    new_width,
                    new_height,
                )
                bboxes_and_texts.append((bbox, custom, 5))  # Label 5 for music

            except Exception as e:
                print(f"Error processing chant {i}: {e}")
                continue

    return background, bboxes_and_texts


def synthesize_music(chant_image, width, height, texture_pool):
    """
    Generate an image with music notation.

    Args:
        chant_image: The numpy array image of a chant
        width: Target width of the output image
        height: Target height of the output image

    Returns:
        PIL Image containing the rendered music and bounding box coordinates
    """
    # Convert numpy array to PIL Image
    try:
        if chant_image is not None:
            pil_image = _prepare_ink_line(chant_image, width - 40, height - 40)
            new_width, new_height = pil_image.size

            # Create a background texture of the desired size
            background = get_texture(width, height, texture_pool)

            x_offset = _line_x_offset(width, new_width)
            y_offset = (height - new_height) // 2

            # Paste the chant onto the background
            background.paste(pil_image, (x_offset, y_offset), pil_image.convert("RGBA"))

            # Return the ACTUAL paste position, not content bounds
            bbox = (x_offset, y_offset, x_offset + new_width, y_offset + new_height)

            return background, bbox
        else:
            print("Received empty chant image")
            return None, None
    except Exception as e:
        print(f"Error in synthesize_music: {e}")
        return None, None


def augmentation_not_included(augmentation, excluded_augmentations, p=0.05):
    if augmentation.__class__.__name__ in excluded_augmentations:
        return False
    if hasattr(augmentation, "fliplr"):
        augmentation.fliplr = False
    if hasattr(augmentation, "flipud"):
        augmentation.flipud = False
    augmentation.p = p
    if hasattr(augmentation, "augmentations"):
        nested_augmentations = []
        probs = []
        for idx, aug in enumerate(augmentation.augmentations):
            if augmentation_not_included(aug, excluded_augmentations):
                nested_augmentations.append(aug)
                if hasattr(augmentation, "augmentation_probabilities"):
                    probs.append(augmentation.augmentation_probabilities[idx])
                if hasattr(aug, "fliplr"):
                    aug.fliplr = False
                if hasattr(aug, "flipud"):
                    aug.flipud = False
                augmentation.p = p
        augmentation.augmentations = nested_augmentations
        if hasattr(augmentation, "augmentation_probabilities"):
            augmentation.augmentation_probabilities = probs
        if len(augmentation.augmentations) == 0:
            return False
    return True


def apply_distortions(image, bboxes, debug, max_retries=10):
    """
    Apply color and geometric distortions with bounding box tracking.

    Args:
        image: PIL Image to distort
        bboxes: List of bounding box coordinates [[x1, y1, x2, y2], ...]
        debug: Enable debug logging
        max_retries: Maximum number of retry attempts if distortion fails

    Returns:
        Distorted image and transformed bounding boxes, or None if all retries fail
    """
    # Convert PIL image to numpy array for Augraphy
    image_np = np.array(image)

    # Remove unwanted augmentations but keep geometric ones
    excluded_augmentations = {
        "Dithering",
        "WaterMark",
        "Faxify",
        "Markup",
        "GlitchEffect",
        "ColorShift",
        "DirtyDrum",
        "NoisyLines",
        "InkColorSwap",
        "ColorPaper",
        "DirtyRollers",
        "Scribbles",
        "Markup",
        "Squish",
        "Hollow",
        "DotMatrix",
        "DelaunayTessellation",
        "VoronoiTessellation",
    }

    for attempt in range(max_retries):
        try:
            # Get fresh pipeline for each attempt
            pipeline = au.default_augraphy_pipeline()

            # Set bounding boxes on the pipeline instance
            pipeline.bounding_boxes = bboxes
            if debug:
                pipeline.log = True
                pipeline.log_prob_path = "logs/au/"
                os.makedirs(pipeline.log_prob_path, exist_ok=True)

            # Filter out excluded augmentations
            for phase in (
                pipeline.ink_phase,
                pipeline.paper_phase,
                pipeline.post_phase,
            ):
                phase.augmentations = [
                    augmentation
                    for augmentation in phase.augmentations
                    if augmentation_not_included(augmentation, excluded_augmentations)
                ]

            # Apply pipeline with return_dict=1 to get dictionary output
            result = pipeline.augment(
                image=image_np,
                return_dict=1,  # This returns a dictionary with all outputs
            )

            # Check if result is valid
            if result is None:
                if debug:
                    print(f"Distortion attempt {attempt + 1} returned None result")
                continue

            if "output" not in result or "bounding_boxes" not in result:
                if debug:
                    print(
                        f"Distortion attempt {attempt + 1} missing required keys in result: {list(result.keys()) if result else 'None'}"
                    )
                continue

            # Extract the transformed image and bounding boxes
            distorted_image = Image.fromarray(result["output"])
            transformed_bboxes = result["bounding_boxes"]

            # Validate the outputs
            if distorted_image is None or transformed_bboxes is None:
                if debug:
                    print(f"Distortion attempt {attempt + 1} produced None outputs")
                continue

            if debug and attempt > 0:
                print(f"Distortion succeeded on attempt {attempt + 1}")

            return distorted_image, transformed_bboxes

        except Exception as e:
            if debug:
                print(f"Distortion attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Error in distortion pipeline after {max_retries} attempts: {e}")
                return image, bboxes  # Return original if all attempts fail
            continue

    # If we get here, all retries failed
    print(f"All {max_retries} distortion attempts failed, returning original image")
    return image, bboxes



def get_fonts(font_dir):
    """Get list of available font files"""
    return [
        str(font_file)
        for pattern in ("./*.ttf", "./*.otf")
        for font_file in Path(font_dir).glob(pattern)
    ]

def generate_single_sample_batch(
    sample_index,
    content_batch,
    mode,
    fonts,
    capitals_dir,
    width,
    height,
    output_dir,
    texture_pool,
    lines_per_image=20,
    debug=False,
    selected_font_path=None,
    font_size=None,
):
    """Generate a single dataset sample with multiple lines batched together"""
    img_filename = f"{mode}{sample_index + 1:04d}.jpg"
    img_path = output_dir / img_filename
    bboxes_and_texts = []

    if mode == "text":
        texts = content_batch
        font_path = selected_font_path or rg.choice(fonts)
        result = synthesize_text_batch(
            texts,
            font_path,
            capitals_dir,
            width,
            height,
            texture_pool,
            lines_per_image,
            font_size=font_size,
        )
        if result is None:
            print(
                f"Warning: synthesize_text_batch returned None for sample {sample_index}"
            )
            return None

        img, bboxes_and_texts_unlabeled = result
        # Add label
        for bbox, text in bboxes_and_texts_unlabeled:
            bboxes_and_texts.append((bbox, text, 6))

    elif mode == "music":
        chant_data_list = content_batch
        result = synthesize_music_batch(
            chant_data_list, width, height, texture_pool, lines_per_image
        )
        if result is None:
            print(
                f"Warning: synthesize_music_batch returned None for sample {sample_index}"
            )
            return None

        img, bboxes_and_texts_unlabeled = result
        # Add label
        for bbox, text in bboxes_and_texts_unlabeled:
            bboxes_and_texts.append((bbox, text, 5))

    elif mode == "mixed":
        font_path = rg.choice(fonts)
        result = synthesize_mixed_batch(
            content_batch, font_path, width, height, texture_pool
        )
        if result is None:
            print(
                f"Warning: synthesize_mixed_batch returned None for sample {sample_index}"
            )
            return None

        img, bboxes_and_texts = result

    if img is not None and bboxes_and_texts:
        # Compositors emit COCO xywh; Augraphy tracks xyxy corners.
        bboxes = [
            [x, y, x + width, y + height]
            for (x, y, width, height), _text, _label in bboxes_and_texts
        ]

        # Apply distortions with geometric transformations (with retry mechanism)
        distortion_result = apply_distortions(img, bboxes, debug, max_retries=10)
        if distortion_result is None:
            print(f"Warning: apply_distortions returned None for sample {sample_index}")
            return None

        img, transformed_bboxes = distortion_result
        if len(transformed_bboxes) != len(bboxes_and_texts):
            # Dropped boxes lose positional correspondence to labels.
            return None

        # Update the bboxes_and_texts with transformed bounding boxes
        updated_bboxes_and_texts = []
        for idx, (_, text, label) in enumerate(bboxes_and_texts):
            if idx < len(transformed_bboxes):
                x0, y0, x1, y1 = transformed_bboxes[idx]
                x0 = max(1, min(int(x0), img.width - 1))
                y0 = max(1, min(int(y0), img.height - 1))
                x1 = max(1, min(int(x1), img.width - 1))
                y1 = max(1, min(int(y1), img.height - 1))
                if x1 > x0 and y1 > y0:
                    updated_bboxes_and_texts.append(
                        ([x0, y0, x1 - x0, y1 - y0], text, label)
                    )
        if not updated_bboxes_and_texts:
            return None

        # Save the image
        img.save(img_path)

        # Create sample JSON with transformed instances
        instances = []
        for idx, (bbox, text, label) in enumerate(updated_bboxes_and_texts):
            instances.append(
                {
                    "bbox": bbox,
                    "image_id": sample_index,
                    "description": text,
                    "id": f"{idx}-{sample_index}",
                    "category_id": label,
                }
            )

        sample_json = {
            "file_name": str(img_filename),
            "height": img.height,
            "width": img.width,
            "id": sample_index,
        }

        return sample_json, instances

    return None


def update_json_file(json_path, new_samples):
    """Update a JSON file with new samples."""
    if not new_samples:
        return

    # Load existing data
    if Path(json_path).exists():
        with open(json_path, "r") as f:
            dataset = json.load(f)

        # Update data
        dataset["images"].extend(new_samples["images"])
        dataset["annotations"].extend(new_samples["annotations"])
    else:
        dataset = new_samples

    # Write to a temporary file and then rename for atomic operation
    temp_path = json_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Atomic replace
    os.replace(temp_path, json_path)


