"""
This script creates a dataset of images and its corresponding JSON files.
It will take as input one text line and one font.
If the script is called with `text` command, the `get_line` will be from `.bibliotecaitaliana` module, while it if the command is `music` it will be from
`.gregobase_generator` module.
It will coherently call `synthesize_music` or `synthesize_text` functions to generate the images.

To synthesize text, it will use fonts from the directory given as argument, which and capitals from another directory given as argument.

It will then set a background color and a text/music color by converting white and black pixels and apply color e perspective distortions to the images.
"""

import copy
import argparse
import json
import os
import random as rg
from pathlib import Path

import augraphy as au
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .bibliotecaitaliana import get_line_generator
from .gregobase import get_music_page_generator

rg = rg.Random(2025)


def get_texture(width, height, texture_pool):
    """Get a random texture from the pre-generated pool and resize it if needed"""
    texture = rg.choice(texture_pool)

    # Resize if the dimensions don't match
    if texture.width != width or texture.height != height:
        texture = texture.resize((width, height), Image.Resampling.LANCZOS)

    return texture


def synthesize_text_batch(
    texts, font_path, capitals_dir, width, height, texture_pool, lines_per_image=20
):
    """
    Generate an image with multiple text lines arranged vertically.

    Args:
        texts: List of text strings to render
        font_path: Path to the font file
        capitals_dir: Directory containing font files for decorative capitals
        width: Width of the output image
        height: Height of the output image
        texture_pool: Pool of background textures
        lines_per_image: Number of text lines to put in each image

    Returns:
        PIL Image and list of bounding boxes with corresponding texts
    """
    capitals_dir = Path(capitals_dir) if capitals_dir else None

    # Create background image
    img = get_texture(width, height, texture_pool)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            font_path, size=rg.randint(32, 45)
        )  # Smaller font for multiple lines
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()

    ink_color = (
        57 + rg.randint(-10, 10),
        41 + rg.randint(-10, 10),
        25 + rg.randint(-10, 10),
    )

    # Calculate line spacing
    sample_bbox = draw.textbbox((0, 0), "Sample Text", font=font)
    line_height = sample_bbox[3] - sample_bbox[1]
    line_spacing = int(line_height * 2)  # 30% spacing between lines

    # Calculate starting Y position to center all lines vertically
    total_height = len(texts) * line_spacing
    start_y = max(20, (height - total_height) // 2)

    bboxes_and_texts = []

    for i, text in enumerate(texts):
        if not text.strip():
            continue

        y_position = start_y + i * line_spacing

        # Skip if text would go outside image bounds
        if y_position + line_height > height - 20:
            break

        # Calculate text positioning
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]

        # Center text horizontally with some margin
        x_position = max(20, (width - text_width) // 2)

        # Skip if text is too wide
        if x_position + text_width > width - 20:
            continue

        # Draw the text
        draw.text((x_position, y_position), text, font=font, fill=ink_color)

        # Get the actual bounding box of the drawn text
        text_bbox = draw.textbbox((x_position, y_position), text, font=font)
        bbox = (
            text_bbox[0],
            text_bbox[1],
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
        )
        bboxes_and_texts.append((bbox, text))

    return img, bboxes_and_texts


def synthesize_text(text, font_path, capitals_dir, width, height, texture_pool):
    """
    Generate an image with the provided text using the specified font.

    Args:
        text: The text to render
        font_path: Path to the font file
        capitals_dir: Directory containing font files for decorative capitals
        width: Width of the output image
        height: Height of the output image

    Returns:
        PIL Image containing the rendered text and bounding box coordinates
    """
    capitals_dir = Path(capitals_dir) if capitals_dir else None
    # Create a blank white image
    img = get_texture(width, height, texture_pool)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, size=rg.randint(52, 65))
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        font = ImageFont.load_default()

    # Check if we should use a decorative capital
    use_decorative_capital = (
        capitals_dir and text and len(text) > 1 and capitals_dir.exists()
    )

    ink_color = (
        57 + rg.randint(-10, 10),
        41 + rg.randint(-10, 10),
        25 + rg.randint(-10, 10),
    )  # rgb

    # If using decorative capital, we'll handle the text differently
    if use_decorative_capital:
        first_char = text[0].upper()
        rest_of_text = text[1:]

        # Calculate size of the rest of the text
        rest_bbox = draw.textbbox((0, 0), rest_of_text, font=font)
        rest_width = rest_bbox[2] - rest_bbox[0]
        rest_height = rest_bbox[3] - rest_bbox[1]

        # Get list of fonts in the capitals directory
        capital_fonts = [
            f for f in os.listdir(capitals_dir) if f.endswith((".ttf", ".otf"))
        ]

        # Choose a random font for the capital
        capital_font_path = os.path.join(capitals_dir, rg.choice(capital_fonts))  # type: ignore

        try:
            # Load the decorative font and make it larger than the normal text
            capital_font_size = max(int(rest_height * (1.5 * rg.random() + 1)), 35)
            capital_font = ImageFont.truetype(capital_font_path, size=capital_font_size)

            # Get the size of the capital letter
            capital_bbox = draw.textbbox((0, 0), first_char, font=capital_font)
            capital_width = capital_bbox[2] - capital_bbox[0]
            capital_height = capital_bbox[3] - capital_bbox[1]

            # Calculate text positioning
            # Position capital letter and ensure rest of text aligns properly
            capital_x = (width - (capital_width + rest_width + 5)) // 2
            capital_y = (height - capital_height) // 2

            # Draw the capital letter
            draw.text(
                (capital_x, capital_y), first_char, font=capital_font, fill=ink_color
            )

            # Get the actual bounding box of the drawn capital
            capital_bbox = draw.textbbox(
                (capital_x, capital_y), first_char, font=capital_font
            )

            # Draw the rest of the text next to the capital
            rest_x = capital_x + capital_width + 5
            rest_y = (height - rest_height) // 2
            draw.text((rest_x, rest_y), rest_of_text, font=font, fill=ink_color)

            # Get the actual bounding box of the drawn text
            text_bbox = draw.textbbox((rest_x, rest_y), rest_of_text, font=font)
            # Create a bounding box that encompasses both the capital and the text
            bbox = (
                min(capital_bbox[0], text_bbox[0]),
                min(capital_bbox[1], text_bbox[1]),
                max(capital_bbox[2], text_bbox[2]),
                max(capital_bbox[3], text_bbox[3]),
            )

        except Exception as e:
            print(f"Error applying capital letter: {e}")
            standard_text_rendering = True
        else:
            standard_text_rendering = False
            # Use the bounding box from the decorative capital path
            bbox = (
                min(capital_bbox[0], text_bbox[0]),
                min(capital_bbox[1], text_bbox[1]),
                max(capital_bbox[2], text_bbox[2]),
                max(capital_bbox[3], text_bbox[3]),
            )
    else:
        standard_text_rendering = True

    if standard_text_rendering:
        # Standard text rendering (used if decorative capital failed or is not requested)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        # Draw text
        draw.text((x, y), text, font=font, fill=ink_color)

        # Get the actual bounding box of the drawn text
        text_bbox = draw.textbbox((x, y), text, font=font)
        bbox = (
            text_bbox[0],
            text_bbox[1],
            text_bbox[2],
            text_bbox[3],
        )

    return img, bbox


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

    def get_content_bbox(img):
        """Get the bounding box of non-transparent content in an RGBA image"""
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Get alpha channel
        alpha = np.array(img.split()[-1])

        # Find where alpha > 0 (non-transparent pixels)
        coords = np.column_stack(np.where(alpha > 0))
        if coords.size == 0:
            return None  # No content

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return (x_min, y_min, x_max + 1, y_max + 1)

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
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(chant_image, "RGBA")

            # Crop to content before resizing
            content_bbox = get_content_bbox(pil_image)
            if content_bbox:
                pil_image = pil_image.crop(content_bbox)

            # Scale to fit within line height while maintaining aspect ratio
            original_width, original_height = pil_image.size
            max_width = width - 40  # Leave horizontal margin
            max_height = line_height - 10  # Leave vertical spacing

            scale = min(max_width / original_width, max_height / original_height)

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            if scale != 1:
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

            # Calculate position
            x_offset = (width - new_width) // 2
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

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x_position = max(20, (width - text_width) // 2)

            if x_position + text_width > width - 20:
                continue

            draw.text((x_position, y_position), text, font=font, fill=ink_color)
            bbox = (
                x_position,
                y_position,
                x_position + text_width,
                y_position + text_height,
            )
            bboxes_and_texts.append((bbox, text, 6))  # Label 6 for text

        elif content_type == "music":
            chant_image, _, custom = content
            if chant_image is None:
                continue

            try:
                pil_image = Image.fromarray(chant_image, "RGBA")
                original_width, original_height = pil_image.size
                max_width = width - 40
                max_height = line_spacing - 10

                scale = min(max_width / original_width, max_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                if scale != 1:
                    pil_image = pil_image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

                x_offset = (width - new_width) // 2
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
                    x_offset + new_width,
                    y_offset + new_height,
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
            # Convert the BGRA image to RGB for compatibility with PIL
            pil_image = Image.fromarray(chant_image, "RGBA")

            # Resize if needed to fit within target dimensions
            original_width, original_height = pil_image.size
            scale = min(width / original_width, height / original_height)

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            if scale != 1:
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )

            # Create a background texture of the desired size
            background = get_texture(width, height, texture_pool)

            # Calculate position to center the chant image on the background
            x_offset = (width - new_width) // 2
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


def initialize_generator(mode, latex_jobs=4, max_pages=None, debug=False):
    if mode == "text":
        generator = get_line_generator(
            allowed_languages=["ita"],
            periods=["Origini", "200", "300", "400"],
        )
        return generator
    elif mode == "music":
        generator = get_music_page_generator(
            width_range=(800, 1200),
            height_range=(300, 600),
            latex_jobs=latex_jobs,
            max_pages=max_pages,
            debug=debug,
        )
        return generator
    elif mode == "mixed":
        text_gen = get_line_generator(
            allowed_languages=["ita"],
            periods=["Origini", "200", "300", "400"],
        )
        music_gen = get_music_page_generator(
            width_range=(800, 1200),
            height_range=(300, 600),
            latex_jobs=latex_jobs,
            max_pages=max_pages,
            debug=debug,
        )
        return {"text": text_gen, "music": music_gen}
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_fonts(font_dir):
    """Get list of available font files"""
    return [
        str(font_file)
        for pattern in ("./*.ttf", "./*.otf")
        for font_file in Path(font_dir).glob(pattern)
    ]


def generate_dataset(
    output_dir,
    font_dir,
    capitals_dir,
    mode,
    task=None,
    num_samples=100,
    width=800,
    height=1280,
    train_ratio=0.8,
    lines_per_image=1,
    latex_jobs=4,
    debug=False,
):
    """
    Generate a dataset of images and corresponding JSON files split into training and validation sets

    Args:
        output_dir: Directory to save the generated dataset
        font_dir: Directory containing font files
        capitals_dir: Directory containing capital letter images
        mode: 'text', 'music', or 'mixed'
        task: 'ocr' or 'omr' (used for output JSON naming)
        num_samples: Number of samples to generate
        width: Width of the output images
        height: Height of the output images
        train_ratio: Portion of samples to include in training set (default: 0.8)
        lines_per_image: Number of text/music lines to include in each image (default: 1)
    """
    # Adjust height automatically for batched mode
    if lines_per_image > 1:
        height = max(height, lines_per_image * 100 + 100)  # Ensure enough space
        print(
            f"   📏 Adjusted image height to {height}px for {lines_per_image} lines per image"
        )
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset structure
    (output_dir / "annotations-diplomatic").mkdir(parents=True, exist_ok=True)

    # List available fonts (for text mode only)
    fonts = get_fonts(font_dir) if mode in ["text", "mixed"] else []
    if mode in ["text", "mixed"] and not fonts:
        raise ValueError(
            f"No fonts found in {font_dir}. Cannot generate {mode} dataset without fonts."
        )

    # Calculate split
    num_train = int(num_samples * train_ratio)

    # Calculate max_pages needed for music mode
    max_pages = None
    if mode == "music":
        max_pages = num_samples * lines_per_image
        print(
            f"   🎵 Will generate up to {max_pages} lines of music (if there is enough data for that)."
        )

    # Initialize content generator
    generator = initialize_generator(
        mode, latex_jobs=latex_jobs, max_pages=max_pages, debug=debug
    )

    # pre-generating textures because voronoi tasselletion is not working in multiprocess nor with a
    # Lock...
    print("Pre-generating texture pool...")
    texture_pool = []
    # Create a single VoronoiTessellation instance
    num_cells_range = (5000, 6000) if not debug else (1000, 2000)
    voronoi_pattern = au.VoronoiTessellation(
        mult_range=(50, 80),
        seed=1994,
        num_cells_range=num_cells_range,
        noise_type="random",
        background_value=(200, 256),
    )
    # Generate variations of textures with different sizes
    for _ in range(100 if not debug else 10):  # Create fewer textures in debug
        w = round((rg.random() + 0.5) * width)
        h = round((rg.random() + 0.5) * height)

        background = (
            91 + rg.randint(-5, 5),
            82 + rg.randint(-5, 5),
            65 + rg.randint(-5, 5),
        )
        image = Image.new("RGB", (w, h), color=background)
        data = np.asarray(image)
        data = voronoi_pattern(data)
        texture = Image.fromarray(data)
        texture_pool.append(texture)

    task_map = {
        "text": "ocr",
        "music": "omr",
        "mixed": "mixed",
    }
    task = task or task_map[mode]

    # Generate data sequentially
    train_json_path = output_dir / "annotations-diplomatic" / f"{task}_train.json"
    val_json_path = output_dir / "annotations-diplomatic" / f"{task}_val.json"
    pbar = tqdm(range(num_samples), desc=f"Generating {mode} dataset")
    for i in pbar:
        train_data = dict(
            annotations=[],
            images=[],
            categories=[
                dict(id=5, name="staff"),
                dict(id=6, name="line"),
            ],
        )
        valid_data = copy.deepcopy(train_data)
        try:
            if lines_per_image > 1:
                content_batch = []
                for _ in range(lines_per_image):
                    if mode == "text":
                        content = next(generator)
                    elif mode == "music":
                        content = next(generator)
                    elif mode == "mixed":
                        gen_type = rg.choice(["text", "music"])
                        content = (gen_type, next(generator[gen_type]))
                    content_batch.append(content)

                if not content_batch:
                    continue

                result = generate_single_sample_batch(
                    i,
                    content_batch,
                    mode,
                    fonts,
                    capitals_dir,
                    width,
                    height,
                    output_dir,
                    texture_pool,
                    lines_per_image,
                    debug,
                )
            else:
                raise RuntimeError("Single line mode not supported anymore")

            if result:
                sample_json, instances = result
                if i < num_train:
                    train_data["images"].append(sample_json)
                    train_data["annotations"].extend(instances)
                    update_json_file(
                        train_json_path,
                        train_data,
                    )
                else:
                    valid_data["images"].append(sample_json)
                    valid_data["annotations"].extend(instances)
                    update_json_file(val_json_path, valid_data)

        except StopIteration:
            pbar.set_description(f"Generator completed after {i + 1} samples")
            break
    # Ensure generator is always closed to shutdown ThreadPoolExecutor
    if mode == "mixed":
        for gen in generator.values():
            gen.close()
    else:
        generator.close()

    # Ensure both train.json and val.json files exist, even if empty
    if not train_json_path.exists():
        empty_data = dict(
            annotations=[],
            images=[],
            categories=[
                dict(id=5, name="staff"),
                dict(id=6, name="line"),
            ],
        )
        with open(train_json_path, "w") as f:
            json.dump(empty_data, f, indent=2)

    if not val_json_path.exists():
        empty_data = dict(
            annotations=[],
            images=[],
            categories=[
                dict(id=5, name="staff"),
                dict(id=6, name="line"),
            ],
        )
        with open(val_json_path, "w") as f:
            json.dump(empty_data, f, indent=2)

    return train_json_path, val_json_path


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
):
    """Generate a single dataset sample with multiple lines batched together"""
    img_filename = f"{mode}{sample_index + 1:04d}.jpg"
    img_path = output_dir / img_filename
    bboxes_and_texts = []

    # Process based on the mode
    if mode == "text":
        texts = content_batch
        font_path = rg.choice(fonts)
        result = synthesize_text_batch(
            texts, font_path, capitals_dir, width, height, texture_pool, lines_per_image
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
        # Extract the bounding boxes as lists
        bboxes = [list(bbox) for bbox, text, label in bboxes_and_texts]

        # Apply distortions with geometric transformations (with retry mechanism)
        distortion_result = apply_distortions(img, bboxes, debug, max_retries=10)
        if distortion_result is None:
            print(f"Warning: apply_distortions returned None for sample {sample_index}")
            return None

        img, transformed_bboxes = distortion_result

        # Update the bboxes_and_texts with transformed bounding boxes
        updated_bboxes_and_texts = []
        for idx, (_, text, label) in enumerate(bboxes_and_texts):
            if idx < len(transformed_bboxes):
                updated_bboxes_and_texts.append((transformed_bboxes[idx], text, label))

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset of text or music notation images"
    )
    parser.add_argument(
        "mode",
        choices=["text", "music"],
        help="Dataset type to generate (text or music)",
    )
    parser.add_argument("--output", "-o", default="dataset", help="Output directory")
    parser.add_argument(
        "--fonts", "-f", help="Directory containing font files (required for text mode)"
    )
    parser.add_argument(
        "--capitals",
        "-c",
        help="Directory containing capital letter images (for text mode)",
    )
    parser.add_argument(
        "--samples", "-n", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=1024, help="Width of the output images"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=200, help="Height of the output images"
    )
    parser.add_argument(
        "--lines-per-image",
        type=int,
        default=1,
        help="Number of lines per image for batch mode.",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.mode == "text" and not args.fonts:
        parser.error("Text mode requires --fonts directory")

    task_map = {
        "text": "ocr",
        "music": "omr",
    }
    generate_dataset(
        args.output,
        args.fonts,
        args.capitals,
        args.mode,
        task=task_map.get(args.mode),
        num_samples=args.samples,
        width=args.width,
        height=args.height,
        lines_per_image=args.lines_per_image,
    )


if __name__ == "__main__":
    main()
