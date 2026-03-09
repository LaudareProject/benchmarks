import argparse
import json
import random
from pathlib import Path
from copy import deepcopy
import xml.etree.ElementTree as ET
import os


def filter_valid_annotations(annotations):
    """
    Filter out annotations with no bounding box or area 0, with warnings.

    Args:
        annotations (list): List of annotation dictionaries

    Returns:
        list: List of valid annotations with non-zero area
    """
    valid_annotations = []
    filtered_count = 0

    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            print(
                f"⚠️  Warning: Annotation {ann.get('id', 'unknown')} has no valid bounding box"
            )
            filtered_count += 1
            continue

        x, y, w, h = bbox
        area = w * h
        if area <= 0:
            print(
                f"⚠️  Warning: Annotation {ann.get('id', 'unknown')} has zero or negative area ({area})"
            )
            filtered_count += 1
            continue

        valid_annotations.append(ann)

    if filtered_count > 0:
        print(
            f"⚠️  Filtered out {filtered_count} annotations with invalid bounding boxes or zero area"
        )

    return valid_annotations


def compute_overlap_area(bbox1, bbox2):
    """
    Compute the overlapping area between two bounding boxes.

    Args:
        bbox1 (list): The first bounding box [x, y, width, height].
        bbox2 (list): The second bounding box [x, y, width, height].

    Returns:
        float: The overlapping area.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    overlap_area = x_overlap * y_overlap
    return overlap_area


def write_json(data, output_file):
    with open(output_file, "w") as f:
        json.dump(data, f)


def create_fold_data(images, annotations, categories, indices):
    # Filter annotations for the selected images
    fold_annotations = [
        ann
        for ann in annotations
        if ann["image_id"] in [images[i]["id"] for i in indices]
    ]

    # Filter out invalid annotations
    valid_annotations = filter_valid_annotations(fold_annotations)

    fold_data = {
        "images": [images[i] for i in indices],
        "annotations": valid_annotations,
        "categories": categories,
    }
    return fold_data


def add_text_regions_to_new_pagexml(page_element, annotations, task_type):
    """Add text regions with transcriptions to a new PageXML's Page element."""
    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

    # Add new TextRegion elements for each annotation
    for i, ann in enumerate(annotations):
        bbox = ann["bbox"]
        text = ann.get("text", "").strip()

        # Skip annotations without text content
        if not text:
            continue

        # Create TextRegion element
        text_region = ET.SubElement(page_element, f"{{{NS}}}TextRegion")
        text_region.set("id", f"region_{task_type}_{i}")
        text_region.set("type", "paragraph")

        # Create Coords element for TextRegion
        coords = ET.SubElement(text_region, f"{{{NS}}}Coords")
        x, y, w, h = bbox
        points = f"{int(x)},{int(y)} {int(x + w)},{int(y)} {int(x + w)},{int(y + h)} {int(x)},{int(y + h)}"
        coords.set("points", points)

        # Create TextLine element
        text_line = ET.SubElement(text_region, f"{{{NS}}}TextLine")
        text_line.set("id", f"line_{task_type}_{i}")

        # Create Coords for TextLine (same as TextRegion for simplicity)
        line_coords = ET.SubElement(text_line, f"{{{NS}}}Coords")
        line_coords.set("points", points)

        # Create Baseline element (required by Kraken)
        baseline = ET.SubElement(text_line, f"{{{NS}}}Baseline")
        # Create a horizontal baseline near the bottom of the text (at ~80% of height)
        baseline_y = int(y + h * 0.8)
        baseline_points = f"{int(x)},{baseline_y} {int(x + w)},{baseline_y}"
        baseline.set("points", baseline_points)

        # Create TextEquiv element
        text_equiv = ET.SubElement(text_line, f"{{{NS}}}TextEquiv")
        text_equiv.set("conf", "1.0")  # Add confidence
        unicode_elem = ET.SubElement(text_equiv, f"{{{NS}}}Unicode")
        unicode_elem.text = text


def add_layout_regions_to_new_pagexml(page_element, annotations, categories_list=None):
    """
    Adds layout regions to a PageXML element based on hierarchical COCO annotations.
    - Categories 4 (text) and 7 (musicText) are treated as TextRegions.
    - Categories 5 (staff) and 6 (line) are treated as TextLines within those regions.
    If `categories_list` is provided, it creates a flat structure for model predictions.
    """
    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

    if categories_list:
        # Logic for flat predictions (from Faster R-CNN)
        category_map = {cat["id"]: cat["name"] for cat in categories_list}
        for i, ann in enumerate(annotations):
            x, y, w, h = ann["bbox"]
            region = ET.SubElement(page_element, f"{{{NS}}}TextRegion")
            region.set("id", f"pred_region_{i}")

            # Set region type from category name for easier evaluation
            region_type = category_map.get(ann["category_id"], "unknown")
            region.set("type", region_type)

            points = f"{int(x)},{int(y)} {int(x + w)},{int(y)} {int(x + w)},{int(y + h)} {int(x)},{int(y + h)}"
            ET.SubElement(region, f"{{{NS}}}Coords").set("points", points)
        return

    # Separate annotations into regions and lines
    # Original logic for generating hierarchical ground truth PageXML
    region_anns = [ann for ann in annotations if ann.get("category_id") in [4, 7]]
    line_anns = [ann for ann in annotations if ann.get("category_id") in [5, 6]]

    # Process each region annotation
    for i, region_ann in enumerate(region_anns):
        region_bbox = region_ann["bbox"]

        # Create TextRegion element
        region = ET.SubElement(page_element, f"{{{NS}}}TextRegion")
        region.set("id", f"region_{i}")

        # Set region coordinates
        x_r, y_r, w_r, h_r = region_bbox
        region_points = f"{int(x_r)},{int(y_r)} {int(x_r + w_r)},{int(y_r)} {int(x_r + w_r)},{int(y_r + h_r)} {int(x_r)},{int(y_r + h_r)}"
        ET.SubElement(region, f"{{{NS}}}Coords").set("points", region_points)

        # Find all lines contained within this region
        contained_lines = []
        for line_ann in line_anns:
            line_bbox = line_ann["bbox"]
            line_area = line_bbox[2] * line_bbox[3]

            if line_area > 0:
                overlap = compute_overlap_area(line_bbox, region_bbox)
                # Check for containment with a small tolerance
                if abs(overlap - line_area) < 1e-6:
                    contained_lines.append(line_ann)

        # Add contained lines to the region, sorted by vertical position
        sorted_lines = sorted(contained_lines, key=lambda a: a["bbox"][1])
        for j, line_ann in enumerate(sorted_lines):
            line_bbox = line_ann["bbox"]

            # Create TextLine element
            text_line = ET.SubElement(region, f"{{{NS}}}TextLine")
            text_line.set("id", f"region_{i}_line_{j}")

            # Set line coordinates
            x, y, w, h = line_bbox
            points = f"{int(x)},{int(y)} {int(x + w)},{int(y)} {int(x + w)},{int(y + h)} {int(x)},{int(y + h)}"
            ET.SubElement(text_line, f"{{{NS}}}Coords").set("points", points)

            # Set baseline
            baseline = ET.SubElement(text_line, f"{{{NS}}}Baseline")
            baseline_y = int(y + h * 0.8)
            baseline.set("points", f"{int(x)},{baseline_y} {int(x + w)},{baseline_y}")


def create_new_pagexml_file(
    output_xml_path,
    image_filename,
    image_width,
    image_height,
    annotations,
    task_type,
    categories_list=None,
    image_base_path=None,
):
    """
    Creates a new PageXML file from scratch with the given annotations.
    """
    # Define the namespace
    NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
    ET.register_namespace("", NS)  # Register default namespace

    # Create root element
    pcgts_element = ET.Element(f"{{{NS}}}PcGts")
    pcgts_element.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    pcgts_element.set(
        "xsi:schemaLocation",
        f"{NS} http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd",
    )

    # Add Metadata
    metadata_element = ET.SubElement(pcgts_element, f"{{{NS}}}Metadata")
    creator_element = ET.SubElement(metadata_element, f"{{{NS}}}Creator")
    creator_element.text = "ann_handler.py"
    created_element = ET.SubElement(metadata_element, f"{{{NS}}}Created")
    import datetime

    created_element.text = datetime.datetime.utcnow().isoformat() + "Z"
    last_change_element = ET.SubElement(metadata_element, f"{{{NS}}}LastChange")
    last_change_element.text = created_element.text

    # Add Page element
    page_element = ET.SubElement(pcgts_element, f"{{{NS}}}Page")

    # Find the absolute path to the image
    # Use provided image_base_path or default to data/I-Ct_91/
    if image_base_path is None:
        image_path = Path("data") / "I-Ct_91" / image_filename
    else:
        image_path = Path(image_base_path) / image_filename
    xml_dir = Path(output_xml_path).parent
    relative_image_path = os.path.relpath(str(image_path), str(xml_dir))

    # Set image filename to relative path for Kraken to find it
    page_element.set("imageFilename", str(relative_image_path))
    page_element.set("imageWidth", str(int(image_width)))
    page_element.set("imageHeight", str(int(image_height)))

    # Filter annotations to only include those with text content for OCR/OMR tasks
    if task_type in ["ocr", "omr", "ocmr"]:
        valid_annotations = []
        for ann in annotations:
            text = ann.get("text", "").strip()
            if text:
                valid_annotations.append(ann)

        if not valid_annotations:
            # Create a dummy annotation to avoid empty files
            dummy_ann = {"bbox": [0, 0, 100, 50], "text": "dummy_text_for_kraken"}
            valid_annotations = [dummy_ann]

        add_text_regions_to_new_pagexml(page_element, valid_annotations, task_type)
    elif task_type == "layout":
        add_layout_regions_to_new_pagexml(page_element, annotations, categories_list)

    # Write XML file
    tree = ET.ElementTree(pcgts_element)
    try:
        ET.indent(tree, space="  ", level=0)  # Pretty print
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"❌ Error writing PageXML {output_xml_path}: {e}")


def create_pagexml_from_annotations(
    json_file, output_pagexml_dir, task_type, image_base_path=None
):
    """
    Create PageXML files from JSON annotations from scratch.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    output_pagexml_dir.mkdir(parents=True, exist_ok=True)

    # Filter out invalid annotations first
    valid_annotations = filter_valid_annotations(data.get("annotations", []))

    # Group annotations by image, also store image metadata
    image_data_map = {}

    image_id_to_meta = {
        img["id"]: (Path(img["file_name"]).name, img["width"], img["height"])
        for img in data.get("images", [])
    }

    for ann in valid_annotations:
        image_id = ann["image_id"]
        if image_id in image_id_to_meta:
            img_filename, img_width, img_height = image_id_to_meta[image_id]
            pagexml_filename = f"{Path(img_filename).stem}.xml"

            if pagexml_filename not in image_data_map:
                image_data_map[pagexml_filename] = {
                    "image_filename": img_filename,
                    "image_width": img_width,
                    "image_height": img_height,
                    "annotations": [],
                }
            # The PageXML creation function expects a 'text' field.
            # The source COCO has 'description'. Let's ensure 'text' exists.
            ann["text"] = ann.get("description", ann.get("text", ""))
            image_data_map[pagexml_filename]["annotations"].append(ann)

    # Create each PageXML file from scratch
    pagexml_count = 0
    for pagexml_filename, img_data in image_data_map.items():
        output_xml_path = output_pagexml_dir / pagexml_filename

        create_new_pagexml_file(
            output_xml_path,
            img_data["image_filename"],
            img_data["image_width"],
            img_data["image_height"],
            img_data["annotations"],
            task_type,
            image_base_path=image_base_path,
        )
        pagexml_count += 1

    print(f"   ✅ Generated {pagexml_count} {task_type.upper()} PageXML files")


def convert_and_write_coco_anns(
    fold_data, fold, output_folder, prefix, layout_categories
):
    if fold is not None:
        fold_str = f"_{fold:02d}"
    else:
        fold_str = ""

    def filter_data(category_ids):
        data = deepcopy(fold_data)
        filtered_annotations = [
            ann
            for ann in fold_data.get("annotations", [])
            if ann["category_id"] in category_ids
        ]
        # Filter out invalid annotations
        data["annotations"] = filter_valid_annotations(filtered_annotations)

        if data["annotations"]:
            referenced_image_ids = {ann["image_id"] for ann in data["annotations"]}
            data["images"] = [
                img
                for img in fold_data.get("images", [])
                if img["id"] in referenced_image_ids
            ]
        else:
            data["images"] = []
        return data

    # Text line is category 6, Music line is 5
    ocr_data = filter_data([6])
    omr_data = filter_data([5])
    ocmr_data = filter_data([5, 6])
    layout_data = filter_data(layout_categories)  # layout_categories is [5, 6]

    # Write JSON files
    write_json(layout_data, output_folder / f"layout_{prefix}{fold_str}.json")
    write_json(ocr_data, output_folder / f"ocr_{prefix}{fold_str}.json")
    write_json(omr_data, output_folder / f"omr_{prefix}{fold_str}.json")
    write_json(ocmr_data, output_folder / f"ocmr_{prefix}{fold_str}.json")


def create_task_specific_jsons(all_data, output_folder, prefix):
    layout_categories = [
        d["id"] for d in all_data["categories"] if d["supercategory"] == "layout"
    ]
    convert_and_write_coco_anns(
        all_data, None, Path(output_folder), prefix, layout_categories
    )


def create_all_pagexml_files(
    output_folder,
    all_data,
    dataset_type="diplomatic",
    image_base_path=None,
    target_tasks=None,
):
    """Create task-specific PageXML files for selected tasks.

    Args:
        output_folder: Directory to create PageXML files in
        all_data: COCO-format annotation data
        dataset_type: Type of dataset ("diplomatic" or "editorial")
        image_base_path: Base directory where images are located (default: data/I-Ct_91)
        target_tasks: List of tasks to create PAGE XML for. If None, creates for OCR, OMR, LAYOUT
    """

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    def filter_data(category_ids):
        data = deepcopy(all_data)
        filtered_annotations = [
            ann
            for ann in all_data.get("annotations", [])
            if ann["category_id"] in category_ids
        ]
        # Filter out invalid annotations
        data["annotations"] = filter_valid_annotations(filtered_annotations)

        if data["annotations"]:
            referenced_image_ids = {ann["image_id"] for ann in data["annotations"]}
            data["images"] = [
                img
                for img in all_data.get("images", [])
                if img["id"] in referenced_image_ids
            ]
        else:
            data["images"] = []
        return data

    # Define directories
    pagexml_ocr_dir = Path(output_folder) / f"pagexml_all_{dataset_type}_ocr"
    pagexml_omr_dir = Path(output_folder) / f"pagexml_all_{dataset_type}_omr"
    pagexml_layout_dir = Path(output_folder) / f"pagexml_all_{dataset_type}_layout"
    # pagexml_ocmr_dir = Path(output_folder) / f"pagexml_all_{dataset_type}_ocmr"
    print("   📄 Creating task-specific PageXML files...")

    def create_xml_for_task(task_name, category_ids, output_dir):
        print(f"   📝 Creating {task_name.upper()} PageXML files in {output_dir}...")
        task_data = filter_data(category_ids)
        temp_json = Path(output_folder) / f"temp_all_{task_name}.json"
        write_json(task_data, temp_json)
        create_pagexml_from_annotations(
            temp_json, output_dir, task_name, image_base_path
        )
        temp_json.unlink()

    # Default to OCR and OMR only (exclude layout for synthesis)
    if target_tasks is None:
        target_tasks = ["ocr", "omr", "layout"]
    if type(target_tasks) is str:
        target_tasks = [target_tasks]

    # Text line is 6, Music line is 5
    if "ocr" in target_tasks:
        create_xml_for_task("ocr", [6], pagexml_ocr_dir)
    if "omr" in target_tasks:
        create_xml_for_task("omr", [5], pagexml_omr_dir)
    if "layout" in target_tasks:
        create_xml_for_task("layout", [4, 5, 6, 7], pagexml_layout_dir)
    # create_xml_for_task("ocmr", [5, 6], pagexml_ocmr_dir)


def split_into_folds(
    input_file,
    output_folder,
    n_folds=5,
    debug=False,
    dataset_type="diplomatic",
    image_base_path=None,
):
    print("📊 Processing COCO annotations...")
    with open(input_file, "r") as f:
        data = json.load(f)

    # Filter out invalid annotations at the beginning
    print("🔍 Filtering annotations...")
    valid_annotations = filter_valid_annotations(data["annotations"])
    data["annotations"] = valid_annotations

    layout_categories = [
        d["id"] for d in data["categories"] if d["supercategory"] == "layout"
    ]
    images = data["images"]

    print(f"📸 Found {len(images)} images")

    random.seed(1992)
    random.shuffle(images)

    # Create task-specific PageXML files for all images
    all_data = {
        "images": images,
        "annotations": data["annotations"],
        "categories": data["categories"],
    }

    create_all_pagexml_files(
        output_folder,
        all_data,
        dataset_type,
        image_base_path,
        target_tasks=["ocr", "omr", "layout"],
    )

    fold_size = round(len(images) * 0.2)
    print(f"🔀 Creating {n_folds} folds ({fold_size} images per fold)...")

    for fold in range(n_folds):
        print(f"   ⚙️  Processing fold {fold}...")
        start_idx = fold * fold_size
        end_idx = min((fold + 1) * fold_size, len(images))
        test_index = list(range(start_idx, end_idx))

        fold_data = create_fold_data(
            images,
            data["annotations"],
            data["categories"],
            test_index,
        )
        convert_and_write_coco_anns(
            fold_data,
            fold,
            Path(output_folder),
            "fold",
            layout_categories,
        )

        fold_output_folder = Path(output_folder) / f"train_test_{fold}"
        fold_output_folder.mkdir(exist_ok=True, parents=True)
        train_index = list(set(range(len(images))) - set(test_index))
        val_index = random.sample(train_index, round(len(images) * 0.16))
        train_index = list(set(train_index) - set(val_index))

        train_data = create_fold_data(
            images,
            data["annotations"],
            data["categories"],
            train_index,
        )
        val_data = create_fold_data(
            images,
            data["annotations"],
            data["categories"],
            val_index,
        )

        convert_and_write_coco_anns(
            train_data,
            None,
            fold_output_folder,
            "train",
            layout_categories,
        )
        convert_and_write_coco_anns(
            val_data,
            None,
            fold_output_folder,
            "val",
            layout_categories,
        )
        convert_and_write_coco_anns(
            fold_data,
            None,
            fold_output_folder,
            "test",
            layout_categories,
        )

    print(f"✅ Created {n_folds} folds successfully")


def split_gt_into_train_val(gt_json_path, val_ratio=0.15):
    """
    Split a gt.json file into train.json and val.json files.

    Args:
        gt_json_path: Path to the gt.json file
        val_ratio: Ratio of images to use for validation (default 0.15)

    Returns:
        tuple: (train_json_path, val_json_path)
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    # Filter out invalid annotations at the beginning
    print("🔍 Filtering annotations...")
    valid_annotations = filter_valid_annotations(data["annotations"])

    images = data["images"]
    random.seed(1992)  # For reproducibility
    random.shuffle(images)

    # Calculate split
    val_size = int(len(images) * val_ratio)
    val_images = images[:val_size]
    train_images = images[val_size:]

    # Get image IDs for filtering annotations
    val_image_ids = {img["id"] for img in val_images}
    train_image_ids = {img["id"] for img in train_images}

    # Filter annotations
    val_annotations = [
        ann for ann in valid_annotations if ann["image_id"] in val_image_ids
    ]
    train_annotations = [
        ann for ann in valid_annotations if ann["image_id"] in train_image_ids
    ]

    # Create train and val data structures
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": data.get("categories", []),
    }

    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": data.get("categories", []),
    }

    gt_data = {
        "images": images,
        "annotations": valid_annotations,
        "categories": data.get("categories", []),
    }

    # Write the files
    gt_path = Path(gt_json_path)
    train_json_path = gt_path.parent / "train.json"
    val_json_path = gt_path.parent / "val.json"

    write_json(train_data, train_json_path)
    write_json(val_data, val_json_path)

    create_task_specific_jsons(train_data, gt_path.parent, "train")
    create_task_specific_jsons(val_data, gt_path.parent, "val")
    create_task_specific_jsons(gt_data, gt_path.parent, "test")

    print(
        f"   ✅ Split {gt_path.name}: {len(train_images)} train, {len(val_images)} val images"
    )

    return train_json_path, val_json_path


def split_sequentially(
    input_file,
    output_folder,
    test_fold=0,
    n_folds=5,
    n_steps=10,
    debug=False,
    dataset_type="diplomatic",
    strategy="random_sample",
    image_base_path=None,
):
    print("📊 Processing COCO annotations for sequential learning...")
    with open(input_file, "r") as f:
        data = json.load(f)

    # Filter out invalid annotations at the beginning
    print("🔍 Filtering annotations...")
    valid_annotations = filter_valid_annotations(data["annotations"])

    images = data["images"]
    layout_categories = [
        d["id"] for d in data["categories"] if d["supercategory"] == "layout"
    ]

    if debug:
        images = images[:15]
        print(f"🐛 DEBUG: Limited to {len(images)} images")
        debug_image_ids = {img["id"] for img in images}
        debug_annotations = [
            ann for ann in valid_annotations if ann["image_id"] in debug_image_ids
        ]
    else:
        debug_annotations = valid_annotations
        print(f"📸 Found {len(images)} images")

    # N.B.
    # Use the same folding logic as split_into_folds for consistency
    random.seed(1992)
    image_indices = list(range(len(images)))
    random.shuffle(image_indices)

    fold_size = round(len(images) * (1 / n_folds))
    print(f"🔀 Using fold {test_fold} as fixed test set...")

    start_idx = test_fold * fold_size
    end_idx = min((test_fold + 1) * fold_size, len(images))
    test_indices = image_indices[start_idx:end_idx]
    train_pool_indices = sorted(list(set(image_indices) - set(test_indices)))

    # Create a subfolder for the strategy
    strategy_folder = Path(output_folder) / strategy
    strategy_folder.mkdir(exist_ok=True, parents=True)

    # Create and write fixed test data
    test_data = create_fold_data(
        images, debug_annotations, data["categories"], test_indices
    )
    test_output_folder = strategy_folder / "sequential_test"
    test_output_folder.mkdir(exist_ok=True, parents=True)
    convert_and_write_coco_anns(
        test_data, None, test_output_folder, "test", layout_categories
    )
    print(f"✅ Fixed test set created with {len(test_indices)} images.")

    step_size = round(len(train_pool_indices) / n_steps)
    splits_created = 0
    print(
        f"🔢 Creating sequential training splits (step size: {step_size}, strategy: {strategy})..."
    )

    for i in range(n_steps):
        memory_size = i * step_size
        if i == n_steps - 1:
            step_size = len(train_pool_indices) - memory_size

        if strategy == "random_sample":
            seen_indices = train_pool_indices[:memory_size]
            if memory_size > 0:
                indices_from_memory = random.sample(seen_indices, step_size)
            else:
                indices_from_memory = []
            new_indices = train_pool_indices[memory_size : memory_size + step_size]
            current_indices_for_split = new_indices + indices_from_memory
        elif strategy == "sequential_sample":
            new_indices = train_pool_indices[memory_size : memory_size + step_size]
            current_indices_for_split = new_indices
        else:
            raise ValueError(f"❌ Unknown strategy: {strategy}")

        print(
            f"   ⚙️  Processing split for {memory_size + step_size} training images (effective dataset size: {len(current_indices_for_split)})..."
        )

        total_train_val = len(current_indices_for_split)
        val_size = int(0.2 * total_train_val)

        random.shuffle(current_indices_for_split)
        val_indices = current_indices_for_split[:val_size]
        train_indices = current_indices_for_split[val_size:]

        seq_folder = strategy_folder / f"seq_{splits_created:02d}"
        seq_folder.mkdir(parents=True, exist_ok=True)

        train_data = create_fold_data(
            images, debug_annotations, data["categories"], train_indices
        )
        convert_and_write_coco_anns(
            train_data, None, seq_folder, "train", layout_categories
        )

        val_data = create_fold_data(
            images, debug_annotations, data["categories"], val_indices
        )
        convert_and_write_coco_anns(
            val_data, None, seq_folder, "val", layout_categories
        )
        splits_created += 1

    print(f"✅ Created {splits_created} sequential splits successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Split COCO JSON files into multiple files."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input COCO JSON file."
    )
    parser.add_argument(
        "output_folder", type=str, help="Folder to save the output files."
    )
    parser.add_argument(
        "--folds", action="store_true", help="Split the input file into n folds."
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of folds to create."
    )
    parser.add_argument(
        "--seq",
        action="store_true",
        help="Split the input file into incremental sequential files.",
    )
    parser.add_argument(
        "--test-fold",
        type=int,
        default=0,
        help="The fold to use as a fixed test set for sequential learning.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=10,
        help="Number of steps for sequential learning.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random_sample",
        choices=[
            "random_sample",
            "sequential_sample",
        ],
        help="Strategy for creating sequential training sets.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with limited number of images.",
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        default=None,
        help="Base path where images are located (default: data/I-Ct_91)",
    )

    args = parser.parse_args()

    # data_root = Path(args.input_file).parent
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Determine dataset type from input file path
    dataset_type = "editorial" if "editorial" in args.input_file else "diplomatic"

    # Auto-infer image_base_path from input file location if not provided
    if args.image_base_path is None:
        input_path = Path(args.input_file)
        # Input file is typically at data/DATASET_NAME/annotations-TYPE/gt.json
        # So we go up two levels to get data/DATASET_NAME
        data_dir = input_path.parent.parent
        args.image_base_path = str(data_dir)
        print(f"   🔍 Auto-detected image base path: {args.image_base_path}")

    # Auto-create train/val split if processing gt.json
    input_path = Path(args.input_file)
    if input_path.name == "gt.json":
        print(f"📂 Auto-creating train/val split for {input_path}")
        split_gt_into_train_val(str(input_path))

    if args.folds:
        split_into_folds(
            args.input_file,
            output_folder,
            n_folds=args.n_folds,
            debug=args.debug,
            dataset_type=dataset_type,
            image_base_path=args.image_base_path,
        )
    elif args.seq:
        split_sequentially(
            args.input_file,
            output_folder,
            test_fold=args.test_fold,
            n_folds=args.n_folds,
            n_steps=args.n_steps,
            debug=args.debug,
            dataset_type=dataset_type,
            strategy=args.strategy,
            image_base_path=args.image_base_path,
        )
    else:
        print("❌ Please specify either --folds or --seq option.")


if __name__ == "__main__":
    main()
