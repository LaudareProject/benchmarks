"""
Evaluation module for OCR/OMR/Layout tasks using WER/CER and layout metrics.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET
from itertools import chain
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    sklearn_available = True
except ImportError:
    print(
        "Warning: sklearn not installed. WWER and R-WER metrics will not be available."
    )
    sklearn_available = False

try:
    import fastwer
except ImportError:
    print("Warning: fastwer not installed. Install with: pip install pybind11 fastwer")
    fastwer = None

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    import werpy

    WERPY_AVAILABLE = True
except ImportError:
    WERPY_AVAILABLE = False
    print("Warning: werpy not available, using fallback WWER implementation")

import re
import string


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text properly handling punctuation for historical documents.

    Returns words normalized to lowercase, stripping punctuation but preserving
    alphabetic and numeric characters. Handles contractions and hyphenated words.
    """
    if not text:
        return []

    # Use regex to split on whitespace and punctuation, but keep word boundaries
    # This handles contractions, hyphens, and various punctuation better than split()
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out empty strings and single characters that aren't meaningful
    return [word for word in words if len(word) > 0 and word.isalnum()]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1, box2: Bounding boxes in format [x, y, width, height]

    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2) format
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]

    # Calculate intersection
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def extract_layout_from_pagexml(xml_file: Path) -> List[Dict]:
    """
    Extract layout regions from a PageXML file.

    Args:
        xml_file: Path to the PageXML file

    Returns:
        List of dictionaries containing region information
    """
    regions = []

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Dynamically handle PageXML namespace
        ns = {}
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0][1:]
            ns = {"page": namespace}

        # Extract TextRegion elements
        region_types = ["TextRegion"]

        for region_type in region_types:
            xpath = f".//{region_type}" if not ns else f".//page:{region_type}"
            for region in root.findall(xpath, ns):
                region_id = region.get("id", "")
                # The 'type' attribute holds the class. Default to a generic 'Text'.
                region_class = region.get("type", "Text")

                # Extract coordinates from Coords element
                coords_elem = region.find("./Coords" if not ns else "./page:Coords", ns)
                if coords_elem is not None:
                    points_attr = coords_elem.get("points", "")
                    if points_attr:
                        # Parse points to get bounding box
                        bbox = parse_points_to_bbox(points_attr)
                        if bbox:
                            regions.append(
                                {
                                    "id": region_id,
                                    "type": region_class,
                                    "bbox": bbox,  # [x, y, width, height]
                                    "confidence": 1.0,  # PageXML doesn't have confidence scores
                                }
                            )

        # Also extract TextLine elements, which might be predicted by some tools
        xpath_textline = ".//page:TextLine" if ns else ".//TextLine"
        for text_line in root.findall(xpath_textline, ns):
            coords_elem = text_line.find("./Coords" if not ns else "./page:Coords", ns)
            if coords_elem is not None:
                points_attr = coords_elem.get("points", "")
                if points_attr:
                    bbox = parse_points_to_bbox(points_attr)
                    if bbox:
                        regions.append(
                            {
                                "id": text_line.get("id", ""),
                                "type": "TextLine",
                                "bbox": bbox,
                                "confidence": 1.0,
                            }
                        )

    except Exception as e:
        print(f"Error extracting layout from {xml_file}: {e}")

    return regions


def parse_points_to_bbox(points_str: str) -> Optional[List[float]]:
    """
    Convert PageXML points string to bounding box.

    Args:
        points_str: String of points like "x1,y1 x2,y2 x3,y3..."

    Returns:
        Bounding box as [x, y, width, height] or None if parsing fails
    """
    try:
        points = []
        for point_str in points_str.split():
            x, y = map(float, point_str.split(","))
            points.append((x, y))

        if not points:
            return None

        # Calculate bounding box from points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return [x_min, y_min, x_max - x_min, y_max - y_min]

    except Exception as e:
        print(f"Error parsing points {points_str}: {e}")
        return None


def load_layout_annotations(json_file: Path) -> Dict[str, List[Dict]]:
    """
    Load layout annotations from COCO-style JSON file.

    Args:
        json_file: Path to layout annotation JSON file

    Returns:
        Dict mapping image names to list of annotations
    """
    annotations = {}

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both COCO format and our custom format
        if "images" in data and "annotations" in data:
            # COCO format
            image_info = {img["id"]: img["file_name"] for img in data["images"]}
            category_info = {
                cat["id"]: cat["name"] for cat in data.get("categories", [])
            }

            for ann in data["annotations"]:
                img_id = ann["image_id"]
                if img_id in image_info:
                    img_name = Path(image_info[img_id]).stem

                    if img_name not in annotations:
                        annotations[img_name] = []

                    annotations[img_name].append(
                        {
                            "bbox": ann["bbox"],  # [x, y, width, height]
                            "category_id": ann["category_id"],
                            "category_name": category_info.get(
                                ann["category_id"], "unknown"
                            ),
                            "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                        }
                    )

        elif "data_list" in data:
            # Our custom format
            for item in data["data_list"]:
                img_path = item.get("img_path", "")
                img_name = Path(img_path).stem

                annotations[img_name] = []
                for instance in item.get("instances", []):
                    bbox = instance.get("bbox", [])
                    if len(bbox) == 4:
                        annotations[img_name].append(
                            {
                                "bbox": bbox,
                                "category_id": instance.get("bbox_label", 0),
                                "category_name": f"class_{instance.get('bbox_label', 0)}",
                                "area": bbox[2] * bbox[3],
                            }
                        )

    except Exception as e:
        print(f"Error loading layout annotations from {json_file}: {e}")

    return annotations


def calculate_layout_metrics_at_threshold(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for layout detection at a specific IoU threshold.

    Args:
        predictions: List of predicted regions
        ground_truth: List of ground truth regions
        iou_threshold: IoU threshold for considering a detection as correct
        class_names: List of class names for per-class metrics

    Returns:
        Dict containing precision, recall, F1 metrics
    """
    if not predictions and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    if not ground_truth:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "tp": 0,
            "fp": len(predictions),
            "fn": 0,
        }

    if not predictions:
        return {
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(ground_truth),
        }

    # Sort predictions by confidence (if available)
    predictions_sorted = sorted(
        predictions, key=lambda x: x.get("confidence", 1.0), reverse=True
    )

    tp = 0  # True positives
    fp = 0  # False positives
    matched_gt = set()  # Keep track of matched ground truth boxes

    # For each prediction, find the best matching ground truth
    for pred in predictions_sorted:
        pred_bbox = pred["bbox"]
        pred_class = pred.get("type", "unknown")

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue  # Already matched

            gt_bbox = gt["bbox"]
            gt_class = gt.get("category_name", "unknown")

            if pred_class.lower() != gt_class.lower():
                continue

            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if the best match exceeds the threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)  # False negatives

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def calculate_map(
    all_predictions: List[List[Dict]],
    matched_files: List[str],
    ground_truth_json: Path,
    iou_thresholds: Optional[List[float]] = None,
):
    """
    Calculate COCO metrics using pycocotools COCOeval.

    Args:
        all_predictions: List of prediction lists (one per image)
        matched_files: List of matched file stems in same order as predictions
        ground_truth_json: Ground truth COCO JSON path
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dict containing mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    coco_gt = COCO(str(ground_truth_json))

    image_id_by_stem = {
        Path(img["file_name"]).stem: img_id for img_id, img in coco_gt.imgs.items()
    }
    category_id_by_name = {
        cat["name"].lower(): cat_id for cat_id, cat in coco_gt.cats.items()
    }

    image_ids = [
        image_id_by_stem[stem] for stem in matched_files if stem in image_id_by_stem
    ]
    category_ids = list(coco_gt.cats.keys())

    detections = []
    for preds, stem in zip(all_predictions, matched_files):
        if stem not in image_id_by_stem:
            continue

        image_id = image_id_by_stem[stem]
        for pred in preds:
            pred_type = pred.get("type", "").lower()
            if pred_type not in category_id_by_name:
                continue

            detections.append(
                {
                    "image_id": image_id,
                    "category_id": category_id_by_name[pred_type],
                    "bbox": pred["bbox"],
                    "score": float(pred.get("confidence", 1.0)),
                }
            )

    if not image_ids or not category_ids:
        return {
            "mAP": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "mAP@[0.5:0.95]": 0.0,
            "AP_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "precision_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "recall_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "f1_per_threshold": {float(t): 0.0 for t in iou_thresholds},
        }

    if not detections:
        return {
            "mAP": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "mAP@[0.5:0.95]": 0.0,
            "AP_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "precision_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "recall_per_threshold": {float(t): 0.0 for t in iou_thresholds},
            "f1_per_threshold": {float(t): 0.0 for t in iou_thresholds},
        }

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = category_ids
    coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])
    map_75 = float(coco_eval.stats[2])

    return {
        "mAP": map_50_95,
        "mAP@0.5": map_50,
        "mAP@0.75": map_75,
        "mAP@[0.5:0.95]": map_50_95,
    }


def evaluate_layout_predictions(
    predictions_dir: Path,
    ground_truth_json: Path,
    iou_thresholds: Optional[List[float]] = None,
    output_file: Optional[Path] = None,
):
    """
    Evaluate layout predictions against ground truth.

    Args:
        predictions_dir: Directory containing PageXML prediction files
        ground_truth_json: JSON file containing ground truth layout annotations
        iou_thresholds: List of IoU thresholds for evaluation
        output_file: Optional file to save evaluation results

    Returns:
        Dict containing evaluation metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    print("Evaluating layout predictions...")

    # Load ground truth annotations
    ground_truth_data = load_layout_annotations(ground_truth_json)
    if not ground_truth_data:
        print(f"No ground truth data found in {ground_truth_json}")
        return {}

    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    matched_files = []

    for pred_file in predictions_dir.glob("*.xml"):
        base_name = pred_file.stem

        if base_name in ground_truth_data:
            predictions = extract_layout_from_pagexml(pred_file)
            ground_truth = ground_truth_data[base_name]

            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)
            matched_files.append(base_name)
        else:
            print(f"Warning: No ground truth found for {base_name}")

    if not all_predictions:
        print("No matching predictions found!")
        return {}

    print(f"Found {len(all_predictions)} matching prediction-reference pairs")

    # Calculate metrics at different thresholds
    results = {}

    # Calculate metrics for each IoU threshold
    for threshold in iou_thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0

        for preds, gts in zip(all_predictions, all_ground_truth):
            metrics = calculate_layout_metrics_at_threshold(preds, gts, threshold)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[f"precision@{threshold:.2f}"] = precision
        results[f"recall@{threshold:.2f}"] = recall
        results[f"f1@{threshold:.2f}"] = f1

    map_results = calculate_map(
        all_predictions, matched_files, ground_truth_json, iou_thresholds
    )

    results.update(map_results)

    # Add summary statistics
    results["num_samples"] = len(all_predictions)
    results["matched_files"] = len(matched_files)

    # Print results
    print("\n=== Layout Evaluation Results ===")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"Precision@0.5: {results['precision@0.50']:.4f}")
    print(f"Recall@0.5: {results['recall@0.50']:.4f}")
    print(f"F1@0.5: {results['f1@0.50']:.4f}")
    print(f"Evaluated samples: {len(all_predictions)}")

    # Save detailed results
    if output_file:
        print(f"   -> Saving layout evaluation results to: {output_file}")
        detailed_results = {"metrics": results, "sample_details": []}

        for i, (preds, gts, filename) in enumerate(
            zip(all_predictions, all_ground_truth, matched_files)
        ):
            sample_metrics = calculate_layout_metrics_at_threshold(preds, gts, 0.5)
            detailed_results["sample_details"].append(
                {
                    "filename": filename,
                    "num_predictions": len(preds),
                    "num_ground_truth": len(gts),
                    "precision": sample_metrics["precision"],
                    "recall": sample_metrics["recall"],
                    "f1": sample_metrics["f1"],
                    "predictions": preds,
                    "ground_truth": gts,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"Detailed results saved to {output_file}")

    return results


def extract_text_from_pagexml(xml_file: Path) -> str:
    """
    Extract text content from a PageXML file.

    Args:
        xml_file: Path to the PageXML file

    Returns:
        str: Extracted text content
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Dynamically handle PageXML namespace
        ns = {}
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0][1:]
            ns = {"page": namespace}

        # Extract all text from TextLine elements
        text_lines = []
        xpath = ".//TextLine" if not ns else ".//page:TextLine"
        for text_line in root.findall(xpath, ns):
            text_equiv_xpath = (
                "./TextEquiv/Unicode" if not ns else "./page:TextEquiv/page:Unicode"
            )
            text_equiv = text_line.find(text_equiv_xpath, ns)
            if text_equiv is not None and text_equiv.text:
                text_lines.append(text_equiv.text.strip())

        return " ".join(text_lines)
    except Exception as e:
        print(f"Error extracting text from {xml_file}: {e}")
        return ""


def extract_text_from_prediction_file(pred_file: Path) -> str:
    """
    Extract text from a prediction output file.

    Args:
        pred_file: Path to the prediction file (.txt or .xml)

    Returns:
        str: Extracted prediction text
    """
    try:
        if pred_file.suffix == ".xml":
            return extract_text_from_pagexml(pred_file)
        else:
            # Assume it's a plain text file with one prediction per line
            with open(pred_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return " ".join(line.strip() for line in lines)
    except Exception as e:
        print(f"Error reading prediction file {pred_file}: {e}")
        return ""


def load_ground_truth_from_json(json_file: Path) -> Dict[str, str]:
    """
    Load ground truth text from task-specific JSON files.

    Args:
        json_file: Path to the JSON annotation file (e.g., ocr_test.json)

    Returns:
        Dict mapping image names to ground truth text
    """
    ground_truth = {}
    image_texts = {}  # img_name -> list of texts

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle COCO format
        if "images" in data and "annotations" in data:
            image_info = {
                img["id"]: Path(img["file_name"]).stem for img in data["images"]
            }

            for ann in data["annotations"]:
                img_id = ann.get("image_id")
                img_name = image_info.get(img_id)
                text = ann.get("description", "").strip()

                if img_name and text:
                    if img_name not in image_texts:
                        image_texts[img_name] = []
                    image_texts[img_name].append(text)

        # Handle custom "data_list" format ala mmdet
        elif "data_list" in data:
            for item in data.get("data_list", []):
                img_path = item.get("img_path", "")
                img_name = Path(img_path).stem

                if img_name not in image_texts:
                    image_texts[img_name] = []

                for instance in item.get("instances", []):
                    text = instance.get("text", "").strip()
                    if text:
                        image_texts[img_name].append(text)

        # Join all texts for each image
        for img_name, texts in image_texts.items():
            if texts:
                ground_truth[img_name] = " ".join(texts)

    except Exception as e:
        print(f"Error loading ground truth from {json_file}: {e}")

    return ground_truth


def _get_cache_paths(ground_truth_json_path: Path) -> Tuple[Path, Path]:
    """
    Get cache file paths for TF-IDF weights and word frequencies.

    Args:
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        Tuple of (tfidf_cache_path, word_freq_cache_path)
    """
    gt_dir = ground_truth_json_path.parent
    gt_stem = ground_truth_json_path.stem

    tfidf_cache = gt_dir / f"{gt_stem}_tfidf_weights.json"
    word_freq_cache = gt_dir / f"{gt_stem}_word_frequencies.json"

    return tfidf_cache, word_freq_cache


def _is_cache_valid(cache_path: Path, ground_truth_json_path: Path) -> bool:
    """
    Check if cache file is valid (exists and newer than ground truth).

    Args:
        cache_path: Path to cache file
        ground_truth_json_path: Path to ground truth JSON file

    Returns:
        bool: True if cache is valid
    """
    if not cache_path.exists():
        return False

    try:
        cache_mtime = cache_path.stat().st_mtime
        gt_mtime = ground_truth_json_path.stat().st_mtime
        return cache_mtime > gt_mtime
    except Exception:
        return False


def _load_cache(cache_path: Path) -> Optional[Dict]:
    """
    Load data from cache file.

    Args:
        cache_path: Path to cache file

    Returns:
        Dict with cached data or None if loading fails
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return None


def _save_cache(data: Dict, cache_path: Path) -> None:
    """
    Save data to cache file in readable JSON format.

    Args:
        data: Data to cache
        cache_path: Path to cache file
    """
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")


def calculate_extended_ocr_metrics(
    predictions: List[str],
    references: List[str],
    matched_files: List[str],
    ground_truth_json_path: Path,
) -> Dict[str, float]:
    """
    Calculate extended OCR metrics: WWER and R-WER-n.

    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings
        matched_files: List of matched file names
        ground_truth_json_path: Path to ground truth JSON file for caching

    Returns:
        Dict containing WWER and R-WER metrics
    """
    if not sklearn_available or fastwer is None:
        print("Warning: sklearn or fastwer not available. Skipping extended metrics.")
        return {}

    if len(predictions) != len(references) or len(predictions) != len(matched_files):
        print("Warning: Mismatch in predictions, references, and matched files lengths")
        return {}

    # Get cache paths
    tfidf_cache_path, word_freq_cache_path = _get_cache_paths(ground_truth_json_path)

    # Try to load cached data
    tfidf_weights = None
    word_frequencies = None

    if _is_cache_valid(tfidf_cache_path, ground_truth_json_path):
        tfidf_weights = _load_cache(tfidf_cache_path)

    if _is_cache_valid(word_freq_cache_path, ground_truth_json_path):
        word_frequencies = _load_cache(word_freq_cache_path)

    # Calculate TF-IDF weights if not cached
    if tfidf_weights is None:
        print("Calculating TF-IDF weights...")
        tfidf_weights = {}

        if references:  # Only calculate if we have references
            try:
                # Create TF-IDF vectorizer (each page is a document)
                vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"\b\w+\b")
                tfidf_matrix = vectorizer.fit_transform(references)
                feature_names = vectorizer.get_feature_names_out()

                # Extract weights for each page (document)
                for i, (ref, filename) in enumerate(zip(references, matched_files)):
                    page_weights = {}
                    tfidf_vector = tfidf_matrix[i].toarray()[0]

                    # Map words to their TF-IDF weights
                    for j, weight in enumerate(tfidf_vector):
                        if weight > 0:  # Only store non-zero weights
                            word = feature_names[j]
                            page_weights[word] = float(weight)

                    tfidf_weights[filename] = page_weights

                # Save to cache
                _save_cache(tfidf_weights, tfidf_cache_path)

            except Exception as e:
                print(f"Error calculating TF-IDF weights: {e}")
                tfidf_weights = {}

    # Calculate word frequencies if not cached
    if word_frequencies is None:
        print("Calculating word frequencies...")
        word_frequencies = {}

        try:
            # Count all words across all references
            all_words = []
            for ref in references:
                # Proper word tokenization (matching TF-IDF tokenizer)
                words = tokenize_text(ref)
                all_words.extend(words)

            # Count frequencies
            word_counter = Counter(all_words)
            word_frequencies = dict(word_counter)

            # Save to cache
            _save_cache(word_frequencies, word_freq_cache_path)

        except Exception as e:
            print(f"Error calculating word frequencies: {e}")
            word_frequencies = {}

    # Calculate WWER (Weighted WER)
    wwer = _calculate_wwer(predictions, references, matched_files, tfidf_weights)

    # Calculate R-WER for different thresholds
    r_wer_2 = _calculate_r_wer(predictions, references, word_frequencies, threshold=2)
    r_wer_4 = _calculate_r_wer(predictions, references, word_frequencies, threshold=4)
    r_wer_8 = _calculate_r_wer(predictions, references, word_frequencies, threshold=8)

    # Calculate p-CER (Punctuation Character Error Rate)
    p_cer = _calculate_p_cer(predictions, references)

    return {
        "WWER": wwer,
        "R-WER-2": r_wer_2,
        "R-WER-4": r_wer_4,
        "R-WER-8": r_wer_8,
        "p-CER": p_cer,
    }


def _calculate_wwer(
    predictions: List[str],
    references: List[str],
    matched_files: List[str],
    tfidf_weights: Dict[str, Dict[str, float]],
) -> float:
    """
    Calculate Weighted WER using TF-IDF weights with word-level alignment.

    Uses werpy for precise word-level alignment and weights individual error words
    by their TF-IDF scores rather than using page-level averaging.
    """
    if not tfidf_weights:
        return 0.0

    if WERPY_AVAILABLE:
        return _calculate_wwer_werpy(
            predictions, references, matched_files, tfidf_weights
        )
    else:
        return _calculate_wwer_fallback(
            predictions, references, matched_files, tfidf_weights
        )


def _calculate_wwer_werpy(
    predictions: List[str],
    references: List[str],
    matched_files: List[str],
    tfidf_weights: Dict[str, Dict[str, float]],
) -> float:
    """
    Calculate WWER using werpy for precise word-level alignment.
    """
    total_weighted_errors = 0.0
    total_reference_weights = 0.0

    def add_fallback_weighted_error(pred_text, ref_text, page_weights):
        if fastwer is None:
            return 0.0, 0.0
        ref_words = tokenize_text(ref_text)
        if not ref_words:
            return 0.0, 0.0
        page_word_weights = [page_weights.get(word, 0.0) for word in ref_words]
        avg_weight = (
            sum(page_word_weights) / len(page_word_weights)
            if page_word_weights
            else 0.0
        )
        sent_wer = fastwer.score_sent(pred_text, ref_text, char_level=False)
        return sent_wer * avg_weight, avg_weight

    for pred, ref, filename in zip(predictions, references, matched_files):
        if filename not in tfidf_weights:
            continue

        page_weights = tfidf_weights[filename]

        try:
            # Use werpy summary for word-level alignment
            summary = werpy.summary(ref, pred)
            if not hasattr(summary, "iloc") or summary.empty:
                weighted_error, ref_weight_sum = add_fallback_weighted_error(
                    pred, ref, page_weights
                )
                total_weighted_errors += weighted_error
                total_reference_weights += ref_weight_sum
                continue

            row = summary.iloc[0]
            deleted_words = row.get("deleted_words", [])
            substituted_words = row.get("substituted_words", [])

            # Get reference words for total weight calculation using proper tokenization
            ref_words = tokenize_text(ref)

            # Calculate total reference weight for this sentence
            ref_weight_sum = sum(page_weights.get(word, 0.0) for word in ref_words)
            total_reference_weights += ref_weight_sum

            # Weight deleted words by their TF-IDF scores
            for deleted_word in deleted_words:
                # Handle different types that werpy might return
                clean_words = tokenize_text(str(deleted_word))
                for clean_word in clean_words:
                    word_weight = page_weights.get(clean_word, 0.0)
                    total_weighted_errors += word_weight * 100.0

            # Weight substituted reference words by their TF-IDF scores
            for ref_word, pred_word in substituted_words:
                # Handle tuples from werpy - tokenize the reference word
                clean_ref_words = tokenize_text(str(ref_word))
                for clean_ref_word in clean_ref_words:
                    word_weight = page_weights.get(clean_ref_word, 0.0)
                    total_weighted_errors += word_weight * 100.0

        except Exception as e:
            print(f"Error calculating WWER with werpy for {filename}: {e}")
            weighted_error, ref_weight_sum = add_fallback_weighted_error(
                pred, ref, page_weights
            )
            total_weighted_errors += weighted_error
            total_reference_weights += ref_weight_sum
            continue

    if total_reference_weights == 0:
        return 0.0

    wwer = total_weighted_errors / total_reference_weights
    return wwer


def _calculate_wwer_fallback(
    predictions: List[str],
    references: List[str],
    matched_files: List[str],
    tfidf_weights: Dict[str, Dict[str, float]],
) -> float:
    """
    Fallback WWER calculation when werpy is not available.
    Uses the original page-level averaging approach with improved tokenization.
    """
    total_weighted_errors = 0.0
    total_weights = 0.0

    for pred, ref, filename in zip(predictions, references, matched_files):
        if filename not in tfidf_weights:
            continue

        page_weights = tfidf_weights[filename]

        try:
            # Calculate WER for this sentence pair
            sent_wer = fastwer.score_sent(pred, ref, char_level=False)

            # Get reference words using proper tokenization
            ref_words = tokenize_text(ref)
            if not ref_words:
                continue

            # Calculate average TF-IDF weight for this page
            page_word_weights = [page_weights.get(word, 0.0) for word in ref_words]
            avg_weight = (
                sum(page_word_weights) / len(page_word_weights)
                if page_word_weights
                else 0.0
            )

            # Weight the error by average TF-IDF weight of reference words
            weighted_error = sent_wer * avg_weight
            total_weighted_errors += weighted_error

            # Sum weights for normalization
            total_weights += avg_weight

        except Exception as e:
            print(f"Error calculating WWER fallback for {filename}: {e}")
            continue

    if total_weights == 0:
        return 0.0

    wwer = total_weighted_errors / total_weights
    return wwer


def _calculate_r_wer(
    predictions: List[str],
    references: List[str],
    word_frequencies: Dict[str, int],
    threshold: int,
) -> float:
    """Calculate Rare WER for words with frequency <= threshold."""
    if not word_frequencies:
        return 0.0

    # Filter rare words
    rare_words = {word for word, freq in word_frequencies.items() if freq <= threshold}

    if not rare_words:
        return 0.0

    # Extract rare word sequences from predictions and references
    rare_predictions = []
    rare_references = []

    for pred, ref in zip(predictions, references):
        # Proper tokenization handling punctuation
        pred_words = tokenize_text(pred)
        ref_words = tokenize_text(ref)

        # Keep only rare words (maintaining order)
        rare_pred = [word for word in pred_words if word in rare_words]
        rare_ref = [word for word in ref_words if word in rare_words]

        rare_predictions.append(" ".join(rare_pred))
        rare_references.append(" ".join(rare_ref))

    # Calculate WER on rare word sequences
    try:
        r_wer = fastwer.score(rare_predictions, rare_references, char_level=False)
        return r_wer  # fastwer already returns percentage
    except Exception as e:
        print(f"Error calculating R-WER-{threshold}: {e}")
        return 0.0


def extract_punctuation_sequence(text: str) -> str:
    """
    Extract non-alphanumeric characters (punctuation) from text, preserving order.

    This includes spaces, punctuation marks, and special symbols but excludes
    regular letters and numbers.
    """
    if not text:
        return ""

    # Keep only non-alphanumeric characters
    punctuation_chars = [char for char in text if not char.isalnum()]
    return "".join(punctuation_chars)


def _calculate_p_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Punctuation Character Error Rate (p-CER).

    Extracts non-alphanumeric character sequences from predictions and references,
    then calculates character-level error rate on punctuation only.
    """
    if not predictions or not references:
        return 0.0

    # Extract punctuation sequences from predictions and references
    punct_predictions = []
    punct_references = []

    for pred, ref in zip(predictions, references):
        punct_pred = extract_punctuation_sequence(pred)
        punct_ref = extract_punctuation_sequence(ref)

        punct_predictions.append(punct_pred)
        punct_references.append(punct_ref)

    # Check if there's any punctuation to evaluate
    total_punct_chars = sum(len(ref) for ref in punct_references)
    if total_punct_chars == 0:
        return 0.0  # No punctuation in references

    # Calculate CER on punctuation sequences
    try:
        p_cer = fastwer.score(punct_predictions, punct_references, char_level=True)
        return p_cer  # fastwer returns percentage
    except Exception as e:
        print(f"Error calculating p-CER: {e}")
        return 0.0


def calculate_wer_cer(
    predictions: List[str], references: List[str]
) -> Tuple[float, float]:
    """
    Calculate Word Error Rate (WER) and Character Error Rate (CER).

    Args:
        predictions: List of predicted text strings
        references: List of reference (ground truth) text strings

    Returns:
        Tuple of (WER, CER) as fractions (0.0 to 1.0)
    """
    if fastwer is None:
        print("Error: fastwer not available. Cannot calculate WER/CER.")
        return 0.0, 0.0

    if len(predictions) != len(references):
        print(
            f"Warning: Mismatch in number of predictions ({len(predictions)}) and references ({len(references)})"
        )
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

    if not predictions or not references:
        print("Warning: Empty predictions or references")
        return 0.0, 0.0

    try:
        # Calculate corpus-level WER and CER
        wer = fastwer.score(predictions, references, char_level=False)
        cer = fastwer.score(predictions, references, char_level=True)
        return wer, cer
    except Exception as e:
        print(f"Error calculating WER/CER: {e}")
        return 0.0, 0.0


def evaluate_ocr_predictions(
    debug,
    ground_truth_json: Path,
    output_file: Optional[Path],
    predictions_dir: Path,
    task: str,
):
    # Handle OCR/OMR tasks
    print(f"Evaluating {task.upper()} predictions...")

    # Load ground truth
    ground_truth = load_ground_truth_from_json(ground_truth_json)
    if not ground_truth:
        print(f"No ground truth data found in {ground_truth_json}")
        return {}

    # Collect predictions
    predictions = []
    references = []
    matched_files = []

    prediction_files = chain(
        predictions_dir.glob("*.xml"), predictions_dir.glob("*.pred.txt")
    )
    for pred_file in prediction_files:
        # Extract base name
        if pred_file.suffix == ".xml":
            base_name = pred_file.stem
        else:
            base_name = pred_file.name.replace(".pred.txt", "")

        # This check avoids processing the same file if it's already been matched
        if base_name in matched_files:
            continue

        if base_name in ground_truth:
            pred_text = extract_text_from_prediction_file(pred_file)
            ref_text = ground_truth[base_name]

            predictions.append(pred_text)
            references.append(ref_text)
            matched_files.append(base_name)
        else:
            print(f"Warning: No ground truth found for {base_name}")

    if not predictions:
        print("No matching predictions found!")
        return {}

    print(f"Found {len(predictions)} matching prediction-reference pairs")

    # Calculate metrics
    wer, cer = calculate_wer_cer(predictions, references)
    if debug:
        print(f"DEBUG: WER={wer}%, CER={cer}%")

    # For OMR, WER is equivalent to Note Error Rate (NER)
    metric_name = "NER" if task == "omr" else "WER"

    # Calculate p-CER for both OCR and OMR
    p_cer = _calculate_p_cer(predictions, references)

    results = {
        f"{metric_name}": wer,
        "CER": cer,
        "p-CER": p_cer,
        "num_samples": len(predictions),
        "matched_files": len(matched_files),
    }

    # Calculate extended metrics for OCR tasks only
    if task == "ocr":
        extended_metrics = calculate_extended_ocr_metrics(
            predictions, references, matched_files, ground_truth_json
        )
        results.update(extended_metrics)

    # Print results
    print(f"\n=== {task.upper()} Evaluation Results ===")
    print(f"{metric_name}: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print(f"p-CER: {p_cer:.2f}%")

    # Display extended metrics for OCR only
    if task == "ocr" and "WWER" in results:
        print(f"WWER: {results['WWER']:.2f}%")
        print(f"R-WER-2: {results['R-WER-2']:.2f}%")
        print(f"R-WER-4: {results['R-WER-4']:.2f}%")
        print(f"R-WER-8: {results['R-WER-8']:.2f}%")

    print(f"Evaluated samples: {len(predictions)}")

    # Save detailed results if requested
    if output_file:
        print(f"   -> Saving {task.upper()} evaluation results to: {output_file}")
        detailed_results = {"metrics": results, "sample_details": []}

        for i, (pred, ref, filename) in enumerate(
            zip(predictions, references, matched_files)
        ):
            if fastwer:
                sample_wer = fastwer.score_sent(pred, ref, char_level=False)
                sample_cer = fastwer.score_sent(pred, ref, char_level=True)
            else:
                sample_wer = sample_cer = 0.0

            detailed_results["sample_details"].append(
                {
                    "filename": filename,
                    "prediction": pred,
                    "reference": ref,
                    f"sample_{metric_name}": sample_wer,
                    "sample_CER": sample_cer,
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        print(f"Detailed results saved to {output_file}")

    return results


def evaluate_predictions(
    predictions_dir: Path,
    ground_truth_json: Path,
    task: str = "ocr",
    output_file: Optional[Path] = None,
    debug=False,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth and calculate appropriate metrics.

    Args:
        predictions_dir: Directory containing prediction files
        ground_truth_json: JSON file containing ground truth annotations
        task: Task type ("ocr", "omr", or "layout")
        output_file: Optional file to save evaluation results

    Returns:
        Dict containing evaluation metrics
    """
    if task == "layout":
        return evaluate_layout_predictions(
            predictions_dir, ground_truth_json, output_file=output_file
        )
    else:
        return evaluate_ocr_predictions(
            debug, ground_truth_json, output_file, predictions_dir, task
        )


def evaluate_predictions_entry(
    predictions_dir: str,
    ground_truth_json: str,
    task: str,
    output_file: Optional[str] = None,
    debug: bool = False,
):
    """
    Entry point for evaluation using expanded parameters.

    Args:
        predictions_dir: Path to directory containing prediction files
        ground_truth_json: Path to JSON file with ground truth annotations
        task: Task type ("ocr", "omr", or "layout")
        output_file: Optional path to output file for detailed results

    Returns:
        Dict containing evaluation metrics
    """
    predictions_dir_path = Path(predictions_dir)
    ground_truth_json_path = Path(ground_truth_json)
    output_file_path = Path(output_file) if output_file else None

    if debug:
        print(f"DEBUG: Predictions dir: {predictions_dir_path}")
        print(f"DEBUG: Ground truth JSON: {ground_truth_json_path}")
        print(f"DEBUG: Task: {task}")
        if output_file_path:
            print(f"DEBUG: Output file: {output_file_path}")
    return evaluate_predictions(
        predictions_dir_path,
        ground_truth_json_path,
        task,
        output_file_path,
        debug=debug,
    )
