#!/usr/bin/env python3
"""
This script will analyze the datasets in `data` to print statistics and create bar plots.
Run with `--help/-h` to view options.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image


LAYOUT_CATEGORIES = {"musicText", "text", "line", "staff"}
LAYOUT_COLOR = "#c0392b"
MUSIC_COLOR = "#2ecc71"
EDITION_ORDER = ["editorial", "diplomatic"]
EDITION_HATCH = {"editorial": "//", "diplomatic": ""}
TITLE_BY_DATASET = {
    "I-Ct_91": "I-CT 91",
    "I-Fn_BR_18": "I-FN BR 18",
}
LABEL_OVERRIDES = {
    "text": "Text Region",
    "line": "Text Line",
    "musicText": "Music & Text Region",
    "staff": "Music Staff",
    "neume_1_note": "Neume 1 Note",
    "neume_2_notes": "Neume 2 Notes",
    "neume_3_notes": "Neume 3 Notes",
    "neume_gt4_notes": "Neume >4 Notes",
}


def load_coco(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def count_line_staff_text(annotations: list[dict], name_by_id: dict[int, str]) -> dict:
    stats = {
        "line": {"annotations": 0, "chars": 0, "words": 0, "parens": 0},
        "staff": {"annotations": 0, "chars": 0, "words": 0, "parens": 0},
    }
    for ann in annotations:
        name = name_by_id[ann["category_id"]]
        if name in stats:
            desc = ann.get("description", "")
            stats[name]["annotations"] += 1
            stats[name]["chars"] += len(desc)
            stats[name]["words"] += len(desc.split())
            stats[name]["parens"] += desc.count("(")
    return stats


def count_neume_notes_and_parens(
    annotations: list[dict], name_by_id: dict[int, str]
) -> dict:
    totals = {"neumes": 0, "notes": 0, "parens": 0}
    for ann in annotations:
        if name_by_id[ann["category_id"]] != "neume":
            continue
        desc = ann.get("description", "")
        totals["neumes"] += 1
        totals["parens"] += desc.count("(")
        text = desc.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        totals["notes"] += len([token for token in text.split() if token])
    return totals


def tokenize_staff_description(text: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if char == "(":
            end = text.find(")", i)
            if end == -1:
                end = len(text) - 1
            tokens.append(text[i : end + 1].strip())
            i = end + 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def report_staff_neume_mismatches(
    data: dict, name_by_id: dict[int, str], edition: str
) -> None:
    image_name_by_id = {
        img["id"]: img.get("file_name", str(img["id"])) for img in data["images"]
    }
    staff_by_image = defaultdict(list)
    neumes_by_image = Counter()

    annotations = data["annotations"]
    for idx, ann in enumerate(annotations):
        name = name_by_id[ann["category_id"]]
        image_id = ann["image_id"]
        if name == "neume":
            neumes_by_image[image_id] += 1
        elif name == "staff":
            desc = ann.get("description", "")
            staff_by_image[image_id].append(
                {
                    "text": desc,
                    "parens": desc.count("("),
                    "annotation_index": idx,
                }
            )

    staff_neumes = defaultdict(list)
    staff_indices_by_image = defaultdict(list)
    current_staff = None
    current_image_id = None
    for idx, ann in enumerate(annotations):
        image_id = ann["image_id"]
        name = name_by_id[ann["category_id"]]
        if name == "staff":
            current_staff = {
                "text": ann.get("description", ""),
                "parens": ann.get("description", "").count("("),
                "annotation_index": idx,
                "image_id": image_id,
                "neumes": 0,
                "reconstructed": [],
            }
            staff_neumes[image_id].append(current_staff)
            staff_indices_by_image[image_id].append(idx)
            current_image_id = image_id
        elif (
            name == "neume"
            and current_staff is not None
            and image_id == current_image_id
        ):
            current_staff["neumes"] += 1

    music_objects = {"clef", "neume", "custos", "musicDelimiter"}
    for image_id, staff_list in staff_neumes.items():
        indices = staff_indices_by_image[image_id]
        for pos, staff in enumerate(staff_list):
            start = staff["annotation_index"] + 1
            end = indices[pos + 1] if pos + 1 < len(indices) else len(annotations)
            tokens = []
            for ann in annotations[start:end]:
                if ann["image_id"] != image_id:
                    break
                name = name_by_id[ann["category_id"]]
                if name in music_objects:
                    desc = ann.get("description", "").strip()
                    if desc:
                        tokens.append(desc)
            staff["reconstructed"] = tokens

    for image_id, staff_list in sorted(staff_neumes.items()):
        total_parens = sum(item["parens"] for item in staff_list)
        neumes = neumes_by_image.get(image_id, 0)
        if edition == "editorial" and neumes < total_parens:
            continue
        if total_parens != neumes:
            image_name = image_name_by_id.get(image_id, str(image_id))
            print(
                "staff_neume_mismatch: "
                f"edition={edition} image_id={image_id} file_name={image_name} "
                f"staff_parens={total_parens} neumes={neumes}"
            )
            for item in staff_list:
                staff_tokens = tokenize_staff_description(item["text"])
                reconstructed_tokens = item.get("reconstructed", [])
                staff_counts = Counter(staff_tokens)
                reconstructed_counts = Counter(reconstructed_tokens)
                missing = staff_counts - reconstructed_counts
                extra = reconstructed_counts - staff_counts
                print(
                    "  staff_neumes="
                    f"{item['neumes']} staff_parens={item['parens']} text={item['text']}"
                )
                print(f"  reconstructed={' '.join(reconstructed_tokens)}")
                if missing:
                    missing_text = " ".join(
                        f"{token}x{count}" for token, count in missing.items()
                    )
                    print(f"  missing_tokens={missing_text}")
                if extra:
                    extra_text = " ".join(
                        f"{token}x{count}" for token, count in extra.items()
                    )
                    print(f"  extra_tokens={extra_text}")


def neume_bucket(description: str) -> str:
    text = description.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    count = len([token for token in text.split() if token])
    if count <= 1:
        return "neume_1_note"
    if count == 2:
        return "neume_2_notes"
    if count == 3:
        return "neume_3_notes"
    return "neume_gt4_notes"


def normalized_category(name: str, ann: dict) -> str:
    if name == "neume":
        return neume_bucket(ann.get("description", ""))
    return name


def count_categories(annotations: list[dict], name_by_id: dict[int, str]) -> Counter:
    counts = Counter()
    for ann in annotations:
        raw_name = name_by_id[ann["category_id"]]
        counts[normalized_category(raw_name, ann)] += 1
    return counts


def avg_pair(values: list[tuple[float, float]]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    sx = sum(v[0] for v in values)
    sy = sum(v[1] for v in values)
    return (sx / len(values), sy / len(values))


def image_dpi_from_file(image_path: Path) -> float | None:
    if not image_path.exists():
        return None
    with Image.open(image_path) as img:
        dpi = img.info.get("dpi")
        if isinstance(dpi, (list, tuple)) and dpi:
            return float(sum(dpi) / len(dpi))
        if isinstance(dpi, (int, float)):
            return float(dpi)

        jfif_unit = img.info.get("jfif_unit")
        jfif_density = img.info.get("jfif_density")
        if isinstance(jfif_density, (list, tuple)) and jfif_density:
            density = float(sum(jfif_density) / len(jfif_density))
            if jfif_unit == 1:
                return density
            if jfif_unit == 2:
                return density * 2.54
    return None


def average_image_metrics(images: list[dict], image_root: Path) -> dict:
    image_sizes = [
        (float(img["width"]), float(img["height"]))
        for img in images
        if "width" in img and "height" in img
    ]
    avg_width, avg_height = avg_pair(image_sizes)

    dpi_values = []
    for img in images:
        if "dpi" in img:
            dpi = img["dpi"]
            if isinstance(dpi, (int, float)):
                dpi_values.append(float(dpi))
            elif isinstance(dpi, (list, tuple)) and dpi:
                dpi_values.append(float(sum(dpi) / len(dpi)))
        elif "dpi_x" in img and "dpi_y" in img:
            dpi_values.append(float((img["dpi_x"] + img["dpi_y"]) / 2))
        else:
            file_name = img.get("file_name")
            if file_name:
                file_dpi = image_dpi_from_file(image_root / file_name)
                if file_dpi is not None:
                    dpi_values.append(file_dpi)

    avg_dpi = (sum(dpi_values) / len(dpi_values)) if dpi_values else 0.0
    return {
        "avg_width": avg_width,
        "avg_height": avg_height,
        "avg_area": avg_width * avg_height,
        "avg_dpi": avg_dpi,
    }


def average_region_metrics(annotations: list[dict], name_by_id: dict[int, str]) -> dict:
    by_name = {"line": [], "staff": []}
    for ann in annotations:
        name = name_by_id[ann["category_id"]]
        if name not in by_name:
            continue
        if "bbox" not in ann:
            continue
        _, _, w, h = ann["bbox"]
        by_name[name].append((float(w), float(h)))

    out = {}
    for name, sizes in by_name.items():
        avg_width, avg_height = avg_pair(sizes)
        out[name] = {
            "avg_width": avg_width,
            "avg_height": avg_height,
            "avg_area": avg_width * avg_height,
        }
    return out


def annotated_images(images: list[dict], annotations: list[dict]) -> list[dict]:
    annotated_ids = {ann["image_id"] for ann in annotations}
    return [img for img in images if img["id"] in annotated_ids]


def print_stats(
    title: str,
    images: int,
    total_annotations: int,
    category_counts: Counter,
    line_staff_stats: dict,
    neume_stats: dict,
    image_metrics: dict,
    region_metrics: dict,
) -> None:
    print(f"== {title} ==")
    print(f"images: {images}")
    print(f"annotations: {total_annotations}")
    print(
        "average_image_resolution_px: "
        f"{image_metrics['avg_width']:.2f}x{image_metrics['avg_height']:.2f} "
        f"(area={image_metrics['avg_area']:.2f})"
    )
    print(f"average_image_dpi: {image_metrics['avg_dpi']:.2f}")
    for label in ("line", "staff"):
        stats = region_metrics[label]
        print(
            f"average_{label}_resolution_px: "
            f"{stats['avg_width']:.2f}x{stats['avg_height']:.2f} "
            f"(area={stats['avg_area']:.2f})"
        )
    print("annotations_per_category:")
    for name, count in sorted(category_counts.items()):
        print(f"  {name}: {count}")
    for label in ("line", "staff"):
        stats = line_staff_stats[label]
        print(
            f"{label}_description: annotations={stats['annotations']} chars={stats['chars']} "
            f"words={stats['words']} parens={stats['parens']}"
        )
    staff_parens = line_staff_stats["staff"]["parens"]
    print(
        "staff_parens_vs_neumes: "
        f"staff_parens={staff_parens} neume_annotations={neume_stats['neumes']} "
        f"diff={staff_parens - neume_stats['neumes']}"
    )
    print(
        "neume_description: "
        f"annotations={neume_stats['neumes']} notes={neume_stats['notes']} "
        f"parens={neume_stats['parens']}"
    )
    print("")


def display_label(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    return name.replace("_", " ").title()


def plot_category_distributions(
    dataset_order: list[str],
    dataset_counts: dict[str, dict[str, Counter]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(dataset_order), 1, figsize=(12, 6 * len(dataset_order))
    )
    if len(dataset_order) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, dataset_order):
        edition_counts = dataset_counts[dataset]
        edition_keys = [e for e in EDITION_ORDER if e in edition_counts]
        if not edition_keys:
            continue
        all_categories = set()
        for counter in edition_counts.values():
            all_categories.update(counter.keys())
        totals = {
            name: sum(counter.get(name, 0) for counter in edition_counts.values())
            for name in all_categories
        }
        categories = sorted(all_categories, key=lambda name: totals[name], reverse=True)
        labels = [display_label(name) for name in categories]
        x = list(range(len(categories)))
        width = 0.4

        for idx, edition in enumerate(edition_keys):
            counts = [edition_counts[edition].get(name, 0) for name in categories]
            colors = [
                LAYOUT_COLOR if name in LAYOUT_CATEGORIES else MUSIC_COLOR
                for name in categories
            ]
            offset = (-width / 2) if idx == 0 else (width / 2)
            bar_positions = [pos + offset for pos in x]
            ax.bar(
                [pos + offset for pos in x],
                counts,
                width=width,
                color=colors,
                hatch=EDITION_HATCH.get(edition, ""),
                label=edition,
            )
            max_count = max(counts) if counts else 1
            y_offset = max(1, int(max_count * 0.01))
            for pos, value in zip(bar_positions, counts):
                ax.text(
                    pos,
                    value + y_offset,
                    str(value),
                    ha="center",
                    va="bottom",
                    rotation=45,
                    fontsize=8,
                )

        ax.set_title(TITLE_BY_DATASET.get(dataset, dataset))
        ax.set_ylabel("Annotations")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        color_legend = [
            Patch(color=LAYOUT_COLOR, label="Layout Categories"),
            Patch(color=MUSIC_COLOR, label="Music Object categories"),
        ]
        color_legend = ax.legend(handles=color_legend, loc="upper right")
        ax.add_artist(color_legend)
        edition_legend = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=EDITION_HATCH.get(edition, ""),
                label=edition,
            )
            for edition in edition_keys
        ]
        ax.legend(handles=edition_legend, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved category distribution plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute annotation statistics for gt.json files"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--pretrain-dir", type=Path, default=Path("data/pretrain_data"))
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("results/annotation_category_distribution.png"),
    )
    args = parser.parse_args()

    gt_paths = sorted(args.data_dir.glob("*/annotations-*/gt.json"))
    dataset_counts: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for path in gt_paths:
        data = load_coco(path)
        name_by_id = {c["id"]: c["name"] for c in data["categories"]}
        dataset = path.parts[path.parts.index("data") + 1]
        edition = path.parent.name.replace("annotations-", "")

        annotated_image_list = annotated_images(data["images"], data["annotations"])
        category_counts = count_categories(data["annotations"], name_by_id)
        line_staff_stats = count_line_staff_text(data["annotations"], name_by_id)
        neume_stats = count_neume_notes_and_parens(data["annotations"], name_by_id)
        image_metrics = average_image_metrics(annotated_image_list, path.parent.parent)
        region_metrics = average_region_metrics(data["annotations"], name_by_id)

        print_stats(
            title=str(path),
            images=len(annotated_image_list),
            total_annotations=len(data["annotations"]),
            category_counts=category_counts,
            line_staff_stats=line_staff_stats,
            neume_stats=neume_stats,
            image_metrics=image_metrics,
            region_metrics=region_metrics,
        )
        report_staff_neume_mismatches(data, name_by_id, edition)

        dataset_counts[dataset][edition].update(category_counts)

    dataset_order = ["I-Ct_91", "I-Fn_BR_18"]
    plot_category_distributions(dataset_order, dataset_counts, args.plot_path)

    pretrain_jsons = sorted(args.pretrain_dir.glob("**/annotations-diplomatic/*.json"))
    pretrain_counts = Counter()
    for path in pretrain_jsons:
        data = load_coco(path)
        name_by_id = {c["id"]: c["name"] for c in data["categories"]}
        for ann in data["annotations"]:
            name = name_by_id[ann["category_id"]]
            if name in ("line", "staff"):
                pretrain_counts[name] += 1

    print("== pretraining_data ==")
    print(f"line: {pretrain_counts['line']}")
    print(f"staff: {pretrain_counts['staff']}")


if __name__ == "__main__":
    main()
