import re
from collections import defaultdict
from itertools import combinations


def _experiment_token(experiment_name: str) -> str:
    return experiment_name.split("_", 1)[0]


def _extract_seq_ids(rel_paths: set[str]) -> set[int]:
    seq_ids = set()
    for rel_path in rel_paths:
        match = re.search(r"(?:^|/)seq_?(\d+)(?:/|$)", rel_path)
        if match:
            seq_ids.add(int(match.group(1)))
    return seq_ids


def _path_template(rel_path: str) -> str:
    return re.sub(r"(^|/)seq_?\d+(?=/|$)", r"\1seq_XX", rel_path)


def _seq_id_from_path(rel_path: str):
    match = re.search(r"(?:^|/)seq_?(\d+)(?:/|$)", rel_path)
    if not match:
        return None
    return int(match.group(1))


def _build_seq_ids_by_template(rel_paths: set[str]):
    seq_ids_by_template = defaultdict(set)
    for rel_path in rel_paths:
        seq_id = _seq_id_from_path(rel_path)
        if seq_id is None:
            continue
        seq_ids_by_template[_path_template(rel_path)].add(seq_id)
    return seq_ids_by_template


def _format_seq_ids(seq_ids: set[int]) -> str:
    if not seq_ids:
        return "-"
    return ", ".join(f"{seq_id:02d}" for seq_id in sorted(seq_ids))


def _collect_relative_json_paths(experiment_dir) -> set[str]:
    return {
        json_path.relative_to(experiment_dir).as_posix()
        for json_path in experiment_dir.rglob("*.json")
        if json_path.is_file()
    }


def _find_collisions(experiment_dirs):
    rel_paths_by_name = {}
    names_by_token = defaultdict(list)

    for exp_dir in experiment_dirs:
        rel_paths_by_name[exp_dir.name] = _collect_relative_json_paths(exp_dir)
        names_by_token[_experiment_token(exp_dir.name)].append(exp_dir.name)

    collisions = []
    for exp_names in names_by_token.values():
        if len(exp_names) < 2:
            continue

        for left_name, right_name in combinations(sorted(exp_names), 2):
            left_paths = rel_paths_by_name[left_name]
            right_paths = rel_paths_by_name[right_name]
            duplicates = sorted(left_paths & right_paths)
            if not duplicates:
                continue

            left_seq_ids_by_template = _build_seq_ids_by_template(left_paths)
            right_seq_ids_by_template = _build_seq_ids_by_template(right_paths)
            duplicated_by_template = defaultdict(list)
            for rel_path in duplicates:
                duplicated_by_template[_path_template(rel_path)].append(rel_path)

            template_reports = []
            for template in sorted(duplicated_by_template.keys()):
                left_seq_ids = left_seq_ids_by_template.get(template, set())
                right_seq_ids = right_seq_ids_by_template.get(template, set())
                template_reports.append(
                    {
                        "template": template,
                        "duplicates": sorted(duplicated_by_template[template]),
                        "seq_both": left_seq_ids & right_seq_ids,
                        "seq_left_only": left_seq_ids - right_seq_ids,
                        "seq_right_only": right_seq_ids - left_seq_ids,
                    }
                )

            collisions.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "duplicates": duplicates,
                    "template_reports": template_reports,
                }
            )

    return collisions


def check_for_duplicate_relative_json_paths(experiment_dirs) -> bool:
    collisions = _find_collisions(experiment_dirs)
    if not collisions:
        return False

    print("\nError: repeated relative JSON paths found between comparable experiments.")
    print("Comparable means same first token before '_' in experiment dir name.")
    for collision in collisions:
        left_name = collision["left"]
        right_name = collision["right"]
        duplicates = collision["duplicates"]

        print(
            f"  - {left_name} vs {right_name}: {len(duplicates)} duplicated JSON paths"
        )
        for report in collision["template_reports"]:
            print(f"    path: {report['template']}")
            if (
                report["seq_both"]
                or report["seq_left_only"]
                or report["seq_right_only"]
            ):
                print(
                    "      seq in both: "
                    f"{_format_seq_ids(report['seq_both'])} | "
                    f"only {left_name}: {_format_seq_ids(report['seq_left_only'])} | "
                    f"only {right_name}: {_format_seq_ids(report['seq_right_only'])}"
                )
            print(f"      duplicated files: {len(report['duplicates'])}")

    return True
