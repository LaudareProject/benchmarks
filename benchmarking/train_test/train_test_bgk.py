"""
BGK-inspired OMR wrapper over existing recognizers.
"""

from __future__ import annotations

import importlib

from ..bgk_postprocess import postprocess_prediction_dir

BACKENDS = {
    "calamari": ("benchmarking.train_test.train_test_calamari", "train_test_calamari"),
    "kraken": ("benchmarking.train_test.train_test_kraken", "train_test_kraken"),
    "trocr": ("benchmarking.train_test.train_test_trocr", "train_test_trocr"),
}
DEFAULT_BACKEND = "calamari"


def _parse_model_identifier(model_identifier: str) -> tuple[str, str]:
    if "::" not in model_identifier:
        return DEFAULT_BACKEND, model_identifier
    backend, inner = model_identifier.split("::", 1)
    backend = backend.strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported BGK backend: {backend}")
    return backend, inner


def _load_backend(backend: str):
    module_name, function_name = BACKENDS[backend]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def train_test_bgk(
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
    if args.task != "omr":
        raise ValueError("bgk supports only task=omr")

    backend, inner_identifier = _parse_model_identifier(model_identifier)
    print(f"🧠 BGK-inspired post-processing over backend={backend}")

    _load_backend(backend)(
        args=args,
        is_train_test_mode=is_train_test_mode,
        is_sequential=is_sequential,
        output_dir=output_dir,
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        save_model_path=save_model_path,
        load_model_path=load_model_path,
        model_identifier=inner_identifier,
    )

    if test_json is None:
        return

    total, changed = postprocess_prediction_dir(output_dir)
    print(f"✅ BGK post-processing updated {changed}/{total} prediction files")
