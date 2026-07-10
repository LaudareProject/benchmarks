from __future__ import annotations

from ..bgk_omr import run_bgk_omr_pipeline


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

    print("🧠 BGK-OMR Laudare-adapted pipeline")
    run_bgk_omr_pipeline(
        args=args,
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        output_dir=output_dir,
        save_model_path=save_model_path,
        load_model_path=load_model_path,
    )
