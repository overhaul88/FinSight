"""Tests for fine-tuning helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.finetune import FineTuningConfig, load_dataset_from_json, run_finetune


def test_create_finetune_data_writes_expected_files(tmp_path):
    from data.eval.create_finetune_data import write_finetune_datasets

    stats = write_finetune_datasets(output_dir=str(tmp_path))

    assert stats["total_examples"] >= 4
    assert (tmp_path / "train.json").exists()
    assert (tmp_path / "val.json").exists()
    assert (tmp_path / "finsight_train.json").exists()


def test_load_dataset_from_json_formats_records(tmp_path):
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    payload = [
        {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        }
    ]
    train_path.write_text(json.dumps(payload), encoding="utf-8")
    val_path.write_text(json.dumps(payload), encoding="utf-8")

    train_records, val_records = load_dataset_from_json(str(train_path), str(val_path))

    assert train_records == [{"text": "[INST] Question [/INST] Answer</s>"}]
    assert val_records == [{"text": "[INST] Question [/INST] Answer</s>"}]


def test_run_finetune_dry_run_writes_adapter_contract(tmp_path):
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    payload = [
        {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer"},
            ]
        }
    ]
    train_path.write_text(json.dumps(payload), encoding="utf-8")
    val_path.write_text(json.dumps(payload), encoding="utf-8")

    config = FineTuningConfig(
        output_dir=str(tmp_path / "model"),
        train_path=str(train_path),
        val_path=str(val_path),
    )
    result = run_finetune(config=config, dry_run=True)

    metadata_path = Path(result["adapter_path"]) / "dry_run_metadata.json"
    assert result["mode"] == "dry_run"
    assert metadata_path.exists()

