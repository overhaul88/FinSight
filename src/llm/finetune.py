"""QLoRA fine-tuning entry points for FinSight."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.config import settings


@dataclass
class FineTuningConfig:
    """Hyperparameters and paths for adapter training."""

    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "models/mistral-finsight"
    train_path: str = "data/finetune/train.json"
    val_path: str = "data/finetune/val.json"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001


def load_dataset_from_json(train_path: str, val_path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load ShareGPT-style JSON data and render it into training text records."""

    def _load(path: str) -> List[Dict[str, Any]]:
        with Path(path).open(encoding="utf-8") as handle:
            return list(json.load(handle))

    def _format(examples: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        for example in examples:
            conversations = example.get("conversations", [])
            human = next(item["value"] for item in conversations if item["from"] == "human")
            assistant = next(item["value"] for item in conversations if item["from"] == "gpt")
            records.append({"text": f"[INST] {human} [/INST] {assistant}</s>"})
        return records

    return _format(_load(train_path)), _format(_load(val_path))


def _maybe_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError:
        return None
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return mlflow


def _write_dry_run_artifacts(config: FineTuningConfig, train_records: List[Dict[str, str]], val_records: List[Dict[str, str]]) -> Dict[str, Any]:
    output_root = Path(config.output_dir)
    adapter_dir = output_root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "status": "dry_run",
        "base_model": config.base_model,
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "target_modules": config.target_modules,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
    }
    with (adapter_dir / "dry_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "adapter_path": str(adapter_dir),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "mode": "dry_run",
    }


def run_finetune(config: FineTuningConfig | None = None, dry_run: bool = False) -> Dict[str, Any]:
    """Run QLoRA fine-tuning or produce a dry-run adapter contract."""

    config = config or FineTuningConfig(base_model=settings.llm_model)
    train_records, val_records = load_dataset_from_json(config.train_path, config.val_path)

    if dry_run:
        return _write_dry_run_artifacts(config, train_records, val_records)

    try:
        import torch  # type: ignore
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Fine-tuning dependencies are not installed. Use --dry-run or install the full stack.") from exc

    mlflow = _maybe_mlflow()
    run_context = mlflow.start_run(run_name="qlora-finetune") if mlflow else None
    if run_context:
        run_context.__enter__()
        mlflow.log_params(asdict(config))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            token=settings.huggingface_token or None,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization,
            device_map="auto",
            token=settings.huggingface_token or None,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora)

        train_dataset = Dataset.from_list(train_records)
        val_dataset = Dataset.from_list(val_records)

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            fp16=False,
            bf16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=["mlflow"] if mlflow else [],
            gradient_checkpointing=True,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            packing=False,
        )
        result = trainer.train()

        adapter_dir = Path(config.output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        payload = {
            "adapter_path": str(adapter_dir),
            "train_examples": len(train_records),
            "val_examples": len(val_records),
            "mode": "train",
            "train_loss": getattr(result, "training_loss", None),
        }
        if mlflow:
            mlflow.log_metrics({"train_examples": len(train_records), "val_examples": len(val_records)})
        return payload
    finally:
        if run_context:
            run_context.__exit__(None, None, None)

