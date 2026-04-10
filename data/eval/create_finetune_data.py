"""Generate deterministic fine-tuning data for FinSight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


QA_PAIRS = [
    {
        "instruction": "What is the minimum KYC documentation required for opening a bank account under RBI guidance?",
        "input": "",
        "output": (
            "RBI KYC guidance requires proof of identity and proof of address before account "
            "activation, subject to any permitted simplified due diligence exception. "
            "Source: RBI KYC guidance, customer identification requirements."
        ),
    },
    {
        "instruction": "What penalties may apply for insider trading violations under SEBI rules?",
        "input": "",
        "output": (
            "SEBI may pursue civil penalties, criminal exposure, and restrictions on market "
            "access for insider trading violations depending on the facts and applicable provisions. "
            "Source: SEBI insider trading framework."
        ),
    },
    {
        "instruction": "What is the maximum loan-to-value ratio referenced in RBI gold loan guidance?",
        "input": "",
        "output": (
            "The maximum loan-to-value ratio for gold loans referenced in RBI guidance is 75 percent. "
            "Source: RBI gold loan guidance."
        ),
    },
    {
        "instruction": "When must a regulated entity collect proof of address under RBI KYC requirements?",
        "input": "",
        "output": (
            "A regulated entity must collect proof of address before activating the account unless "
            "a valid simplified due diligence exception applies. Source: RBI KYC guidance."
        ),
    },
]


def create_sharegpt_format(qa_pairs: Iterable[dict]) -> List[dict]:
    """Convert QA pairs into ShareGPT-style conversations."""

    formatted = []
    for pair in qa_pairs:
        formatted.append(
            {
                "conversations": [
                    {"from": "human", "value": pair["instruction"]},
                    {"from": "gpt", "value": pair["output"]},
                ]
            }
        )
    return formatted


def write_finetune_datasets(output_dir: str = "data/finetune", qa_pairs: Iterable[dict] | None = None) -> dict:
    """Write train, validation, and full datasets to disk."""

    dataset = create_sharegpt_format(qa_pairs or QA_PAIRS)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    split_index = max(1, int(len(dataset) * 0.8))
    train = dataset[:split_index]
    val = dataset[split_index:]
    if not val:
        val = dataset[-1:]
        train = dataset[:-1] or dataset[:1]

    payloads = {
        "finsight_train.json": dataset,
        "train.json": train,
        "val.json": val,
    }
    for filename, payload in payloads.items():
        with (destination / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    return {
        "total_examples": len(dataset),
        "train_examples": len(train),
        "val_examples": len(val),
        "output_dir": str(destination),
    }


if __name__ == "__main__":
    stats = write_finetune_datasets()
    for key, value in stats.items():
        print(f"{key}: {value}")

