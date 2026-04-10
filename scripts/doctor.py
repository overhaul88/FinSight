"""Environment and asset diagnostics for FinSight."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, List
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings


def _check_command(command: str) -> Dict[str, str]:
    path = shutil.which(command)
    return {"available": str(bool(path)).lower(), "path": path or ""}


def _check_paths() -> Dict[str, Dict[str, str]]:
    targets = {
        ".env": ROOT / ".env",
        ".env.example": ROOT / ".env.example",
        "data/raw": ROOT / "data" / "raw",
        "data/processed": ROOT / "data" / "processed",
        "data/eval/qa_pairs.json": ROOT / "data" / "eval" / "qa_pairs.json",
        "data/finetune/train.json": ROOT / "data" / "finetune" / "train.json",
        "data/finetune/val.json": ROOT / "data" / "finetune" / "val.json",
        "models/mistral-finsight/adapter": ROOT / "models" / "mistral-finsight" / "adapter",
    }
    result: Dict[str, Dict[str, str]] = {}
    for label, path in targets.items():
        result[label] = {
            "exists": str(path.exists()).lower(),
            "type": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
            "path": str(path),
        }
    return result


def _check_optional_modules() -> Dict[str, str]:
    modules = [
        "bitsandbytes",
        "faiss",
        "fastapi",
        "httpx",
        "mlflow",
        "peft",
        "pytest",
        "ragas",
        "sentence_transformers",
        "torch",
        "torchvision",
        "transformers",
        "trl",
    ]
    status: Dict[str, str] = {}
    for module in modules:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stderr(io.StringIO()):
                    __import__(module)
            status[module] = "installed"
        except Exception:
            status[module] = "missing"
    return status


def gather_diagnostics() -> Dict[str, object]:
    return {
        "project_root": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "settings": {
            "environment": settings.environment,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "enable_llm_query_expansion": settings.enable_llm_query_expansion,
            "faiss_index_path": settings.faiss_index_path,
            "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        },
        "commands": {
            "docker": _check_command("docker"),
            "git": _check_command("git"),
            "make": _check_command("make"),
        },
        "paths": _check_paths(),
        "optional_modules": _check_optional_modules(),
        "notes": build_notes(),
    }


def build_notes() -> List[str]:
    notes: List[str] = []
    venv_python = ROOT / ".venv" / "bin" / "python"
    if not (ROOT / ".env").exists():
        notes.append("Create .env from .env.example before running service commands.")
    if venv_python.exists() and not str(sys.executable).startswith(str(venv_python.parent)):
        notes.append("Run the doctor with `.venv/bin/python scripts/doctor.py` after activating the project virtualenv for dependency-accurate module checks.")
    if not shutil.which("docker"):
        notes.append("Docker is not installed; compose validation and container runs are unavailable.")
    if not (ROOT / settings.faiss_index_path / "index.faiss").exists():
        notes.append("Build a local FAISS index with `python3 scripts/ingest.py` before starting the API.")
    if not (ROOT / "data" / "finetune" / "train.json").exists():
        notes.append("Generate fine-tuning data with `python3 data/eval/create_finetune_data.py`.")
    if not (ROOT / "models" / "mistral-finsight" / "adapter").exists():
        notes.append("Run `python3 scripts/finetune.py --dry-run` or a real training job to create adapter artifacts.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Show local FinSight environment diagnostics.")
    parser.add_argument("--json", action="store_true", help="Emit diagnostics as JSON")
    args = parser.parse_args()

    diagnostics = gather_diagnostics()
    if args.json:
        print(json.dumps(diagnostics, indent=2))
        return

    print("FinSight Doctor")
    print(f"Project root: {diagnostics['project_root']}")
    print(f"Python: {diagnostics['python_executable']} ({diagnostics['python_version']})")
    print("Commands:")
    for name, payload in diagnostics["commands"].items():
        print(f"  {name}: available={payload['available']} path={payload['path']}")
    print("Key paths:")
    for name, payload in diagnostics["paths"].items():
        print(f"  {name}: exists={payload['exists']} type={payload['type']}")
    print("Optional modules:")
    for name, status in diagnostics["optional_modules"].items():
        print(f"  {name}: {status}")
    if diagnostics["notes"]:
        print("Notes:")
        for note in diagnostics["notes"]:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
