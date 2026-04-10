"""Load and probe the configured embedding model and local LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.ingestion.embedder import EmbeddingModel
from src.llm.model import (
    detect_runtime_capabilities,
    load_local_llm,
    recommended_local_model_profile,
)


def run_probe(prompt: str, use_4bit: bool = True) -> dict:
    started = time.time()
    runtime = detect_runtime_capabilities()
    torch = None
    try:
        import torch as torch_module  # type: ignore

        torch = torch_module
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    embedding_model = EmbeddingModel(settings.embedding_model)
    embedding = embedding_model.embed_query("What is the maximum LTV ratio for gold loans?")

    llm = load_local_llm(
        base_model=settings.llm_model,
        use_4bit=use_4bit,
        fallback_to_dummy=False,
    )
    response = llm.invoke(prompt)

    payload = {
        "runtime": {
            "cuda_available": runtime.cuda_available,
            "device_name": runtime.device_name,
            "total_memory_gb": runtime.total_memory_gb,
            "cuda_version": runtime.cuda_version,
        },
        "recommended_profile": recommended_local_model_profile(runtime),
        "embedding_model": settings.embedding_model,
        "embedding_dim": int(len(embedding)),
        "llm_model": settings.llm_model,
        "response_preview": str(response.content)[:400],
        "duration_seconds": round(time.time() - started, 2),
    }
    if torch is not None and torch.cuda.is_available():
        payload["gpu_memory"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and verify the configured local embedding model and LLM.")
    parser.add_argument(
        "--prompt",
        default="Summarize the RBI rule for gold loan LTV in two short sentences and cite uncertainty if unsupported.",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading for the LLM.")
    args = parser.parse_args()

    result = run_probe(args.prompt, use_4bit=not args.no_4bit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
