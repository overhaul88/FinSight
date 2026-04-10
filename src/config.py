"""Central configuration for FinSight."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _load_env_file(env_path: str = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    log_level: str = "INFO"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    enable_llm_query_expansion: bool = False
    huggingface_token: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    top_k_rerank: int = 3
    pinecone_api_key: str = ""
    pinecone_environment: str = ""
    pinecone_index_name: str = "finsight-docs"
    faiss_index_path: str = "data/faiss_index"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "finsight-rag-eval"
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "finsight-prod"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_file()
    return Settings(
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        enable_llm_query_expansion=_get_bool("ENABLE_LLM_QUERY_EXPANSION", False),
        huggingface_token=os.getenv("HUGGINGFACE_TOKEN", ""),
        chunk_size=_get_int("CHUNK_SIZE", 512),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 64),
        top_k_retrieval=_get_int("TOP_K_RETRIEVAL", 5),
        top_k_rerank=_get_int("TOP_K_RERANK", 3),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "finsight-docs"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "finsight-rag-eval"),
        langchain_tracing_v2=_get_bool("LANGCHAIN_TRACING_V2", True),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "finsight-prod"),
        raw_data_dir=os.getenv("RAW_DATA_DIR", "data/raw"),
        processed_data_dir=os.getenv("PROCESSED_DATA_DIR", "data/processed"),
    )


settings = get_settings()
