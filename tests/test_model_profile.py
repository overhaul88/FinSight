"""Tests for local model profile selection."""

from __future__ import annotations

from src.llm.model import RuntimeCapabilities, recommended_local_model_profile


def test_recommended_profile_for_small_gpu_prefers_1_5b_qwen():
    runtime = RuntimeCapabilities(
        cuda_available=True,
        device_name="GeForce GTX 1650",
        total_memory_gb=4.0,
        cuda_version="12.8",
    )

    profile = recommended_local_model_profile(runtime)

    assert profile["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert profile["llm_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert profile["use_4bit"] is True


def test_recommended_profile_for_cpu_prefers_smallest_llm():
    runtime = RuntimeCapabilities(
        cuda_available=False,
        device_name="CPU",
        total_memory_gb=0.0,
        cuda_version="",
    )

    profile = recommended_local_model_profile(runtime)

    assert profile["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert profile["llm_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert profile["use_4bit"] is False
