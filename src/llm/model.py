"""Model-loading utilities with a development fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator
import asyncio

from src.config import settings


@dataclass
class LLMResponse:
    """Simple response envelope for chain compatibility."""

    content: str


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Minimal runtime capabilities used to choose a practical local model profile."""

    cuda_available: bool
    device_name: str
    total_memory_gb: float
    cuda_version: str


class DevelopmentFallbackLLM:
    """Deterministic fallback when a local model stack is unavailable."""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason or "Local model unavailable."

    def invoke(self, prompt: str) -> LLMResponse:
        question = self._extract_section(prompt, "Question:")
        context = self._extract_section(prompt, "Context Documents:")
        preview = context.split("\n", 1)[0].strip() if context else "No retrieved context."
        answer = (
            "Development fallback response. A production language model is not loaded.\n"
            f"Question: {question or 'Unknown question'}\n"
            f"Context preview: {preview}\n"
            f"Reason: {self.reason}\n"
            "Source: development-fallback"
        )
        return LLMResponse(content=answer)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        for token in self.invoke(prompt).content.split():
            await asyncio.sleep(0)
            yield f"{token} "

    @staticmethod
    def _extract_section(prompt: str, label: str) -> str:
        if label not in prompt:
            return ""
        return prompt.split(label, 1)[1].strip()


class TransformersCausalLLM:
    """Thin adapter around a transformers causal model and tokenizer."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.temperature = temperature

    def invoke(self, prompt: str) -> LLMResponse:
        import torch  # type: ignore

        rendered_prompt = self._render_prompt(prompt)
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        input_device = self._resolve_input_device(torch)
        model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_token_count = model_inputs["input_ids"].shape[1]
        completion_ids = generated_ids[0][prompt_token_count:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return LLMResponse(content=text)

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        for token in self.invoke(prompt).content.split():
            await asyncio.sleep(0)
            yield f"{token} "

    def _render_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are FinSight, a careful assistant for Indian financial regulations. "
                        "Answer only from the retrieved evidence and say when the evidence is insufficient."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _resolve_input_device(self, torch: Any):
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if isinstance(mapped_device, int):
                    return torch.device(f"cuda:{mapped_device}")
                if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                    return torch.device(mapped_device)

        try:
            embeddings = self.model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None:
                return weight.device
        except Exception:
            pass

        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")


def select_inference_dtype(torch: Any):
    """Pick a compute dtype that matches the local accelerator."""

    if not torch.cuda.is_available():
        return torch.float32

    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass

    return torch.float16


def detect_runtime_capabilities() -> RuntimeCapabilities:
    """Inspect the local accelerator and return the small set of values we care about."""

    try:
        import torch  # type: ignore
    except ImportError:
        return RuntimeCapabilities(
            cuda_available=False,
            device_name="CPU",
            total_memory_gb=0.0,
            cuda_version="",
        )

    if not torch.cuda.is_available():
        return RuntimeCapabilities(
            cuda_available=False,
            device_name="CPU",
            total_memory_gb=0.0,
            cuda_version=torch.version.cuda or "",
        )

    try:
        properties = torch.cuda.get_device_properties(0)
        total_memory_gb = round(properties.total_memory / (1024**3), 2)
        device_name = properties.name
    except Exception:
        total_memory_gb = 0.0
        device_name = "CUDA GPU"

    return RuntimeCapabilities(
        cuda_available=True,
        device_name=device_name,
        total_memory_gb=total_memory_gb,
        cuda_version=torch.version.cuda or "",
    )


def recommended_local_model_profile(
    runtime: RuntimeCapabilities | None = None,
) -> dict[str, str | bool]:
    """Return a pragmatic local model choice for the detected runtime."""

    runtime = runtime or detect_runtime_capabilities()

    if not runtime.cuda_available:
        return {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "use_4bit": False,
            "reason": "No CUDA device detected, so the smallest practical local instruct model is the safe default.",
        }

    if runtime.total_memory_gb < 5.0:
        return {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "use_4bit": True,
            "reason": (
                f"{runtime.device_name} exposes about {runtime.total_memory_gb:.1f} GB VRAM, "
                "so BGE-small plus Qwen2.5-1.5B in 4-bit mode is the safest local RAG pair."
            ),
        }

    if runtime.total_memory_gb < 8.0:
        return {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": "Qwen/Qwen2.5-3B-Instruct",
            "use_4bit": True,
            "reason": (
                f"{runtime.device_name} exposes about {runtime.total_memory_gb:.1f} GB VRAM, "
                "which is enough to move up to Qwen2.5-3B in 4-bit mode without oversizing retrieval."
            ),
        }

    return {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "use_4bit": True,
        "reason": (
            f"{runtime.device_name} exposes about {runtime.total_memory_gb:.1f} GB VRAM, "
            "which makes the stronger 7B instruct tier realistic for local grounded generation."
        ),
    }


def load_mistral_with_adapter(
    base_model: str | None = None,
    adapter_path: str | None = None,
    use_4bit: bool = True,
    fallback_to_dummy: bool = True,
) -> Any:
    """Load the local model stack or fall back to a deterministic dev LLM."""

    base_model = base_model or settings.llm_model
    adapter_path = adapter_path or "models/mistral-finsight/adapter"

    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore
    except ImportError as exc:
        if fallback_to_dummy:
            return DevelopmentFallbackLLM(reason=f"Missing model dependencies: {exc}")
        raise

    try:
        compute_dtype = select_inference_dtype(torch)
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            token=settings.huggingface_token or None,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=None if use_4bit else compute_dtype,
            low_cpu_mem_usage=True,
            token=settings.huggingface_token or None,
            trust_remote_code=True,
        )
        model.eval()
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            for field_name in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, field_name):
                    setattr(generation_config, field_name, None)

        adapter_root = Path(adapter_path)
        if (adapter_root / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, adapter_path)

        return TransformersCausalLLM(model=model, tokenizer=tokenizer)
    except Exception as exc:
        if fallback_to_dummy:
            return DevelopmentFallbackLLM(reason=str(exc))
        raise


def load_local_llm(**kwargs: Any) -> Any:
    """Alias for the repository's default local LLM loader."""

    return load_mistral_with_adapter(**kwargs)
