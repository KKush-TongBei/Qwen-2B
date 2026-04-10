from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InferenceConfigError(ValueError):
    """Raised when inference configuration is invalid."""


DEFAULT_IMAGE_PROMPT = "请详细描述图片。"


@dataclass(frozen=True)
class InferenceConfig:
    model: str
    cpu: bool = False
    max_gpus: int = 2
    cuda_devices: str | None = None
    per_gpu_memory: str = "31GiB"
    flash_attn: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_model_path() -> str:
    return str(repo_root() / "Qwen3-VL-2B")


def _physical_gpu_count() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode != 0:
            return None
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return len(lines) if lines else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def apply_visible_gpus(config: InferenceConfig) -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch."""
    if config.cpu:
        return
    if config.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_devices.strip()
        return
    if config.max_gpus <= 0:
        return
    n_phys = _physical_gpu_count()
    n = config.max_gpus if n_phys is None else min(config.max_gpus, n_phys)
    if n <= 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))


def _ensure_model_weights(model_path: str) -> None:
    p = Path(model_path)
    if not p.is_dir():
        return
    has_weights = any(p.glob("*.safetensors")) or any(p.glob("*.bin"))
    if has_weights:
        return
    raise InferenceConfigError(
        f"目录中未找到权重文件（*.safetensors / *.bin）：{model_path}。"
        "请将 model.safetensors 放到该目录，或使用 Hub 模型 ID。"
    )


class QwenVLDescriber:
    def __init__(self, config: InferenceConfig):
        apply_visible_gpus(config)
        self.config = config
        self.model_path = config.model
        self.model: Any | None = None
        self.processor: Any | None = None
        self.device: Any | None = None
        self._torch: Any | None = None

    def load(self) -> None:
        _ensure_model_weights(self.model_path)

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        load_kwargs: dict[str, Any] = {"dtype": "auto"}
        if self.config.cpu:
            load_kwargs["device_map"] = {"": "cpu"}
        else:
            load_kwargs["device_map"] = "auto"
            if (
                self.config.per_gpu_memory
                and torch.cuda.is_available()
                and torch.cuda.device_count() > 0
            ):
                cap = self.config.per_gpu_memory.strip()
                if cap:
                    load_kwargs["max_memory"] = {
                        i: cap for i in range(torch.cuda.device_count())
                    }
                    load_kwargs["max_memory"]["cpu"] = "256GiB"

        if self.config.flash_attn and not self.config.cpu:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        except Exception as e:
            if self.config.flash_attn:
                raise InferenceConfigError(
                    "Flash Attention 2 加载失败，请关闭 flash_attn 或安装 flash-attn。"
                ) from e
            raise

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.device = self.model.device
        self._torch = torch

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def describe(self, image: str | Path, prompt: str, max_new_tokens: int = 256) -> str:
        if max_new_tokens <= 0:
            raise InferenceConfigError("max_new_tokens 必须大于 0。")
        if not self.is_loaded():
            self.load()

        assert self.processor is not None
        assert self.model is not None
        assert self.device is not None
        assert self._torch is not None

        image_value = str(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_value},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return text[0] if text else ""
