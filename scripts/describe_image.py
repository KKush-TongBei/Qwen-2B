#!/usr/bin/env python3
"""Describe an image with Qwen3-VL-2B-Instruct (CUDA when available)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


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


def _apply_visible_gpus(args: argparse.Namespace) -> None:
    """Set CUDA_VISIBLE_DEVICES before importing torch."""
    if args.cpu:
        return
    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices.strip()
        return
    if args.max_gpus <= 0:
        return
    n_phys = _physical_gpu_count()
    n = args.max_gpus if n_phys is None else min(args.max_gpus, n_phys)
    if n <= 0:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))


def parse_args() -> argparse.Namespace:
    default_model = _repo_root() / "Qwen3-VL-2B"
    p = argparse.ArgumentParser(description="Describe an image with Qwen3-VL.")
    p.add_argument(
        "image",
        nargs="?",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image path or URL (default: official demo image).",
    )
    p.add_argument(
        "--model",
        default=str(default_model),
        help="Model directory or Hugging Face model id (default: ./Qwen3-VL-2B).",
    )
    p.add_argument(
        "--prompt",
        default="请详细描述这张图片。",
        help="User text prompt.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (default: use CUDA if available).",
    )
    p.add_argument(
        "--max-gpus",
        type=int,
        default=2,
        metavar="N",
        help="最多使用物理 GPU 数量：可见设备为 0..N-1（默认 2）。设为 0 表示不修改可见 GPU（沿用环境变量）。",
    )
    p.add_argument(
        "--cuda-devices",
        default=None,
        metavar="IDS",
        help='覆盖 --max-gpus，直接设置 CUDA_VISIBLE_DEVICES，例如 "2,3"。',
    )
    p.add_argument(
        "--per-gpu-memory",
        default="31GiB",
        metavar="STR",
        help='device_map=auto 时每卡上限（默认 31GiB，适合 32GB 卡留余量）。设为 "" 关闭。',
    )
    p.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use flash_attention_2 (requires flash-attn; CUDA).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _apply_visible_gpus(args)

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not args.cpu and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU。如需 GPU，请安装带 CUDA 的 PyTorch。", file=sys.stderr)

    load_kwargs: dict = {
        "dtype": "auto",
    }
    if args.cpu:
        load_kwargs["device_map"] = {"": "cpu"}
    else:
        load_kwargs["device_map"] = "auto"
        if (
            args.per_gpu_memory
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 0
        ):
            cap = args.per_gpu_memory.strip()
            if cap:
                load_kwargs["max_memory"] = {
                    i: cap for i in range(torch.cuda.device_count())
                }
                load_kwargs["max_memory"]["cpu"] = "256GiB"

    if args.flash_attn and not args.cpu:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model_path = args.model
    if Path(model_path).is_dir():
        missing_weights = not any(
            Path(model_path).glob("*.safetensors")
        ) and not any(Path(model_path).glob("*.bin"))
        if missing_weights:
            print(
                f"目录中未找到权重文件（*.safetensors / *.bin）：{model_path}\n"
                "请将 model.safetensors 放到该目录，或使用 --model Qwen/Qwen3-VL-2B-Instruct 从 Hub 下载。",
                file=sys.stderr,
            )
            return 1

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs,
        )
    except Exception as e:
        if args.flash_attn:
            print(
                "Flash Attention 2 加载失败，请去掉 --flash-attn 或安装 flash-attn。\n"
                f"原始错误: {e}",
                file=sys.stderr,
            )
            return 1
        raise

    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(text[0] if text else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
