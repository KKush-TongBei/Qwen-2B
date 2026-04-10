#!/usr/bin/env python3
"""Describe an image with Qwen3-VL-2B-Instruct (CUDA when available)."""

from __future__ import annotations

import argparse
import sys

from app.inference import (
    DEFAULT_IMAGE_PROMPT,
    InferenceConfig,
    InferenceConfigError,
    QwenVLDescriber,
    default_model_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe an image with Qwen3-VL.")
    p.add_argument(
        "image",
        nargs="?",
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="Image path or URL (default: official demo image).",
    )
    p.add_argument(
        "--model",
        default=default_model_path(),
        help="Model directory or Hugging Face model id (default: ./Qwen3-VL-2B).",
    )
    p.add_argument(
        "--prompt",
        default=DEFAULT_IMAGE_PROMPT,
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
    config = InferenceConfig(
        model=args.model,
        cpu=args.cpu,
        max_gpus=args.max_gpus,
        cuda_devices=args.cuda_devices,
        per_gpu_memory=args.per_gpu_memory,
        flash_attn=args.flash_attn,
    )
    describer = QwenVLDescriber(config)
    try:
        text = describer.describe(
            image=args.image,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print(text)
        return 0
    except InferenceConfigError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
