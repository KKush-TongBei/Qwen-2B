#!/usr/bin/env python3
"""Quick functional test script for Qwen-VL-Chat.

Covers four tasks in one run:
1) image description
2) visual question answering
3) text reading from image (OCR-like)
4) grounding with bbox output and saved image
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import sys
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _patch_stream_generator_if_needed():
    """Provide a minimal fallback for old Qwen dynamic imports.

    Some Qwen remote modeling files import `transformers_stream_generator`.
    In modern Python/transformers environments this package can be version
    incompatible. This shim keeps loading unblocked for basic `model.chat`.
    """
    if "transformers_stream_generator" in sys.modules:
        return

    shim = types.ModuleType("transformers_stream_generator")

    def init_stream_support(*args, **kwargs):  # noqa: ANN002, ANN003
        if args:
            return args[0]
        return None

    shim.init_stream_support = init_stream_support
    sys.modules["transformers_stream_generator"] = shim


def load_model(model_name_or_path: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _patch_stream_generator_if_needed()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device,
        trust_remote_code=True,
    ).eval()
    return tokenizer, model, device


def run_single_turn(tokenizer, model, image: str, prompt: str) -> Tuple[str, list]:
    query = tokenizer.from_list_format(
        [
            {"image": image},
            {"text": prompt},
        ]
    )
    response, history = model.chat(tokenizer, query=query, history=None)
    return response, history


def describe_image(tokenizer, model, image: str) -> str:
    prompt = "请详细描述这张图片的内容。"
    response, _ = run_single_turn(tokenizer, model, image, prompt)
    return response


def ask_image(tokenizer, model, image: str, question: str) -> str:
    response, _ = run_single_turn(tokenizer, model, image, question)
    return response


def read_text(tokenizer, model, image: str) -> str:
    prompt = "请逐字提取图片中的文字，并按行输出。"
    response, _ = run_single_turn(tokenizer, model, image, prompt)
    return response


def ground_object(tokenizer, model, image: str, target: str, output_path: str) -> str:
    prompt = f"请框出图中的{target}"
    response, history = run_single_turn(tokenizer, model, image, prompt)

    bbox_image = tokenizer.draw_bbox_on_latest_picture(response, history)
    if bbox_image is not None:
        bbox_image.save(output_path)
    else:
        print("[WARN] 模型未返回可解析的框，未生成画框图片。")
    return response


def build_args():
    parser = argparse.ArgumentParser(description="Quick test for Qwen-VL-Chat features.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-VL-Chat",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Image local path or URL.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="图中主要人物在做什么？",
        help="Question for visual QA task.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="主要人物",
        help="Target text used for grounding.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grounding_result.jpg",
        help="Output path for grounded image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional force device: cpu or cuda. Default: auto detect.",
    )
    return parser.parse_args()


def main():
    args = build_args()

    tokenizer, model, device = load_model(args.model, args.device)
    print(f"[INFO] Loaded model: {args.model}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Image: {args.image}")

    print("\n=== 1) 图片描述 ===")
    desc = describe_image(tokenizer, model, args.image)
    print(desc)

    print("\n=== 2) 图片问答 ===")
    qa = ask_image(tokenizer, model, args.image, args.question)
    print(f"Q: {args.question}")
    print(f"A: {qa}")

    print("\n=== 3) 读图文字 ===")
    ocr = read_text(tokenizer, model, args.image)
    print(ocr)

    print("\n=== 4) 框选目标 ===")
    grounding = ground_object(tokenizer, model, args.image, args.target, args.output)
    print(grounding)

    output_file = Path(args.output)
    if output_file.exists():
        print(f"[INFO] 已保存框选结果图: {output_file.resolve()}")


if __name__ == "__main__":
    main()
