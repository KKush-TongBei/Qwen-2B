from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.inference import (
    DEFAULT_IMAGE_PROMPT,
    InferenceConfig,
    InferenceConfigError,
    QwenVLDescriber,
    default_model_path,
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

app = FastAPI(title="Qwen3-VL-2B API", version="1.0.0")
describer: QwenVLDescriber | None = None


def _build_config_from_env() -> InferenceConfig:
    import os

    return InferenceConfig(
        model=os.getenv("QWEN_MODEL", default_model_path()),
        cpu=os.getenv("QWEN_CPU", "false").lower() == "true",
        max_gpus=int(os.getenv("QWEN_MAX_GPUS", "2")),
        cuda_devices=os.getenv("QWEN_CUDA_DEVICES"),
        per_gpu_memory=os.getenv("QWEN_PER_GPU_MEMORY", "31GiB"),
        flash_attn=os.getenv("QWEN_FLASH_ATTN", "false").lower() == "true",
    )


@app.on_event("startup")
def startup_event() -> None:
    global describer
    config = _build_config_from_env()
    describer = QwenVLDescriber(config)
    describer.load()


@app.get("/health")
def health() -> dict[str, str]:
    if describer is None or not describer.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_ready")
    return {"status": "ok"}


@app.post("/v1/describe")
async def describe(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    max_new_tokens: int = Form(256),
) -> dict[str, object]:
    if describer is None or not describer.is_loaded():
        raise HTTPException(status_code=503, detail="model_not_ready")
    effective_prompt = prompt.strip() or DEFAULT_IMAGE_PROMPT
    if max_new_tokens <= 0 or max_new_tokens > 4096:
        raise HTTPException(status_code=400, detail="max_new_tokens 必须在 1-4096 之间。")
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="不支持的图片类型。")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            text = describer.describe(
                image=tmp.name,
                prompt=effective_prompt,
                max_new_tokens=max_new_tokens,
            )
    except InferenceConfigError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {e}") from e

    return {
        "text": text,
        "model": describer.model_path,
        "usage": {
            "max_new_tokens": max_new_tokens,
            "prompt": effective_prompt,
        },
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
            }
        },
    )
