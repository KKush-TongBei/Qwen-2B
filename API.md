# Qwen3-VL-2B API 使用说明

本项目已提供可调用的 FastAPI 服务，支持通过文件上传进行图片描述。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

PyTorch 请按你的 CUDA 版本从官方站点安装。

## 2. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

可选环境变量（启动前设置）：

- `QWEN_MODEL`：模型目录或 Hugging Face 模型 ID，默认 `./Qwen3-VL-2B`
- `QWEN_CPU`：`true/false`，是否强制 CPU，默认 `false`
- `QWEN_MAX_GPUS`：最多使用前 N 张物理 GPU，默认 `2`
- `QWEN_CUDA_DEVICES`：直接指定 `CUDA_VISIBLE_DEVICES`，如 `0,1`
- `QWEN_PER_GPU_MEMORY`：`device_map=auto` 时每卡上限，默认 `31GiB`
- `QWEN_FLASH_ATTN`：`true/false`，是否开启 `flash_attention_2`

## 3. 接口

### 健康检查

```bash
curl http://127.0.0.1:8000/health
```

成功示例：

```json
{"status":"ok"}
```

### 图片描述

- 方法：`POST /v1/describe`
- Content-Type：`multipart/form-data`
- 表单字段：
  - `image`：图片文件（必填）
  - `prompt`：文本提示词（可选）
  - `max_new_tokens`：最大生成长度（可选，1-4096）

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/v1/describe" \
  -F "image=@/absolute/path/to/demo.jpg" \
  -F "prompt=请详细描述这张图片。" \
  -F "max_new_tokens=256"
```

成功响应示例：

```json
{
  "text": "这是一张……",
  "model": "/path/to/Qwen3-VL-2B",
  "usage": {
    "max_new_tokens": 256
  }
}
```

失败响应示例：

```json
{
  "error": {
    "code": 400,
    "message": "不支持的图片类型。"
  }
}
```

## 4. 常见问题

- 模型未就绪（503）：检查模型路径、权重文件是否完整。
- CUDA 不可用：检查 PyTorch 是否为 CUDA 版本，或设置 `QWEN_CPU=true`。
- 显存不足：降低 `QWEN_MAX_GPUS`、减小 `max_new_tokens`、或调整 `QWEN_PER_GPU_MEMORY`。
