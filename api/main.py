"""电商商品图去水印 HTTP 服务：自动生成掩膜 -> 调用 IOPaint 的 LaMa 修复。"""
from __future__ import annotations

import base64
import io
import os
from typing import Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from watermark import build_mask

IOPAINT_URL = os.getenv("IOPAINT_URL", "http://127.0.0.1:8080").rstrip("/")
INPAINT_TIMEOUT = int(os.getenv("INPAINT_TIMEOUT", "300"))

app = FastAPI(title="E-commerce Watermark Removal API", version="1.0.0")


def _to_bgr(raw: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """统一成 BGR，返回 (bgr, alpha 或 None)。"""
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    if raw.shape[2] == 4:
        return raw[:, :, :3], raw[:, :, 3]
    return raw, None


async def _read_upload(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, f"{file.filename} 内容为空")
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise HTTPException(400, f"{file.filename} 不是有效图片")
    return raw


def _parse_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.replace("，", ",").split(",") if k.strip()]


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "图片编码失败")
    return buf.tobytes()


def _match_tone(original_bgr: np.ndarray, result_bgr: np.ndarray, mask: np.ndarray, max_shift: float = 25.0) -> np.ndarray:
    """把修复区域的整体色调校正回周围环境，避免纯色背景上出现偏白的矩形块。"""
    ring = cv2.subtract(cv2.dilate(mask, np.ones((9, 9), np.uint8)), mask)
    sel = ring > 0
    if sel.sum() < 50:
        return result_bgr
    diff = np.median(original_bgr[sel].astype(np.float32), axis=0) - np.median(
        result_bgr[sel].astype(np.float32), axis=0
    )
    diff = np.clip(diff, -max_shift, max_shift)
    out = result_bgr.astype(np.float32)
    out[mask > 0] += diff
    return np.clip(out, 0, 255).astype(np.uint8)


def _flat_fill(original_bgr: np.ndarray, result_bgr: np.ndarray, mask: np.ndarray, max_std: float = 6.0) -> np.ndarray:
    """掩膜周边是纯色（白底/灰底）时，直接用 TELEA 填充，避免 LaMa 留下亮度差。"""
    n, labels = cv2.connectedComponents(mask)
    out = result_bgr.copy()
    for i in range(1, n):
        comp = ((labels == i).astype(np.uint8)) * 255
        ring = cv2.subtract(cv2.dilate(comp, np.ones((9, 9), np.uint8)), comp)
        sel = ring > 0
        if sel.sum() < 30:
            continue
        if float(np.mean(np.std(original_bgr[sel].astype(np.float32), axis=0))) <= max_std:
            filled = cv2.inpaint(original_bgr, comp, 5, cv2.INPAINT_TELEA)
            out[comp > 0] = filled[comp > 0]
    return out


def _call_iopaint(image_png: bytes, mask_png: bytes, hd_strategy: str) -> bytes:
    payload = {
        "image": base64.b64encode(image_png).decode(),
        "mask": base64.b64encode(mask_png).decode(),
        "hd_strategy": hd_strategy,
    }
    try:
        r = requests.post(f"{IOPAINT_URL}/api/v1/inpaint", json=payload, timeout=INPAINT_TIMEOUT)
    except requests.RequestException as e:
        raise HTTPException(502, f"无法连接 IOPaint 服务：{e}")
    if r.status_code != 200:
        raise HTTPException(502, f"IOPaint 返回 {r.status_code}: {r.text[:300]}")

    if r.headers.get("content-type", "").startswith("image/"):
        return r.content

    data = r.json()
    img_b64 = data.get("image") if isinstance(data, dict) else None
    if not img_b64:
        raise HTTPException(502, f"IOPaint 响应格式异常：{str(data)[:300]}")
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    return base64.b64decode(img_b64)


def _mask_and_hits(
    raw: np.ndarray,
    keywords: str,
    template_raw: Optional[np.ndarray],
    dilate: int,
    min_score: float,
    threshold: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, list, Optional[np.ndarray]]:
    bgr, alpha = _to_bgr(raw)
    if max(bgr.shape[:2]) > 2400:
        raise HTTPException(413, "图片长边超过 2400px，CPU 修复会过慢，请先压缩")
    tpl = None
    if template_raw is not None:
        tpl, _ = _to_bgr(template_raw)
    mask, hits = build_mask(
        bgr,
        keywords=_parse_keywords(keywords),
        template_bgr=tpl,
        dilate=dilate,
        ocr_min_score=min_score,
        template_threshold=threshold,
        match_all_text=(mode == "all"),
    )
    return bgr, mask, hits, alpha


@app.get("/health")
def health():
    try:
        r = requests.get(f"{IOPAINT_URL}/api/v1/model", timeout=5)
        model = r.json().get("name") if r.status_code == 200 else None
        return {"status": "ok" if r.status_code == 200 else "degraded", "iopaint": r.status_code, "model": model}
    except requests.RequestException as e:
        return JSONResponse({"status": "down", "error": str(e)}, status_code=503)


@app.post("/v1/mask")
async def mask_preview(
    image: UploadFile = File(...),
    keywords: str = Form(""),
    template: Optional[UploadFile] = File(None),
    dilate: int = Form(5),
    min_score: float = Form(0.6),
    threshold: float = Form(0.8),
    mode: str = Form("keyword"),
):
    """只返回掩膜 PNG（白色区域会被修复），用于人工校验检测是否准确。"""
    raw = await _read_upload(image)
    tpl_raw = await _read_upload(template) if template is not None else None
    _, mask, hits, _ = _mask_and_hits(raw, keywords, tpl_raw, dilate, min_score, threshold, mode)
    coverage = float(mask.mean()) / 255.0
    return Response(
        content=_encode_png(mask),
        media_type="image/png",
        headers={"X-Hits": str(len(hits)), "X-Coverage": f"{coverage:.4f}"},
    )


@app.post("/v1/remove-watermark")
async def remove_watermark(
    image: UploadFile = File(...),
    keywords: str = Form(""),
    template: Optional[UploadFile] = File(None),
    dilate: int = Form(5),
    min_score: float = Form(0.6),
    threshold: float = Form(0.8),
    mode: str = Form("keyword"),
    hd_strategy: str = Form("Original"),
):
    """检测并去除水印，返回 PNG。未命中时原样返回原图。"""
    raw = await _read_upload(image)
    tpl_raw = await _read_upload(template) if template is not None else None
    bgr, mask, hits, alpha = _mask_and_hits(raw, keywords, tpl_raw, dilate, min_score, threshold, mode)

    if not hits:
        out = raw if alpha is not None else bgr
        return Response(
            content=_encode_png(out),
            media_type="image/png",
            headers={"X-Hits": "0", "X-Coverage": "0"},
        )

    result_png = _call_iopaint(_encode_png(bgr), _encode_png(mask), hd_strategy)
    result = cv2.imdecode(np.frombuffer(result_png, np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise HTTPException(502, "IOPaint 返回的图片无法解码")
    result = _match_tone(bgr, result, mask)
    result = _flat_fill(bgr, result, mask)
    if alpha is not None:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = alpha

    return Response(
        content=_encode_png(result),
        media_type="image/png",
        headers={"X-Hits": str(len(hits)), "X-Coverage": f"{float(mask.mean()) / 255.0:.4f}"},
    )
