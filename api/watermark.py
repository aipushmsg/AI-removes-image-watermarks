"""水印掩膜生成：OCR 文字定位 + 关键词过滤 + logo 模板匹配。"""
from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    return _OCR


def _kw_match(text: str, keywords: Sequence[str]) -> Optional[Dict[str, Any]]:
    """关键词支持普通子串（忽略大小写），以 /.../i 形式书写时按正则处理。

    命中时返回 {"kw": 命中的关键词, "left": 左侧缺失字符数, "right": 右侧缺失字符数}，
    用于把被 OCR 截断的文字框（如 "mple-shop.com" / "opyright"）向两侧补齐。
    """
    tl = text.lower()
    for kw in keywords:
        if not kw:
            continue
        if len(kw) > 2 and kw.startswith("/") and kw.endswith("/"):
            if re.search(kw[1:-1], text, re.IGNORECASE):
                return {"kw": kw, "left": 0, "right": 0}
            continue
        kl = kw.lower()
        if kl in tl:
            return {"kw": kw, "left": 0, "right": 0}
        # OCR 只识别出关键词的一部分：要求片段足够长，避免 "PER"/"OL" 之类的误命中
        if len(tl) >= 4 and tl in kl:
            idx = kl.find(tl)
            return {
                "kw": kw,
                "left": idx,
                "right": len(kl) - idx - len(tl),
            }
        # 兜底：OCR 有少量错字（如 "mple-shop.com"/"opyright"），按最长公共子串判断
        if len(tl) >= 4:
            m = difflib.SequenceMatcher(None, tl, kl).find_longest_match(
                0, len(tl), 0, len(kl)
            )
            if m.size >= max(5, int(0.5 * len(tl))):
                return {
                    "kw": kw,
                    "left": m.b,
                    "right": len(kl) - (m.b + m.size),
                }
    return None


def _text_hits(
    image_bgr: np.ndarray,
    keywords: Sequence[str],
    min_score: float,
    match_all: bool,
    upscale_min_side: int = 900,
    upscale_max_factor: int = 4,
) -> List[Dict[str, Any]]:
    """OCR 定位文字。长边过小时先放大再识别，提高小图/缩略图的召回率。"""
    h, w = image_bgr.shape[:2]
    factor = 1
    if max(h, w) < upscale_min_side:
        factor = int(np.ceil(upscale_min_side / max(h, w)))
        factor = max(1, min(factor, upscale_max_factor))

    src = image_bgr
    if factor > 1:
        src = cv2.resize(
            image_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
        )
        # 小图放大后先锐化，OCR 对小字/细笔画的召回率提升明显
        src = cv2.filter2D(
            src,
            -1,
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        )

    result, _ = _get_ocr()(src)
    hits: List[Dict[str, Any]] = []
    for item in result or []:
        box, text, score = item[0], item[1], item[2]
        if score is None or float(score) < min_score:
            continue
        match = {"kw": "", "left": 0, "right": 0}
        if not match_all:
            found = _kw_match(text, keywords)
            if not found:
                continue
            match = found
        pts = (np.asarray(box, dtype=np.float64) / factor).astype(np.int32)
        # 文字被 OCR 截断时，按缺失字符比例向左右外扩（单侧最多 1.5 倍框宽）
        if match["left"] or match["right"]:
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            width = max(1, x_max - x_min)
            per_char = width / max(1, len(text))
            cap = width * 1.0
            if match["left"]:
                ext = int(min(cap, per_char * match["left"]))
                pts[:, 0] = np.where(pts[:, 0] == x_min, x_min - ext, pts[:, 0])
            if match["right"]:
                ext = int(min(cap, per_char * match["right"]))
                pts[:, 0] = np.where(pts[:, 0] == x_max, x_max + ext, pts[:, 0])
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        hits.append({"type": "text", "text": text, "score": float(score), "box": pts})
    return hits


def _logo_hits(
    image_bgr: np.ndarray,
    template_bgr: np.ndarray,
    threshold: float,
    scales: Sequence[float],
) -> List[Dict[str, Any]]:
    ih, iw = image_bgr.shape[:2]
    th, tw = template_bgr.shape[:2]
    hits: List[Dict[str, Any]] = []
    for scale in scales:
        w, h = int(tw * scale), int(th * scale)
        if w < 8 or h < 8 or w > iw or h > ih:
            continue
        tpl = cv2.resize(template_bgr, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(image_bgr, tpl, cv2.TM_CCOEFF_NORMED)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        ys, xs = np.where(res >= threshold)
        for y, x in zip(ys, xs):
            hits.append(
                {
                    "type": "logo",
                    "text": "",
                    "score": float(res[y, x]),
                    "box": np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
                    ),
                }
            )
    # 简单去重：IoU > 0.5 的重叠框只保留分数最高的
    hits.sort(key=lambda h: h["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    for hit in hits:
        if all(_iou(hit["box"], k["box"]) < 0.5 for k in kept):
            kept.append(hit)
    return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
    bx1, by1, bx2, by2 = b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_mask(
    image_bgr: np.ndarray,
    keywords: Optional[Sequence[str]] = None,
    template_bgr: Optional[np.ndarray] = None,
    dilate: int = 8,
    ocr_min_score: float = 0.6,
    template_threshold: float = 0.8,
    match_all_text: bool = False,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """返回 (uint8 掩膜, 命中列表)，掩膜中 255 表示需要修复的区域。"""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    hits: List[Dict[str, Any]] = []

    if template_bgr is not None:
        hits.extend(_logo_hits(image_bgr, template_bgr, template_threshold, (0.4, 0.6, 0.8, 1.0, 1.2, 1.5)))

    if match_all_text or keywords:
        hits.extend(_text_hits(image_bgr, keywords or [], ocr_min_score, match_all_text))

    for hit in hits:
        cv2.fillPoly(mask, [np.asarray(hit["box"], dtype=np.int32)], 255)

    if hits and dilate > 0:
        k = dilate * 2 + 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    for hit in hits:
        hit.pop("box", None)
    return mask, hits
