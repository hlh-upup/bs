"""
AnimeGANv2 ONNX 推理封装（CPU优先，自动缓存会话）。

依赖：onnxruntime, numpy, opencv-python

权重放置建议：
- 默认搜索路径：<repo_root>/weights/cartoon/
- 期望文件名：
  - animeganv2_hayao.onnx
  - animeganv2_shinkai.onnx
  - animeganv2_paprika.onnx
  - animeganv2_celeba.onnx

使用：
from util.anime_cartoonizer import cartoonize_animegan_onnx
out_bgr = cartoonize_animegan_onnx(img_bgr, style="hayao")
"""

from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception as e:  # 延迟报错：只有调用时才需要 onnxruntime
    ort = None  # type: ignore


_SESSION_CACHE = {}


STYLE_TO_FILENAME = {
    # === AnimeGAN v2 styles ===
    "hayao": "animeganv2_hayao.onnx",
    "shinkai": "animeganv2_shinkai.onnx",
    "paprika": "animeganv2_paprika.onnx",
    "celeba": "animeganv2_celeba.onnx",
    # === AnimeGAN v3 Paprika aliases ===
    "animeganv3_paprika": "animeganv3_paprika.onnx",
    "paprika_v3": "animeganv3_paprika.onnx",
    "paprika3": "animeganv3_paprika.onnx",
    # === AnimeGAN v3 Hayao/Shinkai (user-provided filenames) ===
    "animeganv3_hayao": "AnimeGANv3_Hayao_36.onnx",
    "hayao_v3": "AnimeGANv3_Hayao_36.onnx",
    "animeganv3_shinkai": "AnimeGANv3_Shinkai_37.onnx",
    "shinkai_v3": "AnimeGANv3_Shinkai_37.onnx",
}


def _default_weights_dir() -> str:
    # 相对本文件两级上层的 repo 根目录
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, "weights", "cartoon")


def resolve_weight_path(style: str = "hayao", weights_dir: Optional[str] = None) -> str:
    """Resolve ONNX weight path for given style or alias.

    Resolution order:
    1. Direct mapping via STYLE_TO_FILENAME (after alias normalization)
    2. Case-insensitive filename scan fallback if mapped file missing
    3. Fallback to v2 'hayao' if available
    4. Return the computed path (may not exist) for caller error handling
    """
    style_norm = (style or "hayao").lower().strip()
    if style_norm in ("paprika3",):  # unify alias
        style_norm = "animeganv3_paprika"

    fname = STYLE_TO_FILENAME.get(style_norm)
    if not fname:
        fname = STYLE_TO_FILENAME["hayao"]

    wdir = weights_dir or os.environ.get("CARTOON_WEIGHTS_DIR") or _default_weights_dir()
    path = os.path.join(wdir, fname)
    if os.path.exists(path):
        return path

    # case-insensitive scan fallback
    lower_target = fname.lower()
    try:
        for fn in os.listdir(wdir):
            if fn.lower() == lower_target:
                return os.path.join(wdir, fn)
    except Exception:
        pass

    # hayao v2 fallback if exists
    hayao_path = os.path.join(wdir, STYLE_TO_FILENAME["hayao"])
    if os.path.exists(hayao_path):
        return hayao_path

    return path  # may not exist; caller will raise clearer error


def is_session_cached(style: str = "hayao", weights_dir: Optional[str] = None) -> bool:
    path = os.path.abspath(resolve_weight_path(style, weights_dir))
    return path in _SESSION_CACHE


def get_session_providers(style: str = "hayao", weights_dir: Optional[str] = None) -> Optional[list]:
    path = os.path.abspath(resolve_weight_path(style, weights_dir))
    sess = _SESSION_CACHE.get(path)
    if sess is None:
        return None
    try:
        return sess.get_providers()  # type: ignore[attr-defined]
    except Exception:
        return None


def _get_ort_session(weight_path: str):
    if ort is None:
        raise RuntimeError("onnxruntime 未安装，请先在后端环境安装 onnxruntime")
    key = os.path.abspath(weight_path)
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"未找到 AnimeGANv2 权重: {weight_path}\n"
            f"请将对应 .onnx 文件放到 {os.path.dirname(weight_path)}，文件名参考: {list(STYLE_TO_FILENAME.values())}"
        )
    # CPUExecutionProvider 兼容性最好
    sess = ort.InferenceSession(weight_path, providers=["CPUExecutionProvider"])  # type: ignore
    _SESSION_CACHE[key] = sess
    return sess


def _to_multiple_of_32(h: int, w: int) -> Tuple[int, int]:
    nh = int(np.ceil(h / 32.0) * 32)
    nw = int(np.ceil(w / 32.0) * 32)
    return nh, nw


def _preprocess_bgr(img_bgr: np.ndarray, size_policy: str = "pad32") -> Tuple[np.ndarray, Tuple]:
    """
    返回: (tensor_chw, meta)
    meta: (orig_h, orig_w, proc_h, proc_w, unpad_rect)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    if size_policy == "pad32":
        proc_h, proc_w = _to_multiple_of_32(orig_h, orig_w)
        resized = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)
        unpad_rect = (0, 0, orig_w, orig_h)
    else:
        resized = img_bgr.copy()
        proc_h, proc_w = orig_h, orig_w
        unpad_rect = (0, 0, orig_w, orig_h)

    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb = img_rgb / 127.5 - 1.0  # [-1,1]
    chw = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)
    return chw.astype(np.float32), (orig_h, orig_w, proc_h, proc_w, unpad_rect)


def _postprocess_to_bgr(out_chw: np.ndarray, meta: Tuple) -> np.ndarray:
    (orig_h, orig_w, proc_h, proc_w, _unpad) = meta
    # out: (1,3,H,W)
    out = out_chw[0]
    out = np.transpose(out, (1, 2, 0))  # (H,W,3)
    out = (out + 1.0) * 127.5  # [-1,1] -> [0,255]
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if (proc_h, proc_w) != (orig_h, orig_w):
        out_bgr = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return out_bgr


def cartoonize_animegan_onnx(
    img_bgr: np.ndarray,
    style: str = "hayao",
    weights_dir: Optional[str] = None,
    max_side: int = 1440,
) -> np.ndarray:
    """
    使用 AnimeGANv2 ONNX 进行卡通化。
    - img_bgr: OpenCV BGR 图像
    - style: 'hayao'|'shinkai'|'paprika'|'celeba'
    - max_side: 可选，先将最长边缩放到不超过该值以加速
    返回 BGR 图像。
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("输入空图像")

    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    weight_path = resolve_weight_path(style=style, weights_dir=weights_dir)
    sess = _get_ort_session(weight_path)

    inp_chw, meta = _preprocess_bgr(img_bgr, size_policy="pad32")
    # 根据模型输入/输出维度自动判断布局（NCHW vs NHWC）
    try:
        inp_info = sess.get_inputs()[0]
        out_info = sess.get_outputs()[0]
        def _to_int_list(shape):
            vals = []
            for d in shape:
                try:
                    vals.append(int(d))
                except Exception:
                    vals.append(-1)
            return vals
        in_shape = _to_int_list(getattr(inp_info, 'shape', []))
        out_shape = _to_int_list(getattr(out_info, 'shape', []))
        in_layout = 'NCHW'
        if len(in_shape) == 4:
            if in_shape[3] == 3:
                in_layout = 'NHWC'
            elif in_shape[1] == 3:
                in_layout = 'NCHW'
            else:
                # 无法确定时，优先按 NHWC（AnimeGANv2 常见导出）
                in_layout = 'NHWC'
        out_layout = 'NCHW'
        if len(out_shape) == 4:
            if out_shape[3] == 3:
                out_layout = 'NHWC'
            elif out_shape[1] == 3:
                out_layout = 'NCHW'
            else:
                out_layout = in_layout
    except Exception:
        in_layout = 'NHWC'
        out_layout = 'NHWC'

    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    if in_layout == 'NHWC':
        # 将 (1,3,H,W) 转为 (1,H,W,3)
        inp = np.transpose(inp_chw, (0, 2, 3, 1)).astype(np.float32)
    else:
        inp = inp_chw

    out = sess.run([out_name], {inp_name: inp})[0]

    # 统一为 CHW 以复用后处理
    if out_layout == 'NHWC':
        # (1,H,W,3) -> (1,3,H,W)
        out_chw = np.transpose(out, (0, 3, 1, 2)).astype(np.float32)
    else:
        out_chw = out

    out_bgr = _postprocess_to_bgr(out_chw, meta)

    # 若前面缩小过图像，放回原尺寸
    if scale != 1.0:
        out_bgr = cv2.resize(out_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    return out_bgr
