import os
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import cv2
import tensorflow as tf  # type: ignore

try:
    from huggingface_hub import snapshot_download  # type: ignore
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

__all__ = [
    "wbc_cartoonize",
    "is_wbc_model_loaded",
]

_MODEL_LOCK = threading.Lock()
_WBC_MODEL = None  # type: ignore
_WBC_SIGNATURE = None  # type: ignore
_WBC_SNAPSHOT_PATH: Optional[str] = None

HF_REPO_ID = "sayakpaul/whitebox-cartoonizer"
ENV_WBC_LOCAL_DIR = "WBC_MODEL_DIR"  # 指定一个本地 saved_model 路径时将跳过下载
ENV_WBC_OFFLINE = "WBC_OFFLINE"       # =1 时强制离线（不触发下载，只搜本地）
ENV_WBC_REPO_ID = "WBC_REPO_ID"       # 可覆盖默认的 huggingface repo id

_LOCAL_SEARCH_CANDIDATES = [
    "weights/wbc_saved_model",
    "weights/whitebox",
    "models/wbc",
    "assets/wbc",
]

def _is_saved_model_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "saved_model.pb"))
        and os.path.isdir(os.path.join(path, "variables"))
    )

def _find_local_snapshot() -> Optional[str]:
    # 优先使用环境变量 WBC_MODEL_DIR，其次候选目录（相对工程根）
    override = os.environ.get(ENV_WBC_LOCAL_DIR, "").strip()
    if override and _is_saved_model_dir(override):
        return override
    for cand in _LOCAL_SEARCH_CANDIDATES:
        if _is_saved_model_dir(cand):
            return cand
    return None


def _download_and_load() -> None:
    global _WBC_MODEL, _WBC_SIGNATURE, _WBC_SNAPSHOT_PATH
    if _WBC_MODEL is not None:
        return
    with _MODEL_LOCK:
        if _WBC_MODEL is not None:
            return
        # 1) 本地优先（环境变量或常见目录）
        local_snapshot = _find_local_snapshot()
        offline = os.environ.get(ENV_WBC_OFFLINE, "0") in ("1", "true", "True")
        repo_id = os.environ.get(ENV_WBC_REPO_ID, HF_REPO_ID)

        if local_snapshot:
            snapshot_path = local_snapshot
        else:
            if offline:
                raise RuntimeError(
                    "WBC 离线模式启用但未找到本地 saved_model。请设置 WBC_MODEL_DIR 或放置到 weights/wbc_saved_model/"
                )
            if not _HF_AVAILABLE:
                raise RuntimeError("缺少 huggingface_hub，无法下载。请 pip install huggingface_hub 或提供本地模型")
            # 2) 远程下载（缓存）
            snapshot_path = snapshot_download(repo_id)
        model = tf.saved_model.load(snapshot_path)
        signature = model.signatures.get("serving_default")
        if signature is None:
            raise RuntimeError("WBC 模型缺少 serving_default 签名")
        _WBC_MODEL = model
        _WBC_SIGNATURE = signature
        _WBC_SNAPSHOT_PATH = snapshot_path


def is_wbc_model_loaded() -> bool:
    return _WBC_MODEL is not None


def _resize_crop(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def _preprocess(bgr: np.ndarray) -> tf.Tensor:
    # BGR -> keep BGR (original reference uses BGR before converting back)
    img = _resize_crop(bgr)
    img = img.astype(np.float32) / 127.5 - 1.0  # [-1,1]
    img = np.expand_dims(img, 0)
    return tf.constant(img)


def _postprocess(tensor: "tf.Tensor") -> np.ndarray:
    # Expect range [-1,1]
    out = (tensor[0].numpy() + 1.0) * 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    # Model outputs BGR already per reference implementation; if colors look odd, swap channels.
    return out


def wbc_cartoonize(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Run White-box Cartoonization (TensorFlow SavedModel via HF) on a single BGR image.

    Returns dict with keys: output (np.ndarray), debug (dict)
    """
    t0 = time.time()
    _download_and_load()
    load_ms = (time.time() - t0) * 1000.0

    t1 = time.time()
    inp = _preprocess(img_bgr)
    prep_ms = (time.time() - t1) * 1000.0

    t2 = time.time()
    result = _WBC_SIGNATURE(inp)["final_output:0"]  # type: ignore
    infer_ms = (time.time() - t2) * 1000.0

    t3 = time.time()
    out_img = _postprocess(result)
    post_ms = (time.time() - t3) * 1000.0

    total_ms = (time.time() - t0) * 1000.0

    return {
        "output": out_img,
        "debug": {
            "snapshot_path": _WBC_SNAPSHOT_PATH,
            "offline": os.environ.get(ENV_WBC_OFFLINE, "0"),
            "repo_id": os.environ.get(ENV_WBC_REPO_ID, HF_REPO_ID),
            "load_model_ms": load_ms,
            "preprocess_ms": prep_ms,
            "inference_ms": infer_ms,
            "postprocess_ms": post_ms,
            "total_ms": total_ms,
            "input_shape": list(img_bgr.shape),
            "output_shape": list(out_img.shape),
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="White-box Cartoonization local tester")
    parser.add_argument("--input", "-i", required=True, help="Input image path (BGR readable by OpenCV)")
    parser.add_argument("--output", "-o", default="wbc_result.png", help="Output image path (PNG)")
    parser.add_argument("--wbc_model_dir", default=None, help="Local SavedModel directory for WBC (sets WBC_MODEL_DIR)")
    parser.add_argument("--offline", action="store_true", help="Enable offline mode (sets WBC_OFFLINE=1)")
    parser.add_argument("--no_resize_back", action="store_true", help="Do not resize output back to original size")

    args = parser.parse_args()

    if args.wbc_model_dir:
        os.environ[ENV_WBC_LOCAL_DIR] = args.wbc_model_dir
    if args.offline:
        os.environ[ENV_WBC_OFFLINE] = "1"

    src = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"无法读取输入图片: {args.input}")

    orig_h, orig_w = src.shape[:2]
    result = wbc_cartoonize(src)
    out_img = result["output"]

    # 可选：等比拉回原尺寸，便于对比
    if not args.no_resize_back and (out_img.shape[0] != orig_h or out_img.shape[1] != orig_w):
        out_img = cv2.resize(out_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    ok = cv2.imwrite(args.output, out_img)
    if not ok:
        raise SystemExit(f"保存输出失败: {args.output}")

    print("[WBC] Done.")
    print("[WBC] Debug:")
    print(json.dumps(result.get("debug", {}), ensure_ascii=False, indent=2))
