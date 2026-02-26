#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single-video quality evaluation service entry for Chapter 3 model.

Usage:
  python quality_eval_service.py --video <video_path> --checkpoint <best_model.pth> --config <config.yaml>
"""

import argparse
import json
import os
import sys
import traceback
from typing import Any, Dict

import numpy as np
import torch

# Ensure bs root is importable
BS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BS_ROOT not in sys.path:
    sys.path.insert(0, BS_ROOT)

from utils import load_config  # noqa: E402
from models import create_model  # noqa: E402
from features.extractor import FeatureExtractor  # noqa: E402


def _as_numpy_feature(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, dict):
        if "deep" in value and isinstance(value["deep"], np.ndarray):
            arr = value["deep"]
        elif "acoustic" in value and isinstance(value["acoustic"], np.ndarray):
            arr = value["acoustic"]
        else:
            arr = np.array(list(value.values()), dtype=np.float32)
    else:
        arr = np.array(value, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.float32, copy=False)


def evaluate_single_video(video_path: str, checkpoint_path: str, config_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature extraction
    extractor = FeatureExtractor(config, device)
    feats = extractor.extract_all_features(video_path)
    required_keys = ["visual", "audio", "keypoint", "au"]
    missing = [k for k in required_keys if k not in feats]
    if missing:
        raise RuntimeError(f"Missing required features: {missing}")

    visual = _as_numpy_feature(feats["visual"])
    audio = _as_numpy_feature(feats["audio"])
    keypoint = _as_numpy_feature(feats["keypoint"])
    au = _as_numpy_feature(feats["au"])

    # Model init and load
    model = create_model(config["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(
            visual_features=torch.from_numpy(visual).to(device),
            audio_features=torch.from_numpy(audio).to(device),
            keypoint_features=torch.from_numpy(keypoint).to(device),
            au_features=torch.from_numpy(au).to(device),
        )

    scores = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            scores[k] = float(v.squeeze().detach().cpu().item())
    if "overall" not in scores and scores:
        core = [scores.get("lip_sync"), scores.get("expression"), scores.get("audio_quality"), scores.get("cross_modal")]
        core = [x for x in core if x is not None]
        if core:
            scores["overall"] = float(sum(core) / len(core))

    return {
        "result": "Success",
        "video": video_path,
        "checkpoint": checkpoint_path,
        "config": config_path,
        "scores": scores,
        "device": str(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a generated video by Chapter-3 model")
    parser.add_argument("--video", required=True, help="Video path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Model config path")
    args = parser.parse_args()

    try:
        result = evaluate_single_video(args.video, args.checkpoint, args.config)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:
        err = {
            "result": "Failed",
            "error": str(e),
            "traceback": traceback.format_exc(limit=3),
        }
        print(json.dumps(err, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

