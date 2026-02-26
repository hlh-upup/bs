#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仅重新提取 SyncNet 同步置信度并写回聚合数据PKL。
适用场景：现有 processed 聚合文件缺少 syncnet，但仍有 video_id -> 原始视频文件 可定位。
生成 outputs:
  - 更新后的 PKL: 加入 features['syncnet_confidence'] (list[float]) 与 features['syncnet_offset'] (list[float])
  - 可选: 直接构造 lip_sync_score_new （若加 --build-score）

用法示例：
python scripts/extract_syncnet_only.py \
  --input datasets/ac_final_processed.pkl \
  --output datasets/ac_final_processed_with_syncnet.pkl \
  --video-root datasets/ch-simsv2s/Raw \
  --syncnet-model models/pre-trained/stable_syncnet.pt \
  --batch-size 8 --device cuda --build-score

之后再运行 generate_lip_sync_scores.py（若未启用 --build-score）。
"""
import argparse
import pickle
import os
from tqdm import tqdm
import torch
import numpy as np

# 尝试复用已有 SyncNetInstance
try:
    from utils.syncnet.syncnet_instance import SyncNetInstance
except Exception:
    raise ImportError("无法导入 SyncNetInstance，请确认 utils/syncnet 存在且可用")

import warnings
warnings.filterwarnings('ignore')

VIDEO_EXTS = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']

def find_video(video_root, video_id):
    """根据 video_id (形如 video_0001_00027 或 video_0001_00003) 推测目录与文件。"""
    # 拆分: 假设格式 video_XXXX_YYYYY (前缀_编号_片段)
    parts = video_id.split('_')
    if len(parts) < 3:
        return None
    base = '_'.join(parts[:2])  # video_0001
    clip = parts[2]
    # clip 可能带前导0
    for ext in VIDEO_EXTS:
        candidate = os.path.join(video_root, base, clip + ext)
        if os.path.exists(candidate):
            return candidate
    # 再尝试去掉前导0
    clip_int = None
    try:
        clip_int = int(clip)
    except Exception:
        return None
    for width in [5,4,3,2,1]:
        clip_variant = f"{clip_int:0{width}d}"
        for ext in VIDEO_EXTS:
            candidate = os.path.join(video_root, base, clip_variant + ext)
            if os.path.exists(candidate):
                return candidate
    return None

@torch.no_grad()
def extract_syncnet_scores(video_paths, syncnet_inst, device):
    """批量提取 syncnet 分数。当前简单逐个处理（可扩展并行）。"""
    scores = []
    offsets = []
    for vp in video_paths:
        if vp is None:
            scores.append(None)
            offsets.append(None)
            continue
        try:
            # SyncNetInstance 可能需要自定义的 forward；这里假设 net 返回 (sync_score, offset)
            # 若实际 API 不同，需要按项目实际接口调整。
            # 占位：调用 syncnet_inst.net.forward_like(vp) -> 需要你根据实现修改。
            # === 占位开始 ===
            # 由于项目中未展示具体提取函数，这里给出一个伪逻辑：
            # 你需要用现有的特征提取器函数替换。
            # 示例：syncnet_inst.extract(video_path) -> { 'sync_score':..., 'offset':... }
            if hasattr(syncnet_inst, 'extract'):
                out = syncnet_inst.extract(vp)
            else:
                # 如果没有 extract 方法，给出占位报错提示
                raise RuntimeError('SyncNetInstance 缺少 extract 方法，请实现或替换调用逻辑')
            sync_score = out.get('sync_score') if isinstance(out, dict) else None
            offset = out.get('offset') if isinstance(out, dict) else None
            if sync_score is None:
                scores.append(None)
                offsets.append(offset if offset is not None else 0.0)
            else:
                scores.append(float(sync_score))
                offsets.append(float(offset) if offset is not None else 0.0)
        except Exception as e:
            scores.append(None)
            offsets.append(None)
    return scores, offsets

def build_lip_sync_score(scores, offsets, scale_max=5.0):
    raw = []
    valid = []
    for s, off in zip(scores, offsets):
        if s is None or not np.isfinite(s):
            raw.append(0.0); valid.append(False); continue
        penalty = 0.05 * abs(off) if (off is not None and np.isfinite(off)) else 0.0
        val = max(s - penalty, 0.0)
        raw.append(val); valid.append(True)
    arr = np.array([raw[i] for i,v in enumerate(valid) if v], float)
    if arr.size > 1:
        mn, mx = arr.min(), arr.max()
        if mx - mn > 1e-9:
            for i,v in enumerate(valid):
                if v:
                    raw[i] = (raw[i]-mn)/(mx-mn)*scale_max
        else:
            # 全部相同无效
            valid = [False]*len(valid)
    else:
        valid = [False]*len(valid)
    return raw, valid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--video-root', required=True, help='原始视频根目录(含 video_xxxx 子目录)')
    ap.add_argument('--syncnet-model', required=False, help='可选: 模型路径 (若实例内部不自动加载)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--build-score', action='store_true', help='直接构建 lip_sync_score_new')
    ap.add_argument('--scale-max', type=float, default=5.0)
    args = ap.parse_args()

    with open(args.input,'rb') as f:
        data = pickle.load(f)

    # 期望结构: {'train': {...}, 'val': {...}, 'test': {...}} 且每个 split 下有 'labels' (list) 长度
    # 当前 processed 结构: features 各模态 -> 列表/数组, labels 各任务 -> 列表
    # 我们没有逐样本 video_id 列 -> 需要找到保存的位置（如果你当前结构里没有 video_id 列表，需要扩展原处理流程）

    # 如果没有 video_id，只能放弃；这里先尝试 'video_id' 列
    for sp in ['train','val','test']:
        if sp not in data: continue
        split = data[sp]
        # 尝试获取 video_id 列
        video_ids = split.get('video_ids') or split.get('video_id')
        if video_ids is None:
            print(f"[WARN] split {sp} 缺少 video_ids，无法定位视频 -> 跳过")
            continue
        video_paths = [find_video(args.video_root, vid) for vid in video_ids]
        syncnet_inst = SyncNetInstance()  # 真实实现中可加加载权重
        scores, offsets = extract_syncnet_scores(video_paths, syncnet_inst, args.device)
        split.setdefault('features', {})
        split['features']['syncnet_confidence'] = scores
        split['features']['syncnet_offset'] = offsets
        if args.build_score:
            lip_values, lip_mask = build_lip_sync_score(scores, offsets, scale_max=args.scale_max)
            split['labels']['lip_sync_score_new'] = lip_values
            split['valid_masks']['lip_sync_score_new'] = lip_mask
        print(f"完成 split={sp}: 提取 {len(scores)} 条，valid={(np.isfinite([s for s in scores if s is not None])).sum()}")

    with open(args.output,'wb') as f:
        pickle.dump(data,f)
    print(f"写入完成: {args.output}")

if __name__ == '__main__':
    main()
