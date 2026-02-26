#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据已有特征(尤其是 syncnet 特征)重新生成 lip_sync_score_new。

策略优先级：
1. 若 features['syncnet'] 为字典，优先使用其中的 'confidence' 或 'sync_score'。
2. 若为二维/一维向量: 尝试第0维作为置信度；若还有 offset，可对 offset 做平滑惩罚。
3. 若完全缺失，则打 0.0 并标记 mask=False。
4. 归一化：对全部有效置信度做 min-max -> [0,1]，再映射到 [0,5]（可选）。
5. 输出：在原 pkl 基础上新增字段 lip_sync_score_new 与对应 valid_masks。

使用方式：
python scripts/generate_lip_sync_scores.py \
  --input datasets/ac_final_processed.pkl \
  --output datasets/ac_final_processed_lipsync.pkl \
  --scale_max 5.0
"""
import argparse
import pickle
import numpy as np
from pathlib import Path

POSSIBLE_KEYS = ['confidence', 'sync_score', 'score']

def extract_raw_score(sync_feat):
    if sync_feat is None:
        return None
    # 字典
    if isinstance(sync_feat, dict):
        for k in POSSIBLE_KEYS:
            if k in sync_feat and sync_feat[k] is not None:
                try:
                    return float(sync_feat[k])
                except Exception:
                    pass
        # offset 惩罚（可选）
        if 'offset' in sync_feat and sync_feat['offset'] is not None:
            try:
                off = abs(float(sync_feat['offset']))
                base = float(sync_feat.get('confidence', 0.0) or 0.0)
                return max(base - off * 0.05, 0.0)
            except Exception:
                return None
        return None
    # ndarray 或 list
    if isinstance(sync_feat, (list, tuple, np.ndarray)):
        arr = np.array(sync_feat).astype(float).ravel()
        if arr.size == 0:
            return None
        # 取第一个值；若第二个值像 offset，则做简单惩罚
        score = float(arr[0])
        if arr.size > 1:
            offset = abs(arr[1])
            score = max(score - offset * 0.05, 0.0)
        return score
    # 标量可转
    try:
        return float(sync_feat)
    except Exception:
        return None


def process_split(split_dict, scale_max=5.0):
    feats = split_dict['features']
    labels = split_dict['labels']
    masks = split_dict['valid_masks']
    # 旧结构：labels 是 dict[str] -> list/array
    # 我们新增列表
    raw_scores = []
    valid = []
    sync_feats = feats.get('syncnet', None)
    # 若 sync_feats 是按样本堆叠的列表/数组，与其它特征长度一致
    sample_count = len(labels['overall_score'])
    # sync_feats 可能是列表(每样本一个结构)或 ndarray；处理成列表
    if sync_feats is None or len(sync_feats) != sample_count:
        # 尝试：每个样本从 labels 里原 lip_sync_score 如果非常数可回退
        existing = labels.get('lip_sync_score', None)
        if existing is not None:
            arr = np.array(existing, dtype=float)
            if np.std(arr) > 1e-6:
                for v in arr:
                    raw_scores.append(float(v))
                    valid.append(True)
            else:
                raw_scores = [0.0]*sample_count
                valid = [False]*sample_count
        else:
            raw_scores = [0.0]*sample_count
            valid = [False]*sample_count
        return raw_scores, valid
    # 正常情况：逐样本解析
    for i in range(sample_count):
        sf = sync_feats[i]
        val = extract_raw_score(sf)
        if val is None or not np.isfinite(val):
            raw_scores.append(0.0)
            valid.append(False)
        else:
            raw_scores.append(val)
            valid.append(True)
    # 归一化（仅对 valid 样本）
    vals = np.array([raw_scores[i] for i in range(sample_count) if valid[i]], dtype=float)
    if vals.size > 1:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin > 1e-9:
            for i in range(sample_count):
                if valid[i]:
                    norm = (raw_scores[i] - vmin)/(vmax - vmin)
                    raw_scores[i] = norm * scale_max
        else:
            # 所有值相同 -> 全部标记为无效
            for i in range(sample_count):
                valid[i] = False
    else:
        # 不足以归一化
        for i in range(sample_count):
            valid[i] = False
    return raw_scores, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='原始数据 pkl')
    ap.add_argument('--output', required=True, help='输出 pkl')
    ap.add_argument('--scale_max', type=float, default=5.0, help='缩放上限 (默认 5.0)')
    args = ap.parse_args()
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    for split in ['train','val','test']:
        if split not in data: continue
        split_dict = data[split]
        lip_new, valid_new = process_split(split_dict, scale_max=args.scale_max)
        split_dict['labels']['lip_sync_score_new'] = lip_new
        split_dict['valid_masks']['lip_sync_score_new'] = valid_new
    with open(args.output,'wb') as f:
        pickle.dump(data,f)
    print(f'完成，新文件: {args.output}')

if __name__ == '__main__':
    main()
