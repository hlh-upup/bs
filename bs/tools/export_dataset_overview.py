#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从数据集 PKL 导出数据总览：
- 每条样本：video_id、split、标签有效性（每任务是否有效）、各标签值、估计长度、运动强度（基于时序差分方差）
- 全局：按 split 的统计聚合（可由用户用 Excel 透视）

用法：
    python tools/export_dataset_overview.py --input datasets\ac.pkl --out reports\dataset_overview.csv
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd

TASKS = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']


def _safe_float(x, default=-1.0):
    try:
        return float(x)
    except Exception:
        return default


def _label_valid(sample: dict, task: str) -> bool:
    v = _safe_float(sample.get('labels', {}).get(f'{task}_score', -1.0), -1.0)
    return v >= 0.0


def _sequence_length(sample: dict):
    feats = sample.get('features', {})
    for k in ['visual', 'audio', 'keypoint', 'au']:
        if k in feats:
            arr = np.asarray(feats[k])
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return None


def _motion_score(sample: dict) -> float:
    feats = sample.get('features', {})
    for k in ['keypoint', 'visual', 'au', 'audio']:
        if k in feats:
            arr = np.asarray(feats[k])
            if arr.ndim >= 2 and arr.shape[0] >= 2:
                diff = np.diff(arr, axis=0)
                return float(np.mean(np.var(diff, axis=0)))
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='导出数据集总览 CSV')
    parser.add_argument('--input', required=True, type=str, help='数据集 .pkl 路径（list 或 dict(train/val/test)）')
    parser.add_argument('--out', required=True, type=str, help='输出 CSV 路径')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    # 统一为 tvts
    if isinstance(data, list):
        tvts = {'train': [], 'val': [], 'test': []}
        for s in data:
            sp = s.get('split', 'train')
            if sp not in tvts:
                sp = 'train'
            tvts[sp].append(s)
    elif isinstance(data, dict):
        tvts = data
    else:
        raise TypeError('不支持的数据格式')

    rows = []
    for sp in ['train', 'val', 'test']:
        for s in tvts.get(sp, []):
            row = {
                'split': sp,
                'video_id': s.get('video_id', None)
            }
            # 标签
            for t in TASKS:
                v = _safe_float(s.get('labels', {}).get(f'{t}_score', -1.0), -1.0)
                row[f'{t}_score'] = v
                row[f'{t}_valid'] = float(v >= 0.0)
            # 长度与运动
            L = _sequence_length(s)
            row['seq_len'] = L if L is not None else -1
            row['motion_score'] = _motion_score(s)
            rows.append(row)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'已导出：{args.out} (共 {len(df)} 行)')


if __name__ == '__main__':
    main()
