#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据现有数据集（.pkl）自动生成多种数据变体，便于做数据消融实验：
- cleaned_baseline: 清洗后的基础版（去NaN/Inf、去全零特征、裁剪异常标签、去空样本）
- per_task_valid_only_<task>: 仅保留该任务标签有效的样本（其余任务标签可为 -1）
- balanced_overall_q5: 按 overall_score 5分位均衡下采样
- balanced_lip_sync_q5: 按 lip_sync_score 5分位均衡下采样
- high_motion_top20: 依据关键点/视觉时序变化强度，保留前20%“运动大”的样本
- length_filtered_min100: 保留时序长度 >= 100 的样本（如果可判定）

使用：
    python tools/create_ablation_variants.py --input datasets/ac.pkl --out-dir datasets/ablation --bins 5 --topk 0.2

脚本会在输出目录写入：
- 各变体的 .pkl 文件（结构与原始一致：{'train': list, 'val': list, 'test': list}）
- 一份 CSV 质量报告（各变体/各split样本数、任务有效率、分布统计）
"""

import os
import sys
import math
import json
import pickle
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- 基础工具 ----------

TASKS = [
    'lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall'
]
LABEL_KEYS = [f"{t}_score" for t in TASKS]
SCORE_MIN, SCORE_MAX = 1.0, 5.0


def _is_dataset_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in ['train', 'val', 'test'])


def _to_tvts(data: Any) -> Dict[str, List[dict]]:
    """将任意输入格式标准化为 {'train': [], 'val': [], 'test': []}
    支持两种情况：
      - 已是 dict(train/val/test)
      - 是 list，且每个样本中含有 split 字段
    """
    if _is_dataset_dict(data):
        return data
    if isinstance(data, list):
        out = {'train': [], 'val': [], 'test': []}
        for s in data:
            split = s.get('split', 'train')
            if split not in out:
                split = 'train'
            out[split].append(s)
        return out
    raise TypeError(f"不支持的数据格式: {type(data)}")


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _label_valid(sample: dict, task: str) -> bool:
    v = sample.get('labels', {}).get(f'{task}_score', -1.0)
    try:
        v = float(v)
    except Exception:
        v = -1.0
    return v >= 0.0


def _clip_labels(sample: dict, smin=SCORE_MIN, smax=SCORE_MAX) -> dict:
    if 'labels' not in sample:
        return sample
    new = dict(sample)
    new_labels = dict(sample['labels'])
    for k in LABEL_KEYS:
        if k in new_labels and new_labels[k] != -1.0:
            try:
                v = float(new_labels[k])
            except Exception:
                v = -1.0
            if v != -1.0:
                new_labels[k] = float(np.clip(v, smin, smax))
    new['labels'] = new_labels
    return new


def _features_ok(feat: np.ndarray, zero_ratio_th=0.98) -> bool:
    if feat is None:
        return False
    arr = np.asarray(feat)
    if not np.isfinite(arr).all():
        # 存在 NaN/Inf
        return False
    # 过多的 0 视为无效特征
    if arr.size > 0:
        zratio = float(np.mean(arr == 0))
        if zratio >= zero_ratio_th:
            return False
    return True


def _clean_sample(sample: dict) -> dict | None:
    """清洗单样本：
    - 去除 NaN/Inf
    - 丢弃全零或近乎全零特征
    - 裁剪异常标签到 [1,5]
    - 若所有标签均为 -1 则丢弃
    返回清洗后的样本或 None
    """
    s = dict(sample)
    feats = s.get('features', {})
    if not isinstance(feats, dict):
        return None
    new_feats = {}
    has_any_valid_feat = False
    for key in ['visual', 'audio', 'keypoint', 'au', 'syncnet']:
        if key in feats:
            arr = np.asarray(feats[key])
            # 替换 NaN/Inf 为 0，便于后续检测零比例
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            if _features_ok(arr):
                new_feats[key] = arr
                has_any_valid_feat = True
    if not has_any_valid_feat:
        return None
    s['features'] = new_feats

    # 标签裁剪
    s = _clip_labels(s)

    # 是否全为 -1
    labels = s.get('labels', {})
    if not isinstance(labels, dict) or all(_safe_float(labels.get(k, -1.0)) == -1.0 for k in LABEL_KEYS):
        return None

    return s


def _sequence_length(sample: dict) -> int | None:
    feats = sample.get('features', {})
    # 任选存在的时序特征估计长度
    for k in ['visual', 'audio', 'keypoint', 'au']:
        if k in feats:
            arr = np.asarray(feats[k])
            if arr.ndim >= 2:
                return int(arr.shape[0])
    return None


def _motion_score(sample: dict) -> float:
    """用关键点或视觉特征的时间差分方差，粗估运动强度。"""
    feats = sample.get('features', {})
    for k in ['keypoint', 'visual', 'au', 'audio']:
        if k in feats:
            arr = np.asarray(feats[k])
            if arr.ndim >= 2 and arr.shape[0] >= 2:
                diff = np.diff(arr, axis=0)
                return float(np.mean(np.var(diff, axis=0)))
    return 0.0


def _bin_by_quantiles(values: np.ndarray, q: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """将一维数据按分位数切成 q 桶，返回每个样本所属桶id和桶区间。"""
    assert q >= 2
    qs = np.linspace(0, 1, q + 1)
    edges = np.quantile(values, qs)
    # 去重处理，避免重复边界导致空桶
    edges = np.unique(edges)
    # 如果边界过少，退化到等距分箱
    if len(edges) < 3:
        edges = np.linspace(values.min(), values.max(), q + 1)
    # 分配桶
    bin_ids = np.digitize(values, edges[1:-1], right=False)
    bins = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    return bin_ids, bins


# ---------- 生成各类变体 ----------

def make_cleaned_baseline(tvts: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    out = {}
    for split, items in tvts.items():
        cleaned = []
        for s in items:
            ns = _clean_sample(s)
            if ns is not None:
                cleaned.append(ns)
        out[split] = cleaned
    return out


def make_per_task_valid(tvts: Dict[str, List[dict]], task: str) -> Dict[str, List[dict]]:
    out = {}
    for split, items in tvts.items():
        buf = []
        for s in items:
            ns = _clean_sample(s)
            if ns is None:
                continue
            if _label_valid(ns, task):
                buf.append(ns)
        out[split] = buf
    return out


def make_balanced_by_label(tvts: Dict[str, List[dict]], task: str, qbins: int = 5, min_per_bin: int | None = None) -> Dict[str, List[dict]]:
    out = {}
    for split, items in tvts.items():
        # 仅考虑该任务有有效标签的样本
        valid = []
        vals = []
        for s in items:
            ns = _clean_sample(s)
            if ns is None:
                continue
            if _label_valid(ns, task):
                v = _safe_float(ns['labels'].get(f'{task}_score', -1.0), -1.0)
                if v != -1.0:
                    valid.append(ns)
                    vals.append(v)
        if len(valid) == 0:
            out[split] = []
            continue
        vals = np.asarray(vals, dtype=float)
        bin_ids, _ = _bin_by_quantiles(vals, qbins)
        # 统计每桶数量
        counts = np.bincount(bin_ids, minlength=qbins)
        target = int(counts.min()) if min_per_bin is None else min(min_per_bin, int(counts.min()))
        # 每桶随机下采样到 target
        chosen_idx = []
        rng = np.random.default_rng(42)
        for b in range(qbins):
            idx_b = np.where(bin_ids == b)[0]
            if len(idx_b) == 0:
                continue
            take = min(target, len(idx_b))
            chosen = rng.choice(idx_b, size=take, replace=False)
            chosen_idx.extend(chosen.tolist())
        chosen_idx = sorted(chosen_idx)
        out[split] = [valid[i] for i in chosen_idx]
    return out


def make_high_motion(tvts: Dict[str, List[dict]], topk: float = 0.2) -> Dict[str, List[dict]]:
    out = {}
    topk = float(topk)
    topk = min(max(topk, 0.01), 0.9)
    for split, items in tvts.items():
        scored = []
        for s in items:
            ns = _clean_sample(s)
            if ns is None:
                continue
            m = _motion_score(ns)
            scored.append((m, ns))
        if not scored:
            out[split] = []
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(len(scored) * topk))
        out[split] = [x[1] for x in scored[:k]]
    return out


def make_length_filtered(tvts: Dict[str, List[dict]], min_len: int = 100) -> Dict[str, List[dict]]:
    out = {}
    for split, items in tvts.items():
        buf = []
        for s in items:
            ns = _clean_sample(s)
            if ns is None:
                continue
            L = _sequence_length(ns)
            if L is None or L >= int(min_len):
                buf.append(ns)
        out[split] = buf
    return out


# ---------- 报告与保存 ----------

def _tvts_stats(tvts: Dict[str, List[dict]]) -> Dict[str, Any]:
    stats = {}
    for split, items in tvts.items():
        row = {'count': len(items)}
        # 有效率
        for t in TASKS:
            cnt_valid = 0
            values = []
            for s in items:
                if _label_valid(s, t):
                    cnt_valid += 1
                    values.append(_safe_float(s['labels'].get(f'{t}_score', -1.0), -1.0))
            row[f'{t}_valid'] = cnt_valid
            row[f'{t}_valid_ratio'] = (cnt_valid / len(items)) if len(items) > 0 else 0.0
            if values:
                row[f'{t}_mean'] = float(np.mean(values))
                row[f'{t}_std'] = float(np.std(values))
            else:
                row[f'{t}_mean'] = 0.0
                row[f'{t}_std'] = 0.0
        stats[split] = row
    return stats


def save_variant(variant: Dict[str, List[dict]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(variant, f)


def append_report(report_rows: List[dict], name: str, tvts: Dict[str, List[dict]]):
    stats = _tvts_stats(tvts)
    row = {'variant': name}
    for split in ['train', 'val', 'test']:
        s = stats.get(split, {})
        for k, v in s.items():
            row[f'{split}_{k}'] = v
    report_rows.append(row)


# ---------- 主流程 ----------

def find_default_dataset(datasets_dir: str) -> str | None:
    candidates = [
        'ac_final_processed.pkl',
        'ac.pkl',
        'ac_final_processed_lipsync.pkl',
        'ac_processed.pkl',
        'ch_sims_processed_data_cache_1985.pkl'
    ]
    for name in candidates:
        p = os.path.join(datasets_dir, name)
        if os.path.exists(p):
            return p
    # 兜底：挑选 .pkl 中最大的一个
    pkl_files = [os.path.join(datasets_dir, f) for f in os.listdir(datasets_dir) if f.endswith('.pkl')]
    if not pkl_files:
        return None
    pkl_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return pkl_files[0]


def main():
    parser = argparse.ArgumentParser(description='生成数据消融变体。')
    parser.add_argument('--input', type=str, default=None, help='输入数据集 .pkl 路径（缺省时自动在 datasets/ 选择一个）')
    parser.add_argument('--out-dir', type=str, default='datasets/ablation', help='输出目录')
    parser.add_argument('--bins', type=int, default=5, help='均衡分桶个数（默认5分位）')
    parser.add_argument('--topk', type=float, default=0.2, help='high_motion 保留比例（默认前20%）')
    parser.add_argument('--min-len', type=int, default=100, help='长度过滤的最小长度（默认100）')
    args = parser.parse_args()

    # 解析输入
    in_path = args.input
    if in_path is None:
        ds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
        in_path = find_default_dataset(ds_dir)
        if in_path is None:
            print('未找到可用数据集，请使用 --input 指定 .pkl 文件')
            sys.exit(1)
    if not os.path.exists(in_path):
        print(f'输入文件不存在: {in_path}')
        sys.exit(1)

    print(f'加载数据集: {in_path}')
    with open(in_path, 'rb') as f:
        raw = pickle.load(f)
    tvts = _to_tvts(raw)
    print('\n原始数据集大小:')
    for sp in ['train', 'val', 'test']:
        print(f'  {sp}: {len(tvts.get(sp, []))}')

    os.makedirs(args.out_dir, exist_ok=True)

    report_rows: List[dict] = []

    # 1) cleaned_baseline
    cleaned = make_cleaned_baseline(tvts)
    out1 = os.path.join(args.out_dir, 'cleaned_baseline.pkl')
    save_variant(cleaned, out1)
    append_report(report_rows, 'cleaned_baseline', cleaned)
    print(f'[OK] 保存 {out1}')

    # 2) per_task_valid_only_<task>
    for t in TASKS:
        per_t = make_per_task_valid(tvts, t)
        out_t = os.path.join(args.out_dir, f'per_task_valid_only_{t}.pkl')
        save_variant(per_t, out_t)
        append_report(report_rows, f'per_task_valid_only_{t}', per_t)
        print(f'[OK] 保存 {out_t}')

    # 3) balanced_overall_q5 / balanced_lip_sync_q5
    bal_overall = make_balanced_by_label(tvts, task='overall', qbins=int(args.bins))
    out_bo = os.path.join(args.out_dir, f'balanced_overall_q{int(args.bins)}.pkl')
    save_variant(bal_overall, out_bo)
    append_report(report_rows, f'balanced_overall_q{int(args.bins)}', bal_overall)
    print(f'[OK] 保存 {out_bo}')

    bal_lipsync = make_balanced_by_label(tvts, task='lip_sync', qbins=int(args.bins))
    out_bl = os.path.join(args.out_dir, f'balanced_lip_sync_q{int(args.bins)}.pkl')
    save_variant(bal_lipsync, out_bl)
    append_report(report_rows, f'balanced_lip_sync_q{int(args.bins)}', bal_lipsync)
    print(f'[OK] 保存 {out_bl}')

    # 4) high_motion_topk
    high_motion = make_high_motion(tvts, topk=float(args.topk))
    out_hm = os.path.join(args.out_dir, f'high_motion_top{int(args.topk*100)}.pkl')
    save_variant(high_motion, out_hm)
    append_report(report_rows, f'high_motion_top{int(args.topk*100)}', high_motion)
    print(f'[OK] 保存 {out_hm}')

    # 5) length_filtered_minL
    length_filtered = make_length_filtered(tvts, min_len=int(args.min_len))
    out_lf = os.path.join(args.out_dir, f'length_filtered_min{int(args.min_len)}.pkl')
    save_variant(length_filtered, out_lf)
    append_report(report_rows, f'length_filtered_min{int(args.min_len)}', length_filtered)
    print(f'[OK] 保存 {out_lf}')

    # 输出报告
    rep_path = os.path.join(args.out_dir, 'ablation_variants_report.csv')
    pd.DataFrame(report_rows).to_csv(rep_path, index=False)
    print(f'\n报告已保存: {rep_path}')

    # 额外保存一个 JSON 汇总，便于程序化读取
    json_path = os.path.join(args.out_dir, 'ablation_variants_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_rows, f, ensure_ascii=False, indent=2)
    print(f'JSON 汇总: {json_path}')

    print('\n完成：已生成以下变体：')
    for p in [out1, out_bo, out_bl, out_hm, out_lf] + [
        os.path.join(args.out_dir, f'per_task_valid_only_{t}.pkl') for t in TASKS
    ]:
        print(' -', p)


if __name__ == '__main__':
    main()
