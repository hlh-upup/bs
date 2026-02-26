#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
K 折交叉验证脚本

- 在 train+val 上做 K 折交叉验证（保持 test 集固定，仅用于最终评估）。
- 复用现有 Trainer/模型/数据集。
- 示例运行：
  python tools/kfold_cv.py --config_path config/config.yaml --dataset_path datasets/ac.pkl --k 5 --epochs 10 --seed 42 --use_cuda
"""

import os
import sys
import argparse
import pickle
from copy import deepcopy
import random
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# 加入项目根目录到 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils import setup_logging, load_config, save_config, set_seed, get_device, create_experiment_dir  # noqa: E402
from data.dataset import TalkingFaceDataset  # noqa: E402
from models import TalkingFaceEvaluationModel  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="K 折交叉验证（train+val 上划分）")
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='合并数据集pkl路径，键含 train/val/test')
    parser.add_argument('--k', type=int, default=5, help='折数')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖batch size')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='是否使用CUDA')
    return parser.parse_args()


def build_loader(data_list, config, shuffle):
    dataset = TalkingFaceDataset(data_list, config)
    cfg = config['data']
    return DataLoader(dataset,
                      batch_size=cfg['batch_size'],
                      shuffle=shuffle,
                      num_workers=cfg.get('num_workers', 0),
                      pin_memory=cfg.get('pin_memory', True))


def main():
    args = parse_args()
    set_seed(args.seed)

    # 加载配置
    config = load_config(args.config_path)

    # 覆盖基本超参
    base_cfg = deepcopy(config)
    if args.epochs is not None:
        base_cfg['train']['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        base_cfg['data']['batch_size'] = int(args.batch_size)

    # 设备
    device = get_device(args.use_cuda)

    # 加载数据（dict: train/val/test）
    with open(args.dataset_path, 'rb') as f:
        all_data = pickle.load(f)
    if not isinstance(all_data, dict) or not all(k in all_data for k in ['train', 'val', 'test']):
        raise TypeError('数据集pkl应为包含 train/val/test 键的dict')

    train_val = list(all_data['train']) + list(all_data['val'])
    test_data = list(all_data['test'])

    # 实验目录
    exp_dir = create_experiment_dir(os.path.join(ROOT, 'experiments', 'kfold_cv'))
    logger = setup_logging(os.path.join(exp_dir, 'logs'))
    logger.info('K 折交叉验证实验启动')
    logger.info(f"k={args.k}, seed={args.seed}")
    logger.info(f"train+val size={len(train_val)}, test size={len(test_data)}")

    # 交叉验证划分
    indices = list(range(len(train_val)))
    kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(indices), start=1):
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        cfg = deepcopy(base_cfg)
        cfg['train']['output_dir'] = fold_dir

        # 构造当前折的数据
        train_subset = [train_val[i] for i in tr_idx]
        val_subset = [train_val[i] for i in va_idx]

        # DataLoaders
        train_loader = build_loader(train_subset, cfg, shuffle=True)
        val_loader = build_loader(val_subset, cfg, shuffle=False)
        test_loader = build_loader(test_data, cfg, shuffle=False)

        # 模型与训练
        model = TalkingFaceEvaluationModel(cfg['model']).to(device)
        trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, device=device)
        final_results = trainer.train()

        # 记录
        summary = {
            'fold': fold,
            'train_size': len(train_subset),
            'val_size': len(val_subset),
            'test_size': len(test_data),
            'best_epoch': final_results.get('best_epoch', None),
            'best_val_loss': float(final_results.get('best_val_loss', float('nan'))),
            'test_loss': float(final_results.get('test_loss', float('nan'))),
            'test_metrics': final_results.get('test_metrics', {})
        }
        fold_summaries.append(summary)

        save_config(summary, os.path.join(fold_dir, 'summary.yaml'))
        logger.info(f"Fold {fold}: best_epoch={summary['best_epoch']} best_val_loss={summary['best_val_loss']:.4f}")

    # 计算平均±方差（针对 test_metrics 中的每个指标与任务）
    # 假定 test_metrics 结构如: {task: {metric: value}}
    metrics_agg = {}
    for s in fold_summaries:
        tm = s.get('test_metrics', {})
        for task, mdict in tm.items():
            metrics_agg.setdefault(task, {})
            for metric, val in mdict.items():
                metrics_agg[task].setdefault(metric, [])
                try:
                    metrics_agg[task][metric].append(float(val))
                except Exception:
                    metrics_agg[task][metric].append(np.nan)

    metrics_stats = {}
    for task, mdict in metrics_agg.items():
        metrics_stats[task] = {}
        for metric, vals in mdict.items():
            arr = np.asarray(vals, dtype=float)
            metrics_stats[task][metric] = {
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'values': [float(x) if not np.isnan(x) else None for x in arr.tolist()]
            }

    # 保存整体汇总
    out = {
        'k': args.k,
        'fold_summaries': fold_summaries,
        'metrics_stats': metrics_stats
    }
    with open(os.path.join(exp_dir, 'kfold_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info('K 折交叉验证完成')


if __name__ == '__main__':
    main()
