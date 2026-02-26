#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习曲线脚本

- 基于现有 Trainer/模型/数据集，按训练集子样本比例进行训练，收集验证与测试指标。
- 示例运行：
  python tools/learning_curve.py --config_path config/config.yaml --dataset_path datasets/ac.pkl --fractions 0.1,0.25,0.5,1.0 --epochs 10 --seed 42 --use_cuda
"""

import os
import sys
import argparse
import pickle
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

# 加入项目根目录到 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils import setup_logging, load_config, save_config, set_seed, get_device, format_time, create_experiment_dir  # noqa: E402
from data.dataset import TalkingFaceDataset  # noqa: E402
from models import TalkingFaceEvaluationModel  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="学习曲线 - 按训练集比例子样本训练")
    parser.add_argument('--config_path', type=str, required=True, help='配置文件路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='合并数据集pkl路径，键含 train/val/test')
    parser.add_argument('--fractions', type=str, default='0.1,0.25,0.5,1.0', help='训练集比例，逗号分隔')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数，不填则用配置文件')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖batch size，不填则用配置文件')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='是否使用CUDA')
    return parser.parse_args()


def build_dataloaders(sub_train, val, test, config):
    # 数据集
    train_dataset = TalkingFaceDataset(sub_train, config)
    val_dataset = TalkingFaceDataset(val, config)
    test_dataset = TalkingFaceDataset(test, config)

    data_cfg = config['data']
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg.get('num_workers', 0)
    pin_memory = data_cfg.get('pin_memory', True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def _safe_get(d, *keys, default=np.nan):
    cur = d
    try:
        for k in keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        # 如果最终值不是标量，返回 NaN
        if isinstance(cur, (int, float, np.floating)):
            return float(cur)
        return float(cur) if cur is not None else float('nan')
    except Exception:
        return float('nan')


def _write_csv(rows, header, out_path):
    try:
        import csv
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return True
    except Exception:
        return False


def _plot_curves(x_vals, y_map, out_dir, title_prefix='learning_curve_overall'):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return  # 无法绘图时静默跳过

    os.makedirs(out_dir, exist_ok=True)

    for metric, y_vals in y_map.items():
        # 检查是否全为 NaN
        arr = np.asarray(y_vals, dtype=float)
        if np.all(np.isnan(arr)):
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel('Train fraction')
        plt.ylabel(metric.upper())
        plt.title(f"{title_prefix}_{metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(out_dir, f"{title_prefix}_{metric}.png")
        try:
            plt.savefig(png_path)
        except Exception:
            pass
        plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    # 加载配置
    config = load_config(args.config_path)

    # 设备
    device = get_device(args.use_cuda)

    # 加载合并数据（train/val/test 列表）
    with open(args.dataset_path, 'rb') as f:
        all_data = pickle.load(f)
    if not isinstance(all_data, dict) or not all(k in all_data for k in ['train', 'val', 'test']):
        raise TypeError('数据集pkl应为包含 train/val/test 键的dict')

    full_train = list(all_data['train'])
    val_data = list(all_data['val'])
    test_data = list(all_data['test'])

    # 实验目录
    exp_dir = create_experiment_dir(os.path.join(ROOT, 'experiments', 'learning_curve'))
    logger = setup_logging(os.path.join(exp_dir, 'logs'))

    logger.info('学习曲线实验启动')
    logger.info(f"config: {args.config_path}")
    logger.info(f"dataset: {args.dataset_path}")
    logger.info(f"train/val/test sizes: {len(full_train)}/{len(val_data)}/{len(test_data)}")

    # 覆盖训练超参（如提供）
    base_cfg = deepcopy(config)
    if args.epochs is not None:
        base_cfg['train']['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        base_cfg['data']['batch_size'] = int(args.batch_size)
    base_cfg['train']['output_dir'] = exp_dir

    # 解析比例
    fractions = []
    for s in args.fractions.split(','):
        try:
            v = float(s.strip())
            if 0 < v <= 1.0:
                fractions.append(v)
        except Exception:
            continue
    fractions = sorted(set(fractions))
    logger.info(f"fractions: {fractions}")

    # 结果收集
    results_summary = []

    for frac in fractions:
        # 子样本采样
        n = max(1, int(len(full_train) * frac))
        rng = random.Random(args.seed)
        sub_train = rng.sample(full_train, n) if n < len(full_train) else full_train

        # 为当前比例创建子目录
        run_dir = os.path.join(exp_dir, f"frac_{frac:.2f}")
        os.makedirs(run_dir, exist_ok=True)

        cfg = deepcopy(base_cfg)
        cfg['train']['output_dir'] = run_dir

        # 构造dataloader
        train_loader, val_loader, test_loader = build_dataloaders(sub_train, val_data, test_data, cfg)

        # 模型
        model = TalkingFaceEvaluationModel(cfg['model'])
        model.to(device)

        # 训练
        trainer = Trainer(model, cfg, train_loader, val_loader, test_loader, device=device)
        final_results = trainer.train()

        # 汇总
        entry = {
            'fraction': frac,
            'train_size': len(sub_train),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'best_epoch': final_results.get('best_epoch', None),
            'best_val_loss': float(final_results.get('best_val_loss', float('nan'))),
            'test_loss': float(final_results.get('test_loss', float('nan'))),
            'test_metrics': final_results.get('test_metrics', {})
        }
        results_summary.append(entry)

        # 保存当前比例结果
        save_config(entry, os.path.join(run_dir, 'summary.yaml'))
        logger.info(f"完成 fraction={frac:.2f}，best_epoch={entry['best_epoch']}, best_val_loss={entry['best_val_loss']:.4f}")

    # 保存总汇总
    import json
    with open(os.path.join(exp_dir, 'learning_curve_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    # 生成 CSV 与作图（基于 overall 任务的 rmse/mae/r2）
    csv_rows = []
    x_fracs = []
    y_map = {'rmse': [], 'mae': [], 'r2': []}

    for e in results_summary:
        frac = float(e.get('fraction', np.nan))
        x_fracs.append(frac)
        tm = e.get('test_metrics', {}) or {}
        overall = tm.get('overall', {}) if isinstance(tm, dict) else {}
        rmse = float(overall.get('rmse', np.nan)) if isinstance(overall, dict) else np.nan
        mae = float(overall.get('mae', np.nan)) if isinstance(overall, dict) else np.nan
        r2 = float(overall.get('r2', np.nan)) if isinstance(overall, dict) else np.nan

        y_map['rmse'].append(rmse)
        y_map['mae'].append(mae)
        y_map['r2'].append(r2)

        csv_rows.append({
            'fraction': frac,
            'train_size': e.get('train_size'),
            'val_size': e.get('val_size'),
            'test_size': e.get('test_size'),
            'best_epoch': e.get('best_epoch'),
            'best_val_loss': e.get('best_val_loss'),
            'test_loss': e.get('test_loss'),
            'overall_rmse': rmse,
            'overall_mae': mae,
            'overall_r2': r2,
        })

    # 写 CSV
    csv_path = os.path.join(exp_dir, 'learning_curve_summary.csv')
    _write_csv(csv_rows, header=['fraction','train_size','val_size','test_size','best_epoch','best_val_loss','test_loss','overall_rmse','overall_mae','overall_r2'], out_path=csv_path)

    # 画图
    _plot_curves(x_fracs, y_map, out_dir=exp_dir, title_prefix='learning_curve_overall')

    logger.info('学习曲线实验完成')


if __name__ == '__main__':
    main()
