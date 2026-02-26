#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量在数据消融变体上进行快速评估（无需重新训练）。
给定：
- 训练时使用的 config.yaml
- 已训练好的 best_model.pth（或任意 checkpoint）
- 变体目录（例如 datasets/ablation）

将遍历目录中的 .pkl 数据集，对每个变体创建 DataLoader、加载模型与权重，
运行 Evaluator 并把各任务的核心指标（MSE/MAE/RMSE/R2/Pearson/QWK）汇总到 CSV。

用法：
    python tools/eval_on_variants.py \
        --config f:/bs/config/config.yaml \
        --checkpoint f:/bs/experiments/run1/checkpoints/best_model.pth \
        --variants-dir f:/bs/datasets/ablation \
        --output f:/bs/reports/ablation_eval_summary.csv
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, Any
import pandas as pd
import torch

# 项目内导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, get_device, setup_logging
from models import create_model
from evaluation import Evaluator
from torch.utils.data import DataLoader
from data.dataset import TalkingFaceDataset


def _build_test_loader_from_pkl(dataset_pkl_path: str, config: Dict[str, Any]) -> DataLoader | None:
    """仅为测试集构建 DataLoader；若测试集为空则返回 None。"""
    import pickle, os
    if not os.path.exists(dataset_pkl_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_pkl_path}")
    with open(dataset_pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    if not isinstance(all_data, dict) or 'test' not in all_data:
        raise TypeError("数据集格式错误：需要包含 'test' 键的字典")
    test_data = all_data['test']
    if not isinstance(test_data, list) or len(test_data) == 0:
        return None

    test_dataset = TalkingFaceDataset(test_data, config)
    dl_cfg = config.get('data', {})
    batch_size = int(dl_cfg.get('batch_size', 8))
    num_workers = int(dl_cfg.get('num_workers', 0))
    pin_memory = bool(dl_cfg.get('pin_memory', True))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return test_loader


def evaluate_single_variant(config_path: str, checkpoint_path: str, dataset_pkl: str, device: torch.device) -> Dict[str, Any]:
    # 加载配置
    config = load_config(config_path)
    # 设置输出目录（临时，不写入磁盘过多文件）
    base_out = os.path.join(os.path.dirname(checkpoint_path), '..', 'ablation_eval')
    os.makedirs(base_out, exist_ok=True)
    config['train'] = config.get('train', {})
    config['train']['output_dir'] = base_out

    # 仅创建测试 DataLoader（变体可能没有 train/val，避免 0 样本导致 DataLoader 抛错）
    test_loader = _build_test_loader_from_pkl(dataset_pkl, config)
    if test_loader is None:
        raise ValueError("测试集为空，跳过该变体")

    # 创建模型并加载权重
    model = create_model(config['model'])
    model.to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    # 放宽权重加载，忽略旧版本中存在而当前模型不存在的参数（如自适应投影层）
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    # 评估（在 test split 上）
    evaluator = Evaluator(model=model, config=config, test_loader=test_loader, device=device, output_dir=base_out)
    metrics = evaluator.evaluate()
    return metrics


def main():
    parser = argparse.ArgumentParser(description='在各数据变体(.pkl)上批量评估已训练模型。')
    parser.add_argument('--config', required=True, type=str, help='训练时使用的 config.yaml 路径')
    parser.add_argument('--checkpoint', required=True, type=str, help='模型权重 .pth 路径（如 best_model.pth）')
    parser.add_argument('--variants-dir', required=True, type=str, help='包含多个 .pkl 变体的目录')
    parser.add_argument('--output', required=True, type=str, help='汇总 CSV 输出路径')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='是否使用 CUDA')
    args = parser.parse_args()

    logger = setup_logging(os.path.join(os.path.dirname(args.output), 'logs'))
    device = get_device(args.use_cuda)
    logger.info(f"使用设备: {device}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 遍历目录下的 .pkl
    variants_dir = args.variants_dir
    pkl_files = [os.path.join(variants_dir, f) for f in os.listdir(variants_dir) if f.endswith('.pkl')]
    if not pkl_files:
        logger.error(f"目录下未找到 .pkl: {variants_dir}")
        sys.exit(1)

    rows = []
    for pkl_path in pkl_files:
        name = os.path.basename(pkl_path)
        logger.info(f"评估变体: {name}")
        try:
            metrics = evaluate_single_variant(args.config, args.checkpoint, pkl_path, device)
            # 扁平化关键信息
            row = {'variant': name}
            for task, m in metrics.items():
                if isinstance(m, dict):
                    for k in ['mse', 'mae', 'rmse', 'r2', 'pearson', 'qwk']:
                        if k in m:
                            row[f'{task}_{k}'] = m[k]
            rows.append(row)
        except Exception as e:
            logger.exception(f"评估 {name} 失败: {e}")
            rows.append({'variant': name, 'error': str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    logger.info(f"汇总结果已保存: {args.output}")

    # 额外生成 Markdown/TXT 总结，便于快速阅读与分享
    try:
        def _write_text_and_md(df: pd.DataFrame, out_csv_path: str):
            base_dir = os.path.dirname(out_csv_path)
            base_name = os.path.splitext(os.path.basename(out_csv_path))[0]
            md_path = os.path.join(base_dir, base_name + '.md')
            txt_path = os.path.join(base_dir, base_name + '.txt')

            lines = []
            lines.append(f"# Ablation Evaluation Summary\n")
            lines.append(f"CSV: {out_csv_path}\n")

            # 拆分错误与有效指标
            has_error_col = 'error' in df.columns
            if has_error_col:
                err_df = df[df['error'].notna()].copy()
                met_df = df[df['error'].isna()].copy()
            else:
                err_df = pd.DataFrame(columns=['variant','error'])
                met_df = df.copy()

            # 选一个排名列（优先 average_r2 / overall_r2 / average_pearson / overall_pearson）
            rank_candidates = ['average_r2', 'overall_r2', 'average_pearson', 'overall_pearson']
            rank_col = None
            for c in rank_candidates:
                if c in met_df.columns:
                    rank_col = c
                    break
            # 如果都没有，退而求其次用 'overall_qwk' 或反向的 'average_mse'
            if rank_col is None:
                for c in ['overall_qwk', 'average_qwk', 'overall_spearman', 'average_spearman']:
                    if c in met_df.columns:
                        rank_col = c
                        break
            if rank_col is None:
                for c in ['average_mse', 'overall_mse', 'average_rmse', 'overall_rmse', 'average_mae', 'overall_mae']:
                    if c in met_df.columns:
                        rank_col = c
                        break

            # 排名
            top_section = []
            if not met_df.empty and rank_col is not None:
                # 转为数值，非数置 NaN
                met_df[rank_col] = pd.to_numeric(met_df[rank_col], errors='coerce')
                higher_better = any(key in rank_col for key in ['r2', 'pearson', 'spearman', 'kendall', 'ccc', 'accuracy', 'qwk'])
                met_df_sorted = met_df.sort_values(rank_col, ascending=not higher_better, na_position='last')
                # 摘要前 5 项
                top_k = met_df_sorted.head(5).copy()
                lines.append(f"\n## Top variants by {rank_col} ({'higher' if higher_better else 'lower'} is better)\n\n")
                for i, row in enumerate(top_k.itertuples(index=False), start=1):
                    val = getattr(row, rank_col)
                    lines.append(f"{i}. {getattr(row, 'variant')} — {rank_col}: {val}\n")
                top_section = [getattr(r, 'variant') for r in top_k.itertuples(index=False)]

            # 错误列表
            if not err_df.empty:
                lines.append("\n## Skipped / Errors\n\n")
                for r in err_df.itertuples(index=False):
                    lines.append(f"- {getattr(r, 'variant')}: {getattr(r, 'error')}\n")

            # 简要表格（只展示关心的几列）
            show_cols = ['variant']
            for c in ['average_r2', 'overall_r2', 'average_pearson', 'overall_pearson', 'average_qwk', 'overall_qwk', 'average_rmse', 'overall_rmse', 'average_mae', 'overall_mae']:
                if c in met_df.columns:
                    show_cols.append(c)
            if len(show_cols) > 1 and not met_df.empty:
                lines.append("\n## Summary table (key metrics)\n\n")
                # 生成 Markdown 表格
                sub = met_df[show_cols].copy()
                # 保证数值列格式
                for c in show_cols[1:]:
                    sub[c] = pd.to_numeric(sub[c], errors='coerce')
                lines.append(sub.to_markdown(index=False))
                lines.append("\n")

            md_text = ''.join(lines)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_text)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(md_text)

            return md_path, txt_path

        md_path, txt_path = _write_text_and_md(df, args.output)
        logger.info(f"已生成 Markdown/TXT 总结: {md_path} | {txt_path}")
    except Exception as e:
        logger.warning(f"生成 Markdown/TXT 总结失败: {e}")


if __name__ == '__main__':
    main()
