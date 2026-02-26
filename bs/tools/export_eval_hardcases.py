#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从评估输出的 test_predictions.csv 中导出各任务 Top-K 误差样本，便于人工查看与做针对性数据增强/清洗。

输入：
- --pred-csv: test_predictions.csv 文件路径（通常位于 experiments/<exp>/results/test_predictions.csv 或 ablation_eval/ 下）
- --topk: 每个任务导出 Top-K（默认 50）
- --out: 导出的 CSV 路径（默认同目录 hardcases_topk.csv）
- --tasks: 指定任务（默认五个任务：lip_sync, expression, audio_quality, cross_modal, overall）

用法：
    python tools/export_eval_hardcases.py ^
      --pred-csv experiments\run1\results\test_predictions.csv ^
      --topk 50 ^
      --out reports\hardcases_top50.csv
"""

import os
import argparse
import pandas as pd

DEFAULT_TASKS = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']

def main():
    parser = argparse.ArgumentParser(description='导出各任务 Top-K 误差样本。')
    parser.add_argument('--pred-csv', required=True, type=str, help='test_predictions.csv 路径')
    parser.add_argument('--topk', type=int, default=50, help='每任务导出 Top-K')
    parser.add_argument('--out', type=str, default=None, help='输出 CSV 路径（默认与输入同目录）')
    parser.add_argument('--tasks', type=str, nargs='*', default=DEFAULT_TASKS, help='任务列表')
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    out_dir = os.path.dirname(args.pred_csv) if args.out is None else os.path.dirname(args.out)
    os.makedirs(out_dir or '.', exist_ok=True)
    out_path = args.out if args.out is not None else os.path.join(out_dir, f'hardcases_top{args.topk}.csv')

    rows = []
    for task in args.tasks:
        pred_col = f'{task}_pred'
        true_col = f'{task}_true'
        if pred_col not in df.columns or true_col not in df.columns:
            continue
        tmp = df.copy()
        tmp['abs_err'] = (tmp[pred_col] - tmp[true_col]).abs()
        tmp.sort_values('abs_err', ascending=False, inplace=True)
        top = tmp.head(args.topk).copy()
        top.insert(0, 'task', task)
        top.insert(1, 'rank', range(1, len(top) + 1))
        rows.append(top)
    if not rows:
        print('未找到任何任务列，检查输入 CSV 是否包含 *_pred 和 *_true 列。')
        return

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已导出：{out_path}')


if __name__ == '__main__':
    main()
