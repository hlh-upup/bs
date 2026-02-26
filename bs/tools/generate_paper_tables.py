#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 Evaluator 生成的 metrics.json / metrics_ci.json 转换为论文可直接引用的表格：
- paper_metrics_table.tex  (LaTeX)
- paper_metrics_table.md   (Markdown)
- figure_captions.md      (图注建议)

用法（Windows CMD）：
  python tools/generate_paper_tables.py --eval_dir experiments/your_run/evaluation_results
"""
import os
import json
import argparse
from typing import Dict, Any, List

def fmt(v, nd=3):
    try:
        if v is None: return 'NA'
        return f"{float(v):.{nd}f}"
    except Exception:
        return 'NA'

TASKS = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
COLUMNS = [
    ('rmse', 'RMSE (↓)'),
    ('mae', 'MAE (↓)'),
    ('r2', 'R$^2$ (↑)'),
    ('ccc', 'CCC (↑)'),
    ('pearson', 'Pearson (↑)'),
    ('spearman', 'Spearman (↑)'),
    ('qwk', 'QWK (↑)')
]

def load_metrics(eval_dir: str) -> Dict[str, Any]:
    mpath = os.path.join(eval_dir, 'metrics.json')
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"metrics.json 不存在: {mpath}")
    with open(mpath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_ci(eval_dir: str) -> Dict[str, Any]:
    cpath = os.path.join(eval_dir, 'metrics_ci.json')
    if os.path.exists(cpath):
        with open(cpath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def render_latex(metrics: Dict[str, Any], ci: Dict[str, Any]) -> str:
    header_cols = ' & '.join(["任务"] + [label for _, label in COLUMNS]) + r" \\ \hline"
    lines = [r"\begin{tabular}{l" + "c"*len(COLUMNS) + r"}", r"\hline", header_cols]
    # 每个任务
    for t in TASKS:
        row = [
            { 'lip_sync':'口型同步','expression':'表情自然度','audio_quality':'音频质量','cross_modal':'跨模态一致性','overall':'总体'}.get(t,t)
        ]
        for key, _ in COLUMNS:
            val = metrics.get(t, {}).get(key, None)
            cell = fmt(val)
            # 附带 95% CI（若有）
            ci_t = ci.get(t, {})
            if key in ci_t:
                low = ci_t[key].get('low', None)
                high = ci_t[key].get('high', None)
                if low is not None and high is not None:
                    cell += f" ({fmt(low)}–{fmt(high)})"
            row.append(cell)
    lines.append(' & '.join(row) + " \\")
    # 平均行
    avg = metrics.get('average', {})
    avg_row = ["平均"]
    for key, _ in COLUMNS:
        avg_row.append(fmt(avg.get(key, None)))
    lines.append(r"\hline")
    lines.append(' & '.join(avg_row) + " \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return '\n'.join(lines)

def render_markdown(metrics: Dict[str, Any], ci: Dict[str, Any]) -> str:
    header = '| 任务 | ' + ' | '.join([label for _, label in COLUMNS]) + ' |\n'
    sep = '|' + '---|'*(len(COLUMNS)+1) + '\n'
    lines = [header, sep]
    for t in TASKS:
        name = { 'lip_sync':'口型同步','expression':'表情自然度','audio_quality':'音频质量','cross_modal':'跨模态一致性','overall':'总体'}.get(t,t)
        row = [name]
        for key, _ in COLUMNS:
            val = metrics.get(t, {}).get(key, None)
            cell = fmt(val)
            ci_t = ci.get(t, {})
            if key in ci_t:
                low = ci_t[key].get('low', None)
                high = ci_t[key].get('high', None)
                if low is not None and high is not None:
                    cell += f" ({fmt(low)}–{fmt(high)})"
            row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |\n')
    # 平均
    avg = metrics.get('average', {})
    avg_row_vals = [fmt(avg.get(key, None)) for key, _ in COLUMNS]
    lines.append('| 平均 | ' + ' | '.join(avg_row_vals) + ' |\n')
    return ''.join(lines)

def write_fig_captions(out_dir: str):
    text = (
        "图1. 各任务预测值与真实值的散点图；对角虚线表示理想拟合。\n\n"
        "图2. 各任务预测误差直方图（含核密度曲线），展示误差分布与偏态。\n\n"
        "图3. 真实/预测的相关性热图，反映任务间与模型拟合的相关结构。\n\n"
        "图4. 各任务真实与预测分数的箱线图，展示分布差异与异常值。\n\n"
        "图5. 主要评估指标（RMSE、MAE、R²、Pearson、Spearman、CCC、QWK）的条形比较图。\n\n"
        "图6. 总体评分的 Bland–Altman 图，展示系统性偏差与一致性界限。\n"
    )
    with open(os.path.join(out_dir, 'figure_captions.md'), 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_dir', required=True, help='Evaluator 输出目录（包含 metrics.json 等）')
    args = ap.parse_args()

    metrics = load_metrics(args.eval_dir)
    ci = load_ci(args.eval_dir)

    tex = render_latex(metrics, ci)
    md  = render_markdown(metrics, ci)

    with open(os.path.join(args.eval_dir, 'paper_metrics_table.tex'), 'w', encoding='utf-8') as f:
        f.write(tex)
    with open(os.path.join(args.eval_dir, 'paper_metrics_table.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    write_fig_captions(args.eval_dir)
    print('Saved:')
    print(' -', os.path.join(args.eval_dir, 'paper_metrics_table.tex'))
    print(' -', os.path.join(args.eval_dir, 'paper_metrics_table.md'))
    print(' -', os.path.join(args.eval_dir, 'figure_captions.md'))

if __name__ == '__main__':
    main()
