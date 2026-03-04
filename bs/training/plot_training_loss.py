#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练损失曲线可视化脚本

用于绘制多模态多任务说话人脸视频质量评估模型的训练过程损失曲线。
生成两类图表：
  1. 总损失的训练/验证曲线对比图
  2. 五个评估维度的分维度损失曲线对比图

使用方法:
    python plot_training_loss.py

输出文件:
    training_loss_total.png     - 总损失曲线
    training_loss_dimensions.png - 五维度分维度损失曲线
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- 中文字体配置 ----
# 尝试使用系统中文字体，否则回退到英文
def setup_chinese_font():
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                     'Noto Sans CJK SC', 'Source Han Sans SC', 'STSong']
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return True
    return False

USE_CHINESE = setup_chinese_font()

# 标签映射
if USE_CHINESE:
    LABELS = {
        'total_loss': '总损失（训练）',
        'val_total_loss': '总损失（验证）',
        'lip_sync': '唇形同步',
        'expression': '表情自然度',
        'audio_quality': '音频质量',
        'cross_modal': '跨模态一致性',
        'overall': '整体感知质量',
        'epoch': '训练轮次 (Epoch)',
        'loss': '损失值 (Loss)',
        'title_total': '模型训练过程总损失曲线',
        'title_dims': '五维度分任务训练损失曲线',
        'train': '训练',
        'val': '验证',
        'best_epoch': '最优检查点',
    }
else:
    LABELS = {
        'total_loss': 'Total Loss (Train)',
        'val_total_loss': 'Total Loss (Val)',
        'lip_sync': 'Lip Sync',
        'expression': 'Expression',
        'audio_quality': 'Audio Quality',
        'cross_modal': 'Cross-Modal',
        'overall': 'Overall',
        'epoch': 'Epoch',
        'loss': 'Loss',
        'title_total': 'Training Loss Curve',
        'title_dims': 'Per-Dimension Training Loss Curves',
        'train': 'Train',
        'val': 'Val',
        'best_epoch': 'Best Checkpoint',
    }


def load_training_data(csv_path):
    """加载训练损失数据CSV文件。"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                data[key].append(float(row[key]))
    return data


def plot_total_loss(data, save_path):
    """绘制总损失的训练/验证曲线对比图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = data['epoch']
    ax.plot(epochs, data['total_loss'], color='#2196F3', linewidth=1.8,
            label=LABELS['total_loss'], alpha=0.9)
    ax.plot(epochs, data['val_total_loss'], color='#FF5722', linewidth=1.8,
            label=LABELS['val_total_loss'], alpha=0.9, linestyle='--')

    # 标记最优验证检查点
    best_idx = data['val_total_loss'].index(min(data['val_total_loss']))
    best_epoch = int(epochs[best_idx])
    best_val = data['val_total_loss'][best_idx]
    ax.axvline(x=best_epoch, color='#4CAF50', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.scatter([best_epoch], [best_val], color='#4CAF50', s=80, zorder=5, marker='*')
    ax.annotate(f'{LABELS["best_epoch"]} (Epoch {best_epoch})',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 8, best_val + 0.15),
                fontsize=9, color='#4CAF50',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.2))

    ax.set_xlabel(LABELS['epoch'], fontsize=12)
    ax.set_ylabel(LABELS['loss'], fontsize=12)
    ax.set_title(LABELS['title_total'], fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 150)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_dimension_losses(data, save_path):
    """绘制五个评估维度的分维度损失曲线对比图。"""
    dims = [
        ('lip_sync_loss', 'val_lip_sync_loss', LABELS['lip_sync'], '#2196F3'),
        ('expression_loss', 'val_expression_loss', LABELS['expression'], '#FF9800'),
        ('audio_quality_loss', 'val_audio_quality_loss', LABELS['audio_quality'], '#4CAF50'),
        ('cross_modal_loss', 'val_cross_modal_loss', LABELS['cross_modal'], '#E91E63'),
        ('overall_loss', 'val_overall_loss', LABELS['overall'], '#9C27B0'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    epochs = data['epoch']

    for idx, (train_key, val_key, label, color) in enumerate(dims):
        ax = axes_flat[idx]
        ax.plot(epochs, data[train_key], color=color, linewidth=1.5,
                label=f'{label} ({LABELS["train"]})', alpha=0.9)
        ax.plot(epochs, data[val_key], color=color, linewidth=1.5,
                label=f'{label} ({LABELS["val"]})', alpha=0.7, linestyle='--')

        # 标记最终值
        final_train = data[train_key][-1]
        final_val = data[val_key][-1]
        ax.annotate(f'{final_train:.3f}', xy=(150, final_train),
                    fontsize=8, color=color, ha='left', va='bottom')
        ax.annotate(f'{final_val:.3f}', xy=(150, final_val),
                    fontsize=8, color=color, ha='left', va='top', alpha=0.7)

        ax.set_title(label, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel(LABELS['epoch'], fontsize=9)
        ax.set_ylabel(LABELS['loss'], fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 150)

    # 隐藏第6个子图
    axes_flat[5].set_visible(False)

    fig.suptitle(LABELS['title_dims'], fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'training_loss_data.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    data = load_training_data(csv_path)
    print(f"Loaded {len(data['epoch'])} epochs of training data.")

    # 生成图表
    plot_total_loss(data, os.path.join(script_dir, 'training_loss_total.png'))
    plot_dimension_losses(data, os.path.join(script_dir, 'training_loss_dimensions.png'))

    # 输出关键统计
    print("\n=== Training Loss Summary ===")
    best_idx = data['val_total_loss'].index(min(data['val_total_loss']))
    print(f"Best validation loss at epoch {int(data['epoch'][best_idx])}: "
          f"{data['val_total_loss'][best_idx]:.4f}")
    print(f"Final (epoch 150) train: {data['total_loss'][-1]:.4f}, "
          f"val: {data['val_total_loss'][-1]:.4f}")
    print(f"\nPer-dimension final training losses:")
    for dim in ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']:
        train_key = f'{dim}_loss'
        val_key = f'val_{dim}_loss'
        print(f"  {dim:20s}: train={data[train_key][-1]:.4f}, val={data[val_key][-1]:.4f}")


if __name__ == '__main__':
    main()
