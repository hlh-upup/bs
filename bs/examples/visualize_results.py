#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 结果可视化示例

此脚本演示如何可视化模型的评估结果。

使用方法：
    python visualize_results.py --results_dir path/to/results --output_dir path/to/output
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_json, ensure_dir


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="可视化AI生成说话人脸视频评价模型的评估结果")
    
    parser.add_argument("--results_dir", type=str, required=True,
                        help="包含评估结果的目录路径")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="可视化结果输出目录")
    parser.add_argument("--use_chinese_font", action="store_true", default=True,
                        help="是否使用中文字体")
    
    return parser.parse_args()


def setup_chinese_font():
    """设置中文字体
    
    Returns:
        FontProperties: 中文字体属性
    """
    # 尝试使用系统中可能存在的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'KaiTi', 'FangSong']
    
    font_prop = None
    for font in chinese_fonts:
        try:
            font_prop = FontProperties(fname=font)
            break
        except:
            continue
    
    if font_prop is None:
        print("警告：未找到中文字体，将使用默认字体")
        font_prop = FontProperties()
    
    # 设置全局字体
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    return font_prop


def plot_score_distribution(df, output_path, font_prop=None):
    """绘制评分分布图
    
    Args:
        df (pandas.DataFrame): 包含评分的DataFrame
        output_path (str): 输出文件路径
        font_prop (FontProperties, optional): 字体属性
    """
    plt.figure(figsize=(12, 8))
    
    # 设置子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('各项指标评分分布', fontsize=16, fontproperties=font_prop)
    
    # 绘制直方图
    score_columns = ['lip_sync_score', 'expression_score', 'audio_score', 'cross_modal_score']
    titles = ['口型同步性', '表情自然度', '音频质量', '跨模态一致性']
    
    for i, (col, title) in enumerate(zip(score_columns, titles)):
        ax = axes[i // 2, i % 2]
        sns.histplot(df[col], bins=20, kde=True, ax=ax)
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel('评分', fontproperties=font_prop)
        ax.set_ylabel('频数', fontproperties=font_prop)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_correlation(df, output_path, font_prop=None):
    """绘制评分相关性热图
    
    Args:
        df (pandas.DataFrame): 包含评分的DataFrame
        output_path (str): 输出文件路径
        font_prop (FontProperties, optional): 字体属性
    """
    plt.figure(figsize=(10, 8))
    
    # 计算相关系数
    score_columns = ['lip_sync_score', 'expression_score', 'audio_score', 'cross_modal_score', 'overall_score']
    corr = df[score_columns].corr()
    
    # 设置标签映射
    labels = {
        'lip_sync_score': '口型同步性',
        'expression_score': '表情自然度',
        'audio_score': '音频质量',
        'cross_modal_score': '跨模态一致性',
        'overall_score': '总体评分'
    }
    
    # 重命名索引和列
    corr.index = [labels[col] for col in corr.index]
    corr.columns = [labels[col] for col in corr.columns]
    
    # 绘制热图
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('评分指标相关性热图', fontsize=16, fontproperties=font_prop)
    
    # 设置标签字体
    if font_prop:
        plt.xticks(fontproperties=font_prop)
        plt.yticks(fontproperties=font_prop)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_grade_distribution(df, output_path, font_prop=None):
    """绘制等级分布饼图
    
    Args:
        df (pandas.DataFrame): 包含等级的DataFrame
        output_path (str): 输出文件路径
        font_prop (FontProperties, optional): 字体属性
    """
    plt.figure(figsize=(10, 8))
    
    # 计算等级分布
    grade_counts = df['overall_grade'].value_counts().sort_index()
    
    # 设置颜色映射
    colors = plt.cm.YlGnBu(np.linspace(0.2, 0.8, len(grade_counts)))
    
    # 绘制饼图
    plt.pie(
        grade_counts,
        labels=grade_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    plt.title('视频质量等级分布', fontsize=16, fontproperties=font_prop)
    plt.axis('equal')  # 保持饼图为圆形
    
    # 设置图例
    if font_prop:
        for text in plt.gca().texts:
            text.set_fontproperties(font_prop)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_chart(df, output_path, font_prop=None):
    """绘制雷达图
    
    Args:
        df (pandas.DataFrame): 包含评分的DataFrame
        output_path (str): 输出文件路径
        font_prop (FontProperties, optional): 字体属性
    """
    # 计算平均分
    score_columns = ['lip_sync_score', 'expression_score', 'audio_score', 'cross_modal_score']
    avg_scores = df[score_columns].mean().values
    
    # 设置雷达图参数
    categories = ['口型同步性', '表情自然度', '音频质量', '跨模态一致性']
    N = len(categories)
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    avg_scores = np.concatenate((avg_scores, [avg_scores[0]]))
    angles += angles[:1]
    categories += categories[:1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 绘制雷达图
    ax.plot(angles, avg_scores, 'o-', linewidth=2, label='平均评分')
    ax.fill(angles, avg_scores, alpha=0.25)
    
    # 设置刻度和标签
    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.grid(True)
    
    # 设置标题和图例
    plt.title('各项指标平均评分雷达图', fontsize=16, fontproperties=font_prop)
    
    # 设置标签字体
    if font_prop:
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_boxplot(df, output_path, font_prop=None):
    """绘制箱线图
    
    Args:
        df (pandas.DataFrame): 包含评分的DataFrame
        output_path (str): 输出文件路径
        font_prop (FontProperties, optional): 字体属性
    """
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    score_columns = ['lip_sync_score', 'expression_score', 'audio_score', 'cross_modal_score', 'overall_score']
    labels = {
        'lip_sync_score': '口型同步性',
        'expression_score': '表情自然度',
        'audio_score': '音频质量',
        'cross_modal_score': '跨模态一致性',
        'overall_score': '总体评分'
    }
    
    # 转换数据格式
    data = []
    for col in score_columns:
        for score in df[col]:
            data.append({'指标': labels[col], '评分': score})
    
    plot_df = pd.DataFrame(data)
    
    # 绘制箱线图
    sns.boxplot(x='指标', y='评分', data=plot_df)
    plt.title('各项指标评分分布箱线图', fontsize=16, fontproperties=font_prop)
    plt.xlabel('评估指标', fontsize=12, fontproperties=font_prop)
    plt.ylabel('评分', fontsize=12, fontproperties=font_prop)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置标签字体
    if font_prop:
        plt.xticks(fontproperties=font_prop)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始可视化评估结果")
    logger.info(f"结果目录: {args.results_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    ensure_dir(args.output_dir)
    
    # 设置中文字体
    font_prop = None
    if args.use_chinese_font:
        font_prop = setup_chinese_font()
    
    # 加载结果数据
    results_file = os.path.join(args.results_dir, "results_detailed.json")
    if not os.path.exists(results_file):
        logger.error(f"结果文件不存在: {results_file}")
        return
    
    results = load_json(results_file)
    logger.info(f"加载了 {len(results)} 个评估结果")
    
    # 创建DataFrame
    df = pd.DataFrame([
        {
            'video_name': r['video_name'],
            'overall_score': r['overall_score'],
            'overall_grade': r['overall_grade'],
            'lip_sync_score': r['lip_sync_score'],
            'expression_score': r['expression_score'],
            'audio_score': r['audio_score'],
            'cross_modal_score': r['cross_modal_score']
        } for r in results
    ])
    
    # 生成可视化
    logger.info("生成评分分布图...")
    plot_score_distribution(
        df,
        os.path.join(args.output_dir, "score_distribution.png"),
        font_prop
    )
    
    logger.info("生成评分相关性热图...")
    plot_score_correlation(
        df,
        os.path.join(args.output_dir, "score_correlation.png"),
        font_prop
    )
    
    logger.info("生成等级分布饼图...")
    plot_grade_distribution(
        df,
        os.path.join(args.output_dir, "grade_distribution.png"),
        font_prop
    )
    
    logger.info("生成雷达图...")
    plot_radar_chart(
        df,
        os.path.join(args.output_dir, "radar_chart.png"),
        font_prop
    )
    
    logger.info("生成箱线图...")
    plot_boxplot(
        df,
        os.path.join(args.output_dir, "boxplot.png"),
        font_prop
    )
    
    logger.info(f"可视化结果已保存至: {args.output_dir}")
    
    # 打印统计摘要
    print("\n" + "=" * 50)
    print("评估结果统计摘要")
    print("=" * 50)
    print(f"评估视频总数: {len(df)}")
    print("\n各项指标统计:")
    
    stats = df[['lip_sync_score', 'expression_score', 'audio_score', 'cross_modal_score', 'overall_score']].describe()
    stats_formatted = stats.round(2).loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    # 重命名索引和列
    stats_formatted.index = ['平均值', '标准差', '最小值', '25%分位数', '中位数', '75%分位数', '最大值']
    stats_formatted.columns = ['口型同步性', '表情自然度', '音频质量', '跨模态一致性', '总体评分']
    
    print(stats_formatted.to_string())
    print("\n等级分布:")
    grade_counts = df['overall_grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        percentage = count / len(df) * 100
        print(f"- {grade}: {count} 个视频 ({percentage:.1f}%)")
    
    print("\n可视化结果已保存至:")
    print(f"{args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()