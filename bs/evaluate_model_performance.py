#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速评估模型效果的脚本
用于对比原始模型和优化模型的性能
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import pickle

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, get_device
from models import ImprovedMultiTaskTalkingFaceEvaluator, MultiTaskTalkingFaceEvaluator
from data import create_dataloaders_from_pkl

def load_model_results(experiment_dir, device):
    """加载实验结果"""
    results = {}
    
    # 加载配置
    config_path = os.path.join(experiment_dir, "config.yaml")
    if os.path.exists(config_path):
        from utils import load_config
        results['config'] = load_config(config_path)
    
    # 加载最佳模型
    best_model_path = os.path.join(experiment_dir, "checkpoints", "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        results['checkpoint'] = checkpoint
        results['best_epoch'] = checkpoint.get('epoch', 'unknown')
        results['best_val_loss'] = checkpoint.get('val_loss', float('inf'))
    
    # 查找评估结果
    results_dir = os.path.join(experiment_dir, "results")
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl') or f.endswith('.csv')]
        if result_files:
            results_path = os.path.join(results_dir, result_files[0])
            if results_path.endswith('.pkl'):
                with open(results_path, 'rb') as f:
                    results['metrics'] = pickle.load(f)
            else:
                results['metrics'] = pd.read_csv(results_path)
    
    return results

def evaluate_model(model, test_loader, device, config):
    """评估模型性能"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 准备数据
            visual_features = batch['visual_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            keypoints = batch.get('keypoints', torch.zeros(visual_features.size(0), 0).to(device))
            au_features = batch.get('au_features', torch.zeros(visual_features.size(0), 0).to(device))
            
            targets = {
                'lip_sync': batch['lip_sync_score'].to(device),
                'expression': batch['expression_score'].to(device),
                'audio_quality': batch['audio_quality_score'].to(device),
                'cross_modal': batch['cross_modal_score'].to(device),
                'overall': batch.get('overall_score', torch.zeros_like(batch['lip_sync_score'])).to(device)
            }
            
            # 预测
            if hasattr(model, 'compute_loss'):
                predictions, losses = model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoint_features=keypoints,
                    au_features=au_features,
                    targets=targets
                )
            else:
                predictions = model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoint_features=keypoints,
                    au_features=au_features
                )
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    # 计算指标
    metrics = {}
    tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
    
    for task in tasks:
        task_preds = []
        task_targets = []
        
        for pred_dict, target_dict in zip(all_predictions, all_targets):
            if task in pred_dict and task in target_dict:
                task_preds.extend(pred_dict[task].cpu().numpy().flatten())
                task_targets.extend(target_dict[task].cpu().numpy().flatten())
        
        if len(task_preds) > 0:
            task_preds = np.array(task_preds)
            task_targets = np.array(task_targets)
            
            # 计算各种指标
            mse = mean_squared_error(task_targets, task_preds)
            mae = mean_absolute_error(task_targets, task_preds)
            r2 = r2_score(task_targets, task_preds)
            pearson, _ = pearsonr(task_targets, task_preds)
            spearman, _ = spearmanr(task_targets, task_preds)
            
            metrics[task] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'pearson': pearson,
                'spearman': spearman,
                'predictions': task_preds,
                'targets': task_targets
            }
    
    return metrics

def create_comparison_report(original_metrics, improved_metrics, output_dir):
    """创建对比报告"""
    
    # 创建对比表格
    tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
    
    comparison_data = []
    for task in tasks:
        if task in original_metrics and task in improved_metrics:
            orig = original_metrics[task]
            impr = improved_metrics[task]
            
            # 计算改进百分比
            mse_improvement = (orig['mse'] - impr['mse']) / orig['mse'] * 100
            mae_improvement = (orig['mae'] - impr['mae']) / orig['mae'] * 100
            r2_improvement = (impr['r2'] - orig['r2']) / max(abs(orig['r2']), 0.001) * 100
            pearson_improvement = (impr['pearson'] - orig['pearson']) / max(abs(orig['pearson']), 0.001) * 100
            
            comparison_data.append({
                'Task': task,
                'Orig_MSE': f"{orig['mse']:.4f}",
                'Impr_MSE': f"{impr['mse']:.4f}",
                'MSE_Improvement': f"{mse_improvement:.1f}%",
                'Orig_MAE': f"{orig['mae']:.4f}",
                'Impr_MAE': f"{impr['mae']:.4f}",
                'MAE_Improvement': f"{mae_improvement:.1f}%",
                'Orig_R2': f"{orig['r2']:.4f}",
                'Impr_R2': f"{impr['r2']:.4f}",
                'R2_Improvement': f"{r2_improvement:.1f}%",
                'Orig_Pearson': f"{orig['pearson']:.4f}",
                'Impr_Pearson': f"{impr['pearson']:.4f}",
                'Pearson_Improvement': f"{pearson_improvement:.1f}%",
            })
    
    df = pd.DataFrame(comparison_data)
    
    # 保存对比表格
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    df.to_csv(comparison_path, index=False)
    
    # 创建可视化图表
    create_visualization(original_metrics, improved_metrics, output_dir)
    
    # 生成文本报告
    report_path = os.path.join(output_dir, 'performance_comparison_report.txt')
    generate_text_report(df, original_metrics, improved_metrics, report_path)
    
    return df, comparison_path

def create_visualization(original_metrics, improved_metrics, output_dir):
    """创建可视化图表"""
    
    tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
    metrics_names = ['mse', 'mae', 'r2', 'pearson']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        
        orig_values = []
        impr_values = []
        task_labels = []
        
        for task in tasks:
            if task in original_metrics and task in improved_metrics:
                orig_values.append(original_metrics[task][metric_name])
                impr_values.append(improved_metrics[task][metric_name])
                task_labels.append(task.replace('_', ' ').title())
        
        x = np.arange(len(task_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, orig_values, width, label='Original', alpha=0.8)
        bars2 = ax.bar(x + width/2, impr_values, width, label='Improved', alpha=0.8)
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f'{metric_name.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=45)
        ax.legend()
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_text_report(df, original_metrics, improved_metrics, report_path):
    """生成文本报告"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("模型性能对比报告\n")
        f.write("="*60 + "\n\n")
        
        # 总体改进情况
        f.write("📊 总体改进情况\n")
        f.write("-"*40 + "\n")
        
        avg_mse_improvement = df['MSE_Improvement'].str.replace('%', '').astype(float).mean()
        avg_mae_improvement = df['MAE_Improvement'].str.replace('%', '').astype(float).mean()
        avg_r2_improvement = df['R2_Improvement'].str.replace('%', '').astype(float).mean()
        avg_pearson_improvement = df['Pearson_Improvement'].str.replace('%', '').astype(float).mean()
        
        f.write(f"平均MSE改进: {avg_mse_improvement:.1f}%\n")
        f.write(f"平均MAE改进: {avg_mae_improvement:.1f}%\n")
        f.write(f"平均R²改进: {avg_r2_improvement:.1f}%\n")
        f.write(f"平均Pearson改进: {avg_pearson_improvement:.1f}%\n\n")
        
        # 详细对比表格
        f.write("📈 详细性能对比\n")
        f.write("-"*40 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # 关键发现
        f.write("🔍 关键发现\n")
        f.write("-"*40 + "\n")
        
        best_improvement_task = df.loc[df['Pearson_Improvement'].str.replace('%', '').astype(float).idxmax(), 'Task']
        best_improvement = df.loc[df['Pearson_Improvement'].str.replace('%', '').astype(float).idxmax(), 'Pearson_Improvement']
        
        f.write(f"• 改进最显著的任务: {best_improvement_task} (Pearson相关系数提升 {best_improvement})\n")
        
        if avg_mse_improvement > 20:
            f.write("• 整体性能显著提升，优化效果明显\n")
        elif avg_mse_improvement > 10:
            f.write("• 整体性能有良好提升\n")
        else:
            f.write("• 性能提升有限，可能需要进一步优化\n")
        
        # 建议
        f.write("\n💡 建议\n")
        f.write("-"*40 + "\n")
        f.write("1. 继续监控模型在不同任务上的表现\n")
        f.write("2. 考虑针对表现较差的任务进行专门优化\n")
        f.write("3. 可以尝试更大的模型或更长的训练时间\n")
        f.write("4. 建议收集更多数据以进一步提升性能\n")

def main():
    """主函数"""
    print("🔍 开始评估模型效果对比...")
    
    # 设置路径
    original_dir = "experiments_original"
    improved_dir = "experiments_improved"
    output_dir = "model_comparison"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    device = get_device(True)
    
    # 1. 尝试加载已有的实验结果
    print("📂 检查已有实验结果...")
    original_results = load_model_results(original_dir, device)
    improved_results = load_model_results(improved_dir, device)
    
    # 2. 如果没有完整结果，则运行评估
    if 'metrics' not in original_results or 'metrics' not in improved_results:
        print("🚀 运行模型评估...")
        
        # 加载数据
        dataset_path = "datasets/ac.pkl"
        if os.path.exists(dataset_path):
            config = load_config("config/optimized_config.yaml")
            _, _, test_loader = create_dataloaders_from_pkl(dataset_path, config)
            
            # 评估原始模型
            if 'metrics' not in original_results and 'checkpoint' in original_results:
                print("评估原始模型...")
                original_model = MultiTaskTalkingFaceEvaluator(config['model'])
                original_model.load_state_dict(original_results['checkpoint']['model_state_dict'])
                original_model.to(device)
                original_metrics = evaluate_model(original_model, test_loader, device, config)
                original_results['metrics'] = original_metrics
            
            # 评估改进模型
            if 'metrics' not in improved_results and 'checkpoint' in improved_results:
                print("评估改进模型...")
                improved_model = ImprovedMultiTaskTalkingFaceEvaluator(config['model'])
                improved_model.load_state_dict(improved_results['checkpoint']['model_state_dict'])
                improved_model.to(device)
                improved_metrics = evaluate_model(improved_model, test_loader, device, config)
                improved_results['metrics'] = improved_metrics
    
    # 3. 生成对比报告
    if 'metrics' in original_results and 'metrics' in improved_results:
        print("📊 生成对比报告...")
        df, comparison_path = create_comparison_report(
            original_results['metrics'], 
            improved_results['metrics'], 
            output_dir
        )
        
        print(f"\n✅ 评估完成！报告已保存到: {output_dir}")
        print(f"📄 对比表格: {comparison_path}")
        print(f"📊 可视化图表: {output_dir}/performance_comparison.png")
        print(f"📝 详细报告: {output_dir}/performance_comparison_report.txt")
        
        # 显示关键结果
        print("\n" + "="*50)
        print("🎯 关键结果摘要:")
        print("="*50)
        print(df[['Task', 'Orig_Pearson', 'Impr_Pearson', 'Pearson_Improvement']].to_string(index=False))
        
    else:
        print("❌ 无法找到完整的实验结果，请确保先完成模型训练")
        print("💡 建议先运行:")
        print("  python train_improved.py --config_path config/optimized_config.yaml")

if __name__ == "__main__":
    main()