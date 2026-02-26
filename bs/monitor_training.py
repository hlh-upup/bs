#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时监控训练效果的脚本
"""

import os
import sys
import time
import subprocess
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def monitor_training(experiment_dir="experiments_improved", interval=30):
    """实时监控训练进度"""
    
    print(f"🔍 开始监控训练进度 (实验目录: {experiment_dir})")
    print(f"⏱️  刷新间隔: {interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    log_dir = Path(experiment_dir) / "logs"
    
    if not log_dir.exists():
        print(f"❌ 日志目录不存在: {log_dir}")
        return
    
    # 记录训练历史
    training_history = []
    
    try:
        while True:
            # 查找最新的日志文件
            log_files = list(log_dir.glob("*.log"))
            if not log_files:
                print("⏳ 等待日志文件...")
                time.sleep(interval)
                continue
            
            # 获取最新的日志文件
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            # 读取日志内容
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 解析关键指标
            metrics = parse_log_metrics(log_content)
            
            if metrics:
                # 显示当前状态
                display_current_status(metrics)
                
                # 记录历史
                training_history.append({
                    'timestamp': datetime.now(),
                    **metrics
                })
                
                # 绘制实时图表
                if len(training_history) > 1:
                    plot_training_progress(training_history, experiment_dir)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")
        
        # 生成最终报告
        if training_history:
            generate_final_report(training_history, experiment_dir)

def parse_log_metrics(log_content):
    """从日志中解析关键指标"""
    
    metrics = {}
    
    # 查找epoch信息
    epoch_match = re.search(r'Epoch (\d+)/(\d+)', log_content)
    if epoch_match:
        metrics['epoch'] = int(epoch_match.group(1))
        metrics['total_epochs'] = int(epoch_match.group(2))
    
    # 查找损失信息
    loss_patterns = {
        'train_loss': r'Train Loss:\s*([\d.]+)',
        'val_loss': r'Validation Loss:\s*([\d.]+)',
        'lip_sync_loss': r'lip_sync loss:\s*([\d.]+)',
        'expression_loss': r'expression loss:\s*([\d.]+)',
        'audio_quality_loss': r'audio_quality loss:\s*([\d.]+)',
        'cross_modal_loss': r'cross_modal loss:\s*([\d.]+)',
    }
    
    for metric_name, pattern in loss_patterns.items():
        match = re.search(pattern, log_content)
        if match:
            metrics[metric_name] = float(match.group(1))
    
    # 查找评估指标
    eval_patterns = {
        'pearson': r'Pearson.*?([\d.]+)',
        'r2': r'R².*?([\d.]+)',
        'mse': r'MSE.*?([\d.]+)',
        'mae': r'MAE.*?([\d.]+)',
    }
    
    for metric_name, pattern in eval_patterns.items():
        matches = re.findall(pattern, log_content)
        if matches:
            metrics[f'{metric_name}_latest'] = float(matches[-1])
    
    return metrics

def display_current_status(metrics):
    """显示当前训练状态"""
    
    # 清屏 (Windows)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚀 AI生成说话人脸视频评价模型 - 实时监控")
    print("="*60)
    
    # 基本信息
    if 'epoch' in metrics and 'total_epochs' in metrics:
        progress = metrics['epoch'] / metrics['total_epochs'] * 100
        print(f"📊 训练进度: Epoch {metrics['epoch']}/{metrics['total_epochs']} ({progress:.1f}%)")
    
    # 损失信息
    if 'train_loss' in metrics:
        print(f"📉 训练损失: {metrics['train_loss']:.4f}")
    
    if 'val_loss' in metrics:
        print(f"📉 验证损失: {metrics['val_loss']:.4f}")
    
    # 任务损失
    task_losses = ['lip_sync_loss', 'expression_loss', 'audio_quality_loss', 'cross_modal_loss']
    if any(loss in metrics for loss in task_losses):
        print("\n🎯 各任务损失:")
        for loss_name in task_losses:
            if loss_name in metrics:
                task_name = loss_name.replace('_loss', '').replace('_', ' ').title()
                print(f"   {task_name}: {metrics[loss_name]:.4f}")
    
    # 评估指标
    eval_metrics = ['pearson_latest', 'r2_latest', 'mse_latest', 'mae_latest']
    if any(metric in metrics for metric in eval_metrics):
        print("\n📈 最新评估指标:")
        if 'pearson_latest' in metrics:
            print(f"   Pearson相关系数: {metrics['pearson_latest']:.4f}")
        if 'r2_latest' in metrics:
            print(f"   R²分数: {metrics['r2_latest']:.4f}")
        if 'mse_latest' in metrics:
            print(f"   MSE: {metrics['mse_latest']:.4f}")
        if 'mae_latest' in metrics:
            print(f"   MAE: {metrics['mae_latest']:.4f}")
    
    print("\n" + "="*60)
    print("💡 提示: 按 Ctrl+C 停止监控")
    print("📁 日志位置: experiments_improved/logs/")
    print("📊 模型检查点: experiments_improved/checkpoints/")

def plot_training_progress(history, experiment_dir):
    """绘制训练进度图表"""
    
    if len(history) < 2:
        return
    
    # 创建图表目录
    plot_dir = Path(experiment_dir) / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # 准备数据
    df = pd.DataFrame(history)
    
    # 绘制损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练和验证损失
    if 'train_loss' in df.columns and 'val_loss' in df.columns:
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['train_loss'], 'b-', label='Training Loss')
        ax1.plot(df.index, df['val_loss'], 'r-', label='Validation Loss')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 任务损失
    task_losses = ['lip_sync_loss', 'expression_loss', 'audio_quality_loss', 'cross_modal_loss']
    available_task_losses = [col for col in task_losses if col in df.columns]
    
    if available_task_losses:
        ax2 = axes[0, 1]
        for loss_name in available_task_losses:
            task_name = loss_name.replace('_loss', '').replace('_', ' ').title()
            ax2.plot(df.index, df[loss_name], label=task_name)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Loss')
        ax2.set_title('Task-specific Losses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 评估指标
    if 'pearson_latest' in df.columns:
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['pearson_latest'], 'g-', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Pearson Correlation')
        ax3.set_title('Pearson Correlation Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # R² 分数
    if 'r2_latest' in df.columns:
        ax4 = axes[1, 1]
        ax4.plot(df.index, df['r2_latest'], 'm-', linewidth=2)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('R² Score')
        ax4.set_title('R² Score Over Time')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_final_report(history, experiment_dir):
    """生成最终监控报告"""
    
    if not history:
        return
    
    # 创建报告目录
    report_dir = Path(experiment_dir) / "monitoring_report"
    report_dir.mkdir(exist_ok=True)
    
    # 生成报告
    report_path = report_dir / "monitoring_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("训练监控报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据点数量: {len(history)}\n\n")
        
        # 获取最新数据
        latest = history[-1]
        
        f.write("📊 最终状态:\n")
        f.write("-"*40 + "\n")
        
        if 'epoch' in latest:
            f.write(f"训练轮数: {latest.get('epoch', 'N/A')}\n")
        
        if 'train_loss' in latest:
            f.write(f"最终训练损失: {latest['train_loss']:.4f}\n")
        
        if 'val_loss' in latest:
            f.write(f"最终验证损失: {latest['val_loss']:.4f}\n")
        
        if 'pearson_latest' in latest:
            f.write(f"最终Pearson相关系数: {latest['pearson_latest']:.4f}\n")
        
        if 'r2_latest' in latest:
            f.write(f"最终R²分数: {latest['r2_latest']:.4f}\n")
        
        f.write("\n💡 训练建议:\n")
        f.write("-"*40 + "\n")
        
        # 分析趋势并给出建议
        if len(history) > 5:
            recent_losses = [h.get('val_loss', float('inf')) for h in history[-5:]]
            if recent_losses[-1] < min(recent_losses[:-1]):
                f.write("✅ 验证损失呈下降趋势，训练正常\n")
            else:
                f.write("⚠️  验证损失没有明显改善，建议检查学习率或模型复杂度\n")
            
            if 'pearson_latest' in latest:
                if latest['pearson_latest'] > 0.7:
                    f.write("✅ Pearson相关系数良好，模型性能优秀\n")
                elif latest['pearson_latest'] > 0.5:
                    f.write("📈 Pearson相关系数中等，还有优化空间\n")
                else:
                    f.write("⚠️  Pearson相关系数较低，需要改进模型\n")
    
    print(f"\n📄 监控报告已保存: {report_path}")

def check_training_status():
    """检查训练状态"""
    
    experiments = ["experiments_original", "experiments_improved"]
    
    print("🔍 检查实验状态...\n")
    
    for exp_name in experiments:
        exp_path = Path(exp_name)
        if exp_path.exists():
            print(f"📁 {exp_name}:")
            
            # 检查日志
            log_dir = exp_path / "logs"
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    print(f"   📄 日志文件: {latest_log.name}")
                    
                    # 读取最新epoch
                    with open(latest_log, 'r') as f:
                        content = f.read()
                        epoch_match = re.search(r'Epoch (\d+)/(\d+)', content)
                        if epoch_match:
                            print(f"   📊 最新进度: Epoch {epoch_match.group(1)}/{epoch_match.group(2)}")
                else:
                    print("   ⏳ 尚无日志文件")
            else:
                print("   ❌ 日志目录不存在")
            
            # 检查检查点
            checkpoint_dir = exp_path / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                if checkpoints:
                    print(f"   💾 模型检查点: {len(checkpoints)} 个")
                else:
                    print("   ⏳ 尚无模型检查点")
            else:
                print("   ❌ 检查点目录不存在")
            
            print()
        else:
            print(f"❌ {exp_name}: 实验目录不存在\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="监控训练进度")
    parser.add_argument("--mode", choices=["monitor", "check"], default="check",
                       help="运行模式: monitor(实时监控) 或 check(快速检查)")
    parser.add_argument("--experiment", default="experiments_improved",
                       help="实验目录路径")
    parser.add_argument("--interval", type=int, default=30,
                       help="监控刷新间隔(秒)")
    
    args = parser.parse_args()
    
    if args.mode == "check":
        check_training_status()
    elif args.mode == "monitor":
        monitor_training(args.experiment, args.interval)

if __name__ == "__main__":
    main()