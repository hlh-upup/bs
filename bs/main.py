#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 主程序入口

此脚本提供了模型训练、评估和预测的命令行接口。
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import MTLTalkingFaceEvaluator
from data.dataset import TalkingFaceDataset
from utils.utils import compute_metrics
from utils import visualize_results


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(config, args):
    """训练模型"""
    print("=== 开始训练模型 ===")
    
    # 创建保存目录
    save_dir = Path(config['train']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['train']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = TalkingFaceDataset(
        data_dir=args.data_dir,
        split='train',
        config=config['data']
    )
    val_dataset = TalkingFaceDataset(
        data_dir=args.data_dir,
        split='val',
        config=config['data']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("初始化模型...")
    model = MTLTalkingFaceEvaluator(config['model'])
    model = model.to(device)
    
    # 定义优化器和学习率调度器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    if config['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['train']['epochs']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    
    # 训练循环
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(config['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['train']['epochs']}")
        
        # 训练阶段
        model.train()
        train_losses = []
        task_losses = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
        
        for i, batch in enumerate(train_loader):
            # 将数据移至设备
            visual_features = batch['visual_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            keypoints = batch['keypoints'].to(device)
            au_features = batch['au_features'].to(device)
            
            targets = {
                'lip_sync': batch['lip_sync_score'].to(device),
                'expression': batch['expression_score'].to(device),
                'audio_quality': batch['audio_quality_score'].to(device),
                'cross_modal': batch['cross_modal_score'].to(device)
            }
            
            # 前向传播
            outputs, losses = model(
                visual_features=visual_features,
                audio_features=audio_features,
                keypoints=keypoints,
                au_features=au_features,
                targets=targets
            )
            
            # 计算总损失
            total_loss = sum(losses.values())
            
            # 梯度累积
            total_loss = total_loss / config['train']['gradient_accumulation']
            total_loss.backward()
            
            if (i + 1) % config['train']['gradient_accumulation'] == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录损失
            train_losses.append(total_loss.item() * config['train']['gradient_accumulation'])
            for task, loss in losses.items():
                task_losses[task].append(loss.item())
            
            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {total_loss.item() * config['train']['gradient_accumulation']:.4f}")
        
        # 计算平均损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_task_losses = {task: sum(losses) / len(losses) for task, losses in task_losses.items()}
        
        print(f"训练损失: {avg_train_loss:.4f}")
        for task, loss in avg_task_losses.items():
            print(f"  - {task} 损失: {loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_losses = []
        val_task_losses = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
        val_predictions = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
        val_targets = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
        
        with torch.no_grad():
            for batch in val_loader:
                # 将数据移至设备
                visual_features = batch['visual_features'].to(device)
                audio_features = batch['audio_features'].to(device)
                keypoints = batch['keypoints'].to(device)
                au_features = batch['au_features'].to(device)
                
                targets = {
                    'lip_sync': batch['lip_sync_score'].to(device),
                    'expression': batch['expression_score'].to(device),
                    'audio_quality': batch['audio_quality_score'].to(device),
                    'cross_modal': batch['cross_modal_score'].to(device)
                }
                
                # 前向传播
                outputs, losses = model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoints=keypoints,
                    au_features=au_features,
                    targets=targets
                )
                
                # 计算总损失
                total_loss = sum(losses.values())
                
                # 记录损失和预测
                val_losses.append(total_loss.item())
                for task, loss in losses.items():
                    val_task_losses[task].append(loss.item())
                
                for task in outputs.keys():
                    val_predictions[task].append(outputs[task].cpu().numpy())
                    val_targets[task].append(targets[task].cpu().numpy())
        
        # 计算平均损失
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_task_losses = {task: sum(losses) / len(losses) for task, losses in val_task_losses.items()}
        
        print(f"验证损失: {avg_val_loss:.4f}")
        for task, loss in avg_val_task_losses.items():
            print(f"  - {task} 损失: {loss:.4f}")
        
        # 计算评估指标
        for task in outputs.keys():
            task_preds = np.concatenate(val_predictions[task])
            task_targets = np.concatenate(val_targets[task])
            metrics = compute_metrics(task_preds, task_targets, config['eval']['metrics'])
            print(f"  - {task} 指标:")
            for metric_name, metric_value in metrics.items():
                print(f"    - {metric_name}: {metric_value:.4f}")
        
        # 更新学习率
        if config['train']['scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            
            # 保存模型
            checkpoint_path = save_dir / f"best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"保存最佳模型到 {checkpoint_path}")
        else:
            early_stopping_counter += 1
            print(f"验证损失未改善。早停计数: {early_stopping_counter}/{config['train']['early_stopping']}")
        
        # 早停
        if early_stopping_counter >= config['train']['early_stopping']:
            print("早停触发，停止训练")
            break
        
        # 保存最后一个epoch的模型
        checkpoint_path = save_dir / f"last_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'config': config
        }, checkpoint_path)
    
    print("训练完成！")


def evaluate(config, args):
    """评估模型"""
    print("=== 开始评估模型 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据集
    print("加载测试数据集...")
    test_dataset = TalkingFaceDataset(
        data_dir=args.data_dir,
        split='test',
        config=config['data']
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    print(f"从 {args.checkpoint} 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MTLTalkingFaceEvaluator(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 评估
    test_losses = []
    test_task_losses = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
    test_predictions = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
    test_targets = {'lip_sync': [], 'expression': [], 'audio_quality': [], 'cross_modal': []}
    video_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移至设备
            visual_features = batch['visual_features'].to(device)
            audio_features = batch['audio_features'].to(device)
            keypoints = batch['keypoints'].to(device)
            au_features = batch['au_features'].to(device)
            
            targets = {
                'lip_sync': batch['lip_sync_score'].to(device),
                'expression': batch['expression_score'].to(device),
                'audio_quality': batch['audio_quality_score'].to(device),
                'cross_modal': batch['cross_modal_score'].to(device)
            }
            
            # 记录视频ID
            if 'video_id' in batch:
                video_ids.extend(batch['video_id'])
            
            # 前向传播
            outputs, losses = model(
                visual_features=visual_features,
                audio_features=audio_features,
                keypoints=keypoints,
                au_features=au_features,
                targets=targets
            )
            
            # 计算总损失
            total_loss = sum(losses.values())
            
            # 记录损失和预测
            test_losses.append(total_loss.item())
            for task, loss in losses.items():
                test_task_losses[task].append(loss.item())
            
            for task in outputs.keys():
                test_predictions[task].append(outputs[task].cpu().numpy())
                test_targets[task].append(targets[task].cpu().numpy())
    
    # 计算平均损失
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_task_losses = {task: sum(losses) / len(losses) for task, losses in test_task_losses.items()}
    
    print(f"测试损失: {avg_test_loss:.4f}")
    for task, loss in avg_test_task_losses.items():
        print(f"  - {task} 损失: {loss:.4f}")
    
    # 计算评估指标
    all_metrics = {}
    for task in outputs.keys():
        task_preds = np.concatenate(test_predictions[task])
        task_targets = np.concatenate(test_targets[task])
        metrics = compute_metrics(task_preds, task_targets, config['eval']['metrics'])
        all_metrics[task] = metrics
        print(f"  - {task} 指标:")
        for metric_name, metric_value in metrics.items():
            print(f"    - {metric_name}: {metric_value:.4f}")
    
    # 可视化结果
    if config['eval']['visualization']:
        print("生成可视化结果...")
        visualize_results(
            predictions=test_predictions,
            targets=test_targets,
            video_ids=video_ids,
            metrics=all_metrics,
            output_dir=args.output_dir
        )
        print(f"可视化结果已保存到 {args.output_dir}")
    
    print("评估完成！")


def predict(config, args):
    """对单个视频进行预测"""
    print("=== 开始预测视频质量 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"从 {args.checkpoint} 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MTLTalkingFaceEvaluator(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 预处理视频
    from data.preprocess import preprocess_video
    from data.feature_extraction import extract_features
    
    print(f"预处理视频: {args.video}")
    processed_data = preprocess_video(args.video, config['data'])
    
    print("提取特征...")
    features = extract_features(processed_data, config['feature_extraction'])
    
    # 转换为模型输入
    visual_features = torch.tensor(features['visual_features']).unsqueeze(0).to(device)
    audio_features = torch.tensor(features['audio_features']).unsqueeze(0).to(device)
    keypoints = torch.tensor(features['keypoints']).unsqueeze(0).to(device)
    au_features = torch.tensor(features['au_features']).unsqueeze(0).to(device)
    
    # 预测
    print("进行预测...")
    with torch.no_grad():
        outputs, _ = model(
            visual_features=visual_features,
            audio_features=audio_features,
            keypoints=keypoints,
            au_features=au_features
        )
    
    # 输出结果
    print("\n预测结果:")
    for task, score in outputs.items():
        score_value = score.item()
        threshold = config['inference']['threshold'][task]
        quality = "良好" if score_value >= threshold else "不佳"
        print(f"  - {task}: {score_value:.2f}/5.0 ({quality})")
    
    # 保存结果
    if args.output_dir:
        import json
        from pathlib import Path
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(args.video).stem
        output_file = output_dir / f"{video_name}_evaluation.json"
        
        results = {
            "video": args.video,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scores": {task: score.item() for task, score in outputs.items()},
            "thresholds": config['inference']['threshold'],
            "quality": {task: "良好" if score.item() >= config['inference']['threshold'][task] else "不佳" 
                      for task, score in outputs.items()}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到 {output_file}")
    
    print("预测完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI生成说话人脸视频评价模型")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'predict'],
                        help='运行模式: train (训练), eval (评估), predict (预测)')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (用于评估和预测)')
    parser.add_argument('--video', type=str, default=None,
                        help='要评估的视频路径 (用于预测)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 根据模式执行相应功能
    if args.mode == 'train':
        if args.data_dir is None:
            raise ValueError("训练模式需要指定 --data_dir")
        train(config, args)
    elif args.mode == 'eval':
        if args.data_dir is None or args.checkpoint is None:
            raise ValueError("评估模式需要指定 --data_dir 和 --checkpoint")
        evaluate(config, args)
    elif args.mode == 'predict':
        if args.video is None or args.checkpoint is None:
            raise ValueError("预测模式需要指定 --video 和 --checkpoint")
        predict(config, args)


if __name__ == "__main__":
    main()