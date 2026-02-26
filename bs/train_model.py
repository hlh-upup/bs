#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

使用方法：
    python train_model.py --config_path path/to/config.yaml --data_dir path/to/data --output_dir path/to/output
"""

import os
import sys
import argparse
import torch
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_config, save_config, set_seed, get_device, format_time, create_experiment_dir
from models import create_model
from data import create_dataloaders_from_pkl
from training import Trainer
from evaluation import Evaluator


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="训练AI生成说话人脸视频评价模型")
    
    parser.add_argument("--config_path", type=str, default="../config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--dataset_path", type=str, default="../datasets/ac.pkl",
                        help="处理后的数据集文件路径 (.pkl)")
    parser.add_argument("--output_dir", type=str, default="../experiments",
                        help="输出目录路径")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="实验名称，默认使用时间戳")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="是否仅进行数据预处理")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="是否使用CUDA加速")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建实验目录
    experiment_dir = create_experiment_dir(args.output_dir, args.experiment_name)
    
    # 设置日志
    logger = setup_logging(os.path.join(experiment_dir, "logs"))
    logger.info("开始训练AI生成说话人脸视频评价模型")
    logger.info(f"配置路径: {args.config_path}")
    logger.info(f"数据集路径: {args.dataset_path}")
    logger.info(f"输出目录: {experiment_dir}")
    
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"随机种子: {args.seed}")
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 更新配置中的路径
    config['data']['dataset_path'] = args.dataset_path
    if 'train' not in config:
        config['train'] = {}
    config['train']['output_dir'] = experiment_dir
    
    # 保存配置到实验目录
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    save_config(config, config_save_path)
    logger.info(f"配置已保存至: {config_save_path}")
    
    # 获取设备
    device = get_device(args.use_cuda)
    logger.info(f"使用设备: {device}")
    
    # 如果指定了preprocess_only，则先执行数据准备脚本
    if args.preprocess_only:
        logger.info("检测到 --preprocess_only 参数，将执行数据准备...")
        # 此处可以调用 analyze_and_create_dataset.py，但更推荐用户手动运行
        logger.warning("请手动运行 'analyze_and_create_dataset.py' 来生成最新的数据集文件。")
        logger.info("训练过程已跳过。")
        return

    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders_from_pkl(
        dataset_pkl_path=args.dataset_path,
        config=config
    )
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config['model'])
    model.to(device)
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"恢复训练从第 {start_epoch} 轮开始")
    
    # 训练模型
    logger.info("开始训练模型...")
    train_start_time = time.time()
    
    final_results = trainer.train()
    
    train_time = time.time() - train_start_time
    logger.info(f"模型训练完成，耗时: {format_time(train_time)}")
    logger.info(f"最佳模型在第 {final_results['best_epoch']} 轮")
    
    # 评估模型
    logger.info("开始评估模型...")
    eval_start_time = time.time()
    
    # 获取最佳模型路径
    best_model_path = os.path.join(experiment_dir, 'checkpoints', 'best_model.pth')
    logger.info(f"最佳模型保存路径: {best_model_path}")
    
    # 加载最佳模型
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载最佳模型 (epoch {checkpoint['epoch']})")
    else:
        logger.warning(f"最佳模型文件不存在: {best_model_path}")
    
    # 创建评估器
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        output_dir=os.path.join(experiment_dir, "results")
    )
    
    # 执行评估
    metrics = evaluator.evaluate()
    
    eval_time = time.time() - eval_start_time
    logger.info(f"模型评估完成，耗时: {format_time(eval_time)}")
    
    # 生成评估报告
    report_path = evaluator.generate_report()
    logger.info(f"评估报告已保存至: {report_path}")
    
    # 打印评估结果摘要
    print("\n" + "=" * 50)
    print("模型评估结果摘要")
    print("=" * 50)
    print(f"总体MSE: {metrics['overall']['mse']:.4f}")
    print(f"总体MAE: {metrics['overall']['mae']:.4f}")
    print(f"总体R²: {metrics['overall']['r2']:.4f}")
    print(f"总体Pearson相关系数: {metrics['overall']['pearson']:.4f}")
    print("\n各任务评估结果:")
    for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal']:
        print(f"- {task}:")
        print(f"  MSE: {metrics[task]['mse']:.4f}")
        print(f"  MAE: {metrics[task]['mae']:.4f}")
        print(f"  R²: {metrics[task]['r2']:.4f}")
        print(f"  Pearson: {metrics[task]['pearson']:.4f}")
    print("=" * 50)
    
    # 总结训练过程
    total_time = train_time + eval_time
    logger.info(f"总耗时: {format_time(total_time)}")
    logger.info("训练过程已完成")
    
    return metrics


if __name__ == "__main__":
    main()