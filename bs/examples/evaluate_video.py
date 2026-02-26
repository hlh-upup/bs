#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 视频质量评估示例

此脚本演示如何使用训练好的模型评估单个视频的质量。

使用方法：
    python evaluate_video.py --video_path path/to/video.mp4 --model_path path/to/model.pth --config_path path/to/config.yaml
"""

import os
import sys
import argparse
import torch
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_config, get_device, format_time
from models import TalkingFaceEvaluationModel
from evaluation import VideoQualityEvaluator
from features.extractor import FeatureExtractor


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="评估AI生成说话人脸视频的质量")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="要评估的视频文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型权重路径")
    parser.add_argument("--config_path", type=str, default="../config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="评估结果输出目录")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="是否使用CUDA加速")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="是否显示详细信息")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, "logs"))
    logger.info("开始评估视频质量")
    logger.info(f"视频路径: {args.video_path}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"配置路径: {args.config_path}")
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 获取设备
    device = get_device(args.use_cuda)
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    logger.info("加载模型...")
    model = TalkingFaceEvaluationModel(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("模型加载完成")
    
    # 创建特征提取器
    logger.info("初始化特征提取器...")
    feature_extractor = FeatureExtractor(config, device)
    logger.info("特征提取器初始化完成")
    
    # 创建评估器
    evaluator = VideoQualityEvaluator(model, feature_extractor, config, device)
    
    # 评估视频
    logger.info(f"开始评估视频: {args.video_path}")
    start_time = time.time()
    
    # 执行评估
    result = evaluator.evaluate_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # 计算评估时间
    elapsed_time = time.time() - start_time
    logger.info(f"评估完成，耗时: {format_time(elapsed_time)}")
    
    # 打印评估结果摘要
    print("\n" + "=" * 50)
    print("视频质量评估结果摘要")
    print("=" * 50)
    print(f"视频文件: {os.path.basename(args.video_path)}")
    print(f"总体评分: {result['overall_score']:.2f}/5.0 ({result['overall_grade']})")
    print("\n各项指标得分:")
    print(f"- 口型同步性: {result['lip_sync_score']:.2f}/5.0")
    print(f"- 表情自然度: {result['expression_score']:.2f}/5.0")
    print(f"- 音频质量: {result['audio_score']:.2f}/5.0")
    print(f"- 跨模态一致性: {result['cross_modal_score']:.2f}/5.0")
    print("\n评估总结:")
    print(result['summary'])
    print("\n详细报告已保存至:")
    print(f"{os.path.join(args.output_dir, 'report.html')}")
    print("=" * 50)
    
    logger.info(f"评估报告已保存至: {os.path.join(args.output_dir, 'report.html')}")
    
    return result


if __name__ == "__main__":
    main()