#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 批量视频评估示例

此脚本演示如何使用训练好的模型批量评估多个视频的质量。

使用方法：
    python batch_evaluate.py --videos_dir path/to/videos --model_path path/to/model.pth --config_path path/to/config.yaml
"""

import os
import sys
import argparse
import torch
import time
import glob
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_logging, load_config, get_device, format_time, save_json
from models import TalkingFaceEvaluationModel
from evaluation import VideoQualityEvaluator


def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="批量评估AI生成说话人脸视频的质量")
    
    parser.add_argument("--videos_dir", type=str, required=True,
                        help="包含视频文件的目录路径")
    parser.add_argument("--video_ext", type=str, default="mp4",
                        help="视频文件扩展名")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型权重路径")
    parser.add_argument("--config_path", type=str, default="../config/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./batch_results",
                        help="评估结果输出目录")
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="是否使用CUDA加速")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="是否显示详细信息")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    logger = setup_logging(os.path.join(args.output_dir, "logs"))
    logger.info("开始批量评估视频质量")
    logger.info(f"视频目录: {args.videos_dir}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"配置路径: {args.config_path}")
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 获取设备
    device = get_device(args.use_cuda)
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # 加载模型
    logger.info("加载模型...")
    model = TalkingFaceEvaluationModel(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("模型加载完成")
    
    # 创建评估器
    evaluator = VideoQualityEvaluator(model, config, device)
    
    # 获取视频文件列表
    video_pattern = os.path.join(args.videos_dir, f"*.{args.video_ext}")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        logger.error(f"未找到任何 .{args.video_ext} 视频文件在 {args.videos_dir} 目录中")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    
    # 批量评估视频
    results = []
    total_start_time = time.time()
    
    for video_path in tqdm(video_files, desc="评估进度"):
        video_name = os.path.basename(video_path)
        video_output_dir = os.path.join(reports_dir, os.path.splitext(video_name)[0])
        os.makedirs(video_output_dir, exist_ok=True)
        
        logger.info(f"评估视频: {video_name}")
        start_time = time.time()
        
        try:
            # 执行评估
            result = evaluator.evaluate_video(
                video_path=video_path,
                output_dir=video_output_dir,
                verbose=args.verbose
            )
            
            # 添加视频名称到结果
            result['video_name'] = video_name
            results.append(result)
            
            elapsed_time = time.time() - start_time
            logger.info(f"评估完成，耗时: {format_time(elapsed_time)}")
            
        except Exception as e:
            logger.error(f"评估视频 {video_name} 时出错: {str(e)}")
    
    total_time = time.time() - total_start_time
    logger.info(f"批量评估完成，总耗时: {format_time(total_time)}")
    
    # 保存汇总结果
    if results:
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
        
        # 保存为CSV
        csv_path = os.path.join(args.output_dir, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"汇总结果已保存至: {csv_path}")
        
        # 保存为JSON
        json_path = os.path.join(args.output_dir, "results_detailed.json")
        save_json(results, json_path)
        logger.info(f"详细结果已保存至: {json_path}")
        
        # 打印评估结果摘要
        print("\n" + "=" * 50)
        print("批量视频质量评估结果摘要")
        print("=" * 50)
        print(f"评估视频总数: {len(results)}")
        print(f"平均总体评分: {df['overall_score'].mean():.2f}/5.0")
        print("\n各项指标平均得分:")
        print(f"- 口型同步性: {df['lip_sync_score'].mean():.2f}/5.0")
        print(f"- 表情自然度: {df['expression_score'].mean():.2f}/5.0")
        print(f"- 音频质量: {df['audio_score'].mean():.2f}/5.0")
        print(f"- 跨模态一致性: {df['cross_modal_score'].mean():.2f}/5.0")
        print("\n评分分布:")
        for grade in sorted(df['overall_grade'].unique()):
            count = (df['overall_grade'] == grade).sum()
            percentage = count / len(df) * 100
            print(f"- {grade}: {count} 个视频 ({percentage:.1f}%)")
        print("\n详细报告已保存至:")
        print(f"{reports_dir}")
        print("=" * 50)
    
    return results


if __name__ == "__main__":
    main()