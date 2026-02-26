#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 增强特征提取模块

带有详细日志记录功能的特征提取器，记录成功和失败的处理结果。
"""

import os
import logging
import datetime
from typing import Dict, List, Tuple
from .extractor import FeatureExtractor

class EnhancedFeatureExtractor(FeatureExtractor):
    """增强的特征提取器
    
    在原有功能基础上添加详细的日志记录功能，包括：
    - 成功处理的视频日志
    - 失败处理的视频日志
    - 统计信息
    """
    
    def __init__(self, config, device, log_dir="logs"):
        super().__init__(config, device)
        self.log_dir = log_dir
        self.success_count = 0
        self.failure_count = 0
        self.success_videos = []
        self.failed_videos = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件路径
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.success_log_path = os.path.join(log_dir, f"feature_extraction_success_{timestamp}.log")
        self.failure_log_path = os.path.join(log_dir, f"feature_extraction_failure_{timestamp}.log")
        self.summary_log_path = os.path.join(log_dir, f"feature_extraction_summary_{timestamp}.log")
        
        # 初始化日志文件
        self._init_log_files()
        
        # 设置专用logger
        self.logger = logging.getLogger(f"{__name__}.enhanced")
        self.logger.setLevel(logging.INFO)
        
        # 添加文件处理器
        if not self.logger.handlers:
            handler = logging.FileHandler(self.summary_log_path, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _init_log_files(self):
        """初始化日志文件"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 初始化成功日志文件
        with open(self.success_log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 特征提取成功日志\n")
            f.write(f"# 开始时间: {timestamp}\n")
            f.write(f"# 格式: [时间] 视频路径 | 提取的特征类型\n")
            f.write("\n")
        
        # 初始化失败日志文件
        with open(self.failure_log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 特征提取失败日志\n")
            f.write(f"# 开始时间: {timestamp}\n")
            f.write(f"# 格式: [时间] 视频路径 | 错误类型 | 错误信息\n")
            f.write("\n")
    
    def extract_all_features(self, video_path: str) -> Dict:
        """提取所有特征并记录结果
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            Dict: 提取的特征字典，如果失败则返回空字典
        """
        video_name = os.path.basename(video_path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # 调用父类方法提取特征
            features = super().extract_all_features(video_path)
            
            # 确保SyncNet特征被正确转换为NumPy数组
            if 'syncnet' in features and isinstance(features['syncnet'], dict):
                import numpy as np
                features['syncnet'] = np.array([
                    features['syncnet'].get('sync_score', 0.0), 
                    features['syncnet'].get('offset', 0.0)
                ])
            
            # 记录成功
            self._log_success(video_path, features, timestamp)
            self.success_count += 1
            self.success_videos.append(video_path)
            
            self.logger.info(f"成功提取特征: {video_name}")
            return features
            
        except Exception as e:
            # 记录失败
            error_type = type(e).__name__
            error_msg = str(e)
            self._log_failure(video_path, error_type, error_msg, timestamp)
            self.failure_count += 1
            self.failed_videos.append((video_path, error_type, error_msg))
            
            self.logger.error(f"特征提取失败: {video_name} - {error_type}: {error_msg}")
            return {}
    
    def _log_success(self, video_path: str, features: Dict, timestamp: str):
        """记录成功的特征提取"""
        feature_types = list(features.keys())
        feature_info = ", ".join(feature_types)
        
        with open(self.success_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {video_path} | {feature_info}\n")
    
    def _log_failure(self, video_path: str, error_type: str, error_msg: str, timestamp: str):
        """记录失败的特征提取"""
        with open(self.failure_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {video_path} | {error_type} | {error_msg}\n")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total_count = self.success_count + self.failure_count
        success_rate = (self.success_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'total_processed': total_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'success_videos': self.success_videos,
            'failed_videos': self.failed_videos
        }
    
    def print_summary(self):
        """打印处理摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("特征提取处理摘要")
        print("="*60)
        print(f"总处理视频数: {stats['total_processed']}")
        print(f"成功处理数: {stats['success_count']}")
        print(f"失败处理数: {stats['failure_count']}")
        print(f"成功率: {stats['success_rate']:.2f}%")
        print("="*60)
        
        if stats['failure_count'] > 0:
            print("\n失败视频列表:")
            for i, (video_path, error_type, error_msg) in enumerate(stats['failed_videos'][:10], 1):
                video_name = os.path.basename(video_path)
                print(f"{i}. {video_name} - {error_type}")
            
            if len(stats['failed_videos']) > 10:
                print(f"... 还有 {len(stats['failed_videos']) - 10} 个失败视频")
        
        print(f"\n详细日志文件:")
        print(f"成功日志: {self.success_log_path}")
        print(f"失败日志: {self.failure_log_path}")
        print(f"摘要日志: {self.summary_log_path}")
    
    def save_summary(self):
        """保存处理摘要到文件"""
        stats = self.get_statistics()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.summary_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"特征提取处理摘要 - {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(f"总处理视频数: {stats['total_processed']}\n")
            f.write(f"成功处理数: {stats['success_count']}\n")
            f.write(f"失败处理数: {stats['failure_count']}\n")
            f.write(f"成功率: {stats['success_rate']:.2f}%\n")
            
            if stats['failure_count'] > 0:
                f.write(f"\n失败视频详情:\n")
                for video_path, error_type, error_msg in stats['failed_videos']:
                    f.write(f"- {os.path.basename(video_path)}: {error_type} - {error_msg}\n")
    
    def __del__(self):
        """析构函数，保存最终摘要"""
        try:
            self.save_summary()
        except:
            pass


def create_enhanced_extractor(config, device, log_dir="logs"):
    """创建增强特征提取器的工厂函数
    
    Args:
        config: 配置字典
        device: 计算设备
        log_dir: 日志目录
        
    Returns:
        EnhancedFeatureExtractor: 增强特征提取器实例
    """
    return EnhancedFeatureExtractor(config, device, log_dir)