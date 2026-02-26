#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试真实视频文件的SyncNet特征提取
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.extractor import FeatureExtractor

def get_device():
    """获取设备信息"""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        return 'cpu'

def find_test_video():
    """查找测试视频文件"""
    # 查找可能的视频文件位置
    possible_paths = [
        "datasets/ch-simsv2s/Raw",
        "datasets",
        "."
    ]
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_path = os.path.join(root, file)
                        print(f"找到视频文件: {video_path}")
                        return video_path
    
    return None

def test_real_syncnet_extraction():
    """测试真实的SyncNet特征提取"""
    print("=" * 60)
    print("测试真实视频文件的SyncNet特征提取")
    print("=" * 60)
    
    # 查找测试视频
    video_path = find_test_video()
    if not video_path:
        print("❌ 未找到测试视频文件")
        print("请确保datasets目录中有视频文件")
        return False
    
    print(f"使用视频文件: {video_path}")
    
    try:
        # 加载配置
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 暂时禁用可能有问题的特征提取器
        if 'features' in config and 'visual' in config['features']:
            del config['features']['visual']
            print("已从配置中移除visual特征提取器")
        if 'feature_extraction' in config and 'visual' in config['feature_extraction']:
            del config['feature_extraction']['visual']
            print("已从feature_extraction配置中移除visual特征提取器")
        
        # 禁用依赖visual特征的consistency特征提取器
        if 'features' in config and 'consistency' in config['features']:
            del config['features']['consistency']
            print("已从配置中移除consistency特征提取器")
        if 'feature_extraction' in config and 'consistency' in config['feature_extraction']:
            del config['feature_extraction']['consistency']
            print("已从feature_extraction配置中移除consistency特征提取器")
        
        # 创建特征提取器
        device = get_device()
        print(f"使用设备: {device}")
        
        feature_extractor = FeatureExtractor(config, device)
        print("✅ 特征提取器创建成功")
        
        # 提取特征
        print("\n开始提取特征...")
        try:
            features = feature_extractor.extract_all_features(video_path)
        except Exception as e:
            print(f"特征提取过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            features = {}
        
        print("\n特征提取结果:")
        print(f"提取到的特征数量: {len(features)}")
        print(f"特征名称: {list(features.keys())}")
        
        for name, feature in features.items():
            if isinstance(feature, np.ndarray):
                print(f"  {name}: 数组形状={feature.shape}, 类型={feature.dtype}")
                if feature.size <= 10:  # 只显示小数组的值
                    print(f"    值: {feature}")
                else:
                    print(f"    前5个值: {feature.flatten()[:5]}")
            elif isinstance(feature, dict):
                print(f"  {name}: 字典格式 - {feature}")
            else:
                print(f"  {name}: {type(feature)} - {feature}")
        
        # 特别检查SyncNet特征
        if 'syncnet' in features:
            syncnet_feature = features['syncnet']
            print(f"\n🔍 SyncNet特征详细分析:")
            print(f"  类型: {type(syncnet_feature)}")
            
            if isinstance(syncnet_feature, np.ndarray):
                print(f"  ✅ SyncNet特征已成功转换为NumPy数组")
                print(f"  形状: {syncnet_feature.shape}")
                print(f"  数据类型: {syncnet_feature.dtype}")
                print(f"  值: {syncnet_feature}")
                return True
            elif isinstance(syncnet_feature, dict):
                print(f"  ⚠️ SyncNet特征仍为字典格式: {syncnet_feature}")
                print(f"  这可能表明音频提取或特征转换过程中出现了问题")
                return False
            else:
                print(f"  ❌ SyncNet特征格式异常: {syncnet_feature}")
                return False
        else:
            print("\n❌ 未找到SyncNet特征")
            return False
            
    except Exception as e:
        print(f"\n❌ 特征提取过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_syncnet_extraction()
    if success:
        print("\n🎉 SyncNet特征提取和转换测试成功！")
    else:
        print("\n💥 SyncNet特征提取测试失败")