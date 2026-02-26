#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版PKL文件检查器
"""

import pickle
import os
import numpy as np

def inspect_pkl_file(pkl_path):
    """详细检查pkl文件的内容"""
    
    if not os.path.exists(pkl_path):
        print(f"文件不存在: {pkl_path}")
        return False
    
    try:
        # 获取文件大小
        file_size = os.path.getsize(pkl_path) / 1024**2  # MB
        print(f"文件: {pkl_path}")
        print(f"文件大小: {file_size:.2f} MB")
        print("=" * 60)
        
        # 加载数据
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"数据类型: {type(data)}")
        print(f"数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        print("=" * 60)
        
        if isinstance(data, list) and len(data) > 0:
            # 检查第一个数据点
            first_item = data[0]
            print("第一个数据点结构:")
            print(f"   类型: {type(first_item)}")
            
            if isinstance(first_item, dict):
                print("   键值:")
                for key, value in first_item.items():
                    if key == 'features':
                        print(f"     {key}: {type(value)}")
                        if isinstance(value, dict):
                            print("       特征类型:")
                            for feat_key, feat_value in value.items():
                                if hasattr(feat_value, 'shape'):
                                    print(f"         {feat_key}: {type(feat_value)} - shape: {feat_value.shape}")
                                elif hasattr(feat_value, '__len__'):
                                    print(f"         {feat_key}: {type(feat_value)} - length: {len(feat_value)}")
                                else:
                                    print(f"         {feat_key}: {type(feat_value)}")
                    elif key == 'labels':
                        print(f"     {key}: {type(value)}")
                        if isinstance(value, dict):
                            print("       标签:")
                            for label_key, label_value in value.items():
                                print(f"         {label_key}: {label_value}")
                    else:
                        print(f"     {key}: {value}")
            
            print("=" * 60)
            
            # 统计信息
            video_ids = []
            splits = []
            feature_types = set()
            
            for item in data[:10]:  # 只检查前10个
                if isinstance(item, dict):
                    video_ids.append(item.get('video_id', 'unknown'))
                    splits.append(item.get('split', 'unknown'))
                    
                    if 'features' in item and isinstance(item['features'], dict):
                        feature_types.update(item['features'].keys())
            
            print("数据统计 (前10个样本):")
            print(f"   视频ID示例: {video_ids[:5]}")
            print(f"   数据集分割: {set(splits)}")
            print(f"   特征类型: {sorted(feature_types)}")
            
        return True
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return False

def main():
    """主函数"""
    print("PKL文件内容检查工具")
    print("=" * 60)
    
    # 检查最新的pkl文件
    pkl_files = [
        'datasets/ch_sims_processed_data_cache_1985.pkl',
        'datasets/ch_sims_final_dataset.pkl',
        'datasets/ac.pkl'
    ]
    
    for pkl_file in pkl_files:
        if os.path.exists(pkl_file):
            print(f"\n检查文件: {pkl_file}")
            success = inspect_pkl_file(pkl_file)
            if success:
                break
            print("\n" + "="*80 + "\n")
        else:
            print(f"跳过不存在的文件: {pkl_file}")

if __name__ == "__main__":
    main()