#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理优化脚本 - 修正版
基于实际数据结构进行优化
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pickle
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessorFixed:
    """修正版数据预处理优化器"""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.pca_models = {}
        
    def load_data(self, file_path):
        """加载数据集"""
        logger.info(f"正在加载数据: {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 检查数据结构
        logger.info(f"数据类型: {type(data)}")
        logger.info(f"数据键: {list(data.keys())}")
        
        for split in ['train', 'val', 'test']:
            if split in data:
                split_data = data[split]
                logger.info(f"{split}数据类型: {type(split_data)}")
                if isinstance(split_data, list):
                    logger.info(f"{split}数据长度: {len(split_data)}")
                    if len(split_data) > 0:
                        logger.info(f"{split}第一个元素类型: {type(split_data[0])}")
                        if hasattr(split_data[0], 'keys'):
                            logger.info(f"{split}第一个元素键: {list(split_data[0].keys())}")
        
        return data
    
    def analyze_list_structure(self, data_list):
        """分析列表结构的数据"""
        if not data_list:
            return None
            
        # 检查第一个样本
        sample = data_list[0]
        
        if isinstance(sample, dict):
            # 如果是字典，提取特征
            keys = list(sample.keys())
            logger.info(f"样本键: {keys}")
            
            # 尝试识别特征和标签
            features = {}
            labels = {}
            
            for key in keys:
                value = sample[key]
                if isinstance(value, np.ndarray):
                    logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
                    
                    # 根据键名判断特征类型
                    if 'feature' in str(key).lower() or 'visual' in str(key).lower():
                        features[key] = len(value) if value.ndim == 1 else value.shape[-1]
                    elif 'label' in str(key).lower() or 'score' in str(key).lower():
                        labels[key] = value
                    else:
                        # 默认根据维度判断
                        if value.ndim >= 2 and value.shape[-1] > 10:
                            features[key] = value.shape[-1]
                        else:
                            labels[key] = value
                else:
                    logger.info(f"{key}: {type(value)} - {value}")
            
            return features, labels
        
        return None, None
    
    def convert_list_to_dict(self, data_list):
        """将列表格式转换为字典格式"""
        if not data_list:
            return None
            
        # 初始化收集器
        all_features = {}
        all_labels = {}
        
        # 分析第一个样本确定结构
        sample = data_list[0]
        
        if isinstance(sample, dict):
            # 情况1: 列表中的每个元素是字典
            keys = list(sample.keys())
            logger.info(f"检测到字典结构，键: {keys}")
            
            # 为每个键收集数据
            for key in keys:
                values = [item[key] for item in data_list]
                
                # 检查是否为数值类型
                if isinstance(values[0], (np.ndarray, list, int, float)):
                    # 转换为numpy数组
                    try:
                        array_data = np.array(values)
                        logger.info(f"{key}: 转换为数组，形状 {array_data.shape}")
                        
                        # 根据形状判断是特征还是标签
                        if array_data.ndim >= 2 and array_data.shape[-1] > 10:
                            all_features[key] = array_data
                        else:
                            all_labels[key] = array_data
                    except:
                        logger.warning(f"{key}: 无法转换为数组")
                else:
                    logger.info(f"{key}: 非数值类型，跳过")
        
        elif isinstance(sample, (tuple, list)):
            # 情况2: 列表中的每个元素是元组或列表
            logger.info(f"检测到元组结构，长度: {len(sample)}")
            
            # 假设第一个元素是特征，第二个是标签
            if len(sample) >= 2:
                features_list = [item[0] for item in data_list]
                labels_list = [item[1] for item in data_list]
                
                # 转换为数组
                try:
                    features_array = np.array(features_list)
                    labels_array = np.array(labels_list)
                    
                    logger.info(f"特征形状: {features_array.shape}")
                    logger.info(f"标签形状: {labels_array.shape}")
                    
                    all_features['combined'] = features_array
                    all_labels['combined'] = labels_array
                    
                except Exception as e:
                    logger.error(f"转换失败: {e}")
        
        return all_features, all_labels
    
    def process_data(self, data):
        """处理数据"""
        logger.info("开始处理数据...")
        
        processed_data = {}
        
        for split in ['train', 'val', 'test']:
            if split in data:
                logger.info(f"处理 {split} 数据...")
                split_data = data[split]
                
                if isinstance(split_data, list):
                    features, labels = self.convert_list_to_dict(split_data)
                    
                    processed_data[split] = {
                        'features': features,
                        'labels': labels
                    }
                    
                    # 打印统计信息
                    if features:
                        logger.info(f"{split} - 特征维度: {list(features.keys())}")
                        for key, value in features.items():
                            logger.info(f"  {key}: {value.shape}")
                    
                    if labels:
                        logger.info(f"{split} - 标签维度: {list(labels.keys())}")
                        for key, value in labels.items():
                            logger.info(f"  {key}: {value.shape}")
        
        return processed_data
    
    def save_processed_data(self, data, output_path):
        """保存处理后的数据"""
        logger.info(f"保存处理后的数据到: {output_path}")
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("数据预处理完成")
    
    def create_sample_analysis(self, data, output_path):
        """创建样本分析"""
        analysis = []
        analysis.append("# 数据结构分析报告\n\n")
        
        for split in ['train', 'val', 'test']:
            if split in data:
                split_data = data[split]
                analysis.append(f"## {split.upper()} 数据集\n\n")
                
                if isinstance(split_data, list):
                    analysis.append(f"- **样本数量**: {len(split_data)}\n")
                    
                    if len(split_data) > 0:
                        sample = split_data[0]
                        analysis.append(f"- **样本类型**: {type(sample)}\n")
                        
                        if isinstance(sample, dict):
                            for key, value in sample.items():
                                if hasattr(value, 'shape'):
                                    analysis.append(f"- **{key}**: 形状 {value.shape}\n")
                                else:
                                    analysis.append(f"- **{key}**: {type(value)} - {value}\n")
                        elif isinstance(sample, (tuple, list)):
                            analysis.append(f"- **样本长度**: {len(sample)}\n")
                            for i, item in enumerate(sample):
                                if hasattr(item, 'shape'):
                                    analysis.append(f"  - **元素{i}**: 形状 {item.shape}\n")
                                else:
                                    analysis.append(f"  - **元素{i}**: {type(item)}\n")
                
                analysis.append("\n")
        
        with open(output_path, 'w') as f:
            f.writelines(analysis)
        
        logger.info(f"分析报告已保存: {output_path}")

def main():
    """主函数"""
    logger.info("开始数据预处理分析...")
    
    preprocessor = DataPreprocessorFixed({})
    
    # 加载原始数据
    input_path = "datasets/ac.pkl"
    output_path = "datasets/ac_processed.pkl"
    analysis_path = "reports/data_structure_analysis.md"
    
    data = preprocessor.load_data(input_path)
    
    # 创建数据结构分析
    preprocessor.create_sample_analysis(data, analysis_path)
    
    # 处理数据
    processed_data = preprocessor.process_data(data)
    
    # 保存处理后的数据
    preprocessor.save_processed_data(processed_data, output_path)
    
    logger.info("数据预处理分析完成！")

if __name__ == "__main__":
    main()