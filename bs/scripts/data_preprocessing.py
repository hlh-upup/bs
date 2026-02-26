#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理优化脚本
基于专家分析结果，解决数据质量问题
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理优化器"""
    
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
        return data
    
    def analyze_data_quality(self, data):
        """分析数据质量"""
        logger.info("分析数据质量...")
        
        # 特征维度分析
        feature_dims = {
            'visual': 163,
            'audio': 768, 
            'keypoint': 1404,
            'au': 17
        }
        
        quality_report = {}
        
        for split in ['train', 'val', 'test']:
            if split in data:
                features = data[split]['features']
                labels = data[split]['labels']
                
                # 检查NaN值
                nan_report = {}
                for key, dim in feature_dims.items():
                    if key in features:
                        feature_data = features[key]
                        nan_count = np.sum(np.isnan(feature_data))
                        nan_report[key] = {
                            'nan_count': nan_count,
                            'nan_ratio': nan_count / (feature_data.size + 1e-8)
                        }
                
                # 检查标签质量
                label_report = {}
                for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']:
                    if task in labels:
                        label_data = labels[task]
                        invalid_count = np.sum(label_data == -1.0)
                        valid_count = np.sum(label_data != -1.0)
                        label_report[task] = {
                            'total': len(label_data),
                            'valid': valid_count,
                            'invalid': invalid_count,
                            'valid_ratio': valid_count / len(label_data)
                        }
                
                quality_report[split] = {
                    'nan_analysis': nan_report,
                    'label_analysis': label_report
                }
        
        return quality_report
    
    def handle_missing_values(self, data):
        """处理缺失值"""
        logger.info("处理缺失值...")
        
        for split in ['train', 'val', 'test']:
            if split in data:
                features = data[split]['features']
                
                # 为每种特征类型创建imputer
                for feature_type in features.keys():
                    feature_data = features[feature_type]
                    
                    # 处理NaN值
                    if np.any(np.isnan(feature_data)):
                        logger.info(f"{split} - {feature_type}: 发现NaN值，进行中位数插补")
                        
                        # 重塑数据为2D (samples * timesteps, features)
                        original_shape = feature_data.shape
                        reshaped_data = feature_data.reshape(-1, original_shape[-1])
                        
                        # 创建并训练imputer
                        imputer = SimpleImputer(strategy='median')
                        imputed_data = imputer.fit_transform(reshaped_data)
                        
                        # 恢复原始形状
                        features[feature_type] = imputed_data.reshape(original_shape)
                        self.imputers[f"{split}_{feature_type}"] = imputer
                        
                        logger.info(f"{split} - {feature_type}: NaN值处理完成")
    
    def normalize_features(self, data):
        """特征标准化"""
        logger.info("进行特征标准化...")
        
        # 在训练集上拟合scaler
        train_features = data['train']['features']
        
        for feature_type in train_features.keys():
            logger.info(f"标准化 {feature_type} 特征...")
            
            # 重塑数据
            feature_data = train_features[feature_type]
            original_shape = feature_data.shape
            reshaped_data = feature_data.reshape(-1, original_shape[-1])
            
            # 创建并训练scaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(reshaped_data)
            
            # 保存scaler
            self.scalers[feature_type] = scaler
            
            # 应用到所有数据集
            for split in ['train', 'val', 'test']:
                if split in data:
                    split_data = data[split]['features'][feature_type]
                    split_shape = split_data.shape
                    split_reshaped = split_data.reshape(-1, split_shape[-1])
                    split_scaled = scaler.transform(split_reshaped)
                    data[split]['features'][feature_type] = split_scaled.reshape(split_shape)
                    
            logger.info(f"{feature_type}: 标准化完成")
    
    def apply_pca(self, data):
        """应用PCA降维"""
        logger.info("应用PCA降维...")
        
        pca_config = {
            'keypoint': 50,
            'visual': 100,
            'audio': 200
        }
        
        # 在训练集上拟合PCA
        train_features = data['train']['features']
        
        for feature_type, n_components in pca_config.items():
            if feature_type in train_features:
                logger.info(f"对 {feature_type} 进行PCA降维至 {n_components} 维...")
                
                # 重塑数据
                feature_data = train_features[feature_type]
                original_shape = feature_data.shape
                reshaped_data = feature_data.reshape(-1, original_shape[-1])
                
                # 创建并训练PCA
                pca = PCA(n_components=n_components, random_state=42)
                pca.fit(reshaped_data)
                
                # 保存PCA模型
                self.pca_models[feature_type] = pca
                
                # 应用到所有数据集
                for split in ['train', 'val', 'test']:
                    if split in data:
                        split_data = data[split]['features'][feature_type]
                        split_shape = split_data.shape
                        split_reshaped = split_data.reshape(-1, split_shape[-1])
                        split_pca = pca.transform(split_reshaped)
                        
                        # 重塑回原始形状，但最后一个维度变为n_components
                        new_shape = (*split_shape[:-1], n_components)
                        data[split]['features'][feature_type] = split_pca.reshape(new_shape)
                        
                logger.info(f"{feature_type}: PCA降维完成")
    
    def handle_labels(self, data):
        """处理标签数据"""
        logger.info("处理标签数据...")
        
        for split in ['train', 'val', 'test']:
            if split in data:
                labels = data[split]['labels']
                
                # 处理-1.0的无效标签
                for task in labels.keys():
                    label_data = labels[task]
                    mask = label_data != -1.0
                    
                    # 创建有效标签掩码
                    if 'valid_masks' not in data[split]:
                        data[split]['valid_masks'] = {}
                    data[split]['valid_masks'][task] = mask
                    
                    # 计算有效标签比例
                    valid_ratio = np.mean(mask)
                    logger.info(f"{split} - {task}: 有效标签比例 {valid_ratio:.2%}")
    
    def save_processed_data(self, data, output_path):
        """保存处理后的数据"""
        logger.info(f"保存处理后的数据到: {output_path}")
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("数据预处理完成")
    
    def save_preprocessors(self, output_dir):
        """保存预处理模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存scalers
        with open(output_path / 'scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
            
        # 保存imputers
        with open(output_path / 'imputers.pkl', 'wb') as f:
            pickle.dump(self.imputers, f)
            
        # 保存PCA模型
        with open(output_path / 'pca_models.pkl', 'wb') as f:
            pickle.dump(self.pca_models, f)
            
        logger.info(f"预处理模型已保存到: {output_dir}")

def main():
    """主函数"""
    config = {
        'preprocessing': {
            'feature_scaling': True,
            'scaling_method': 'standard',
            'nan_handling': 'impute',
            'pca_components': {
                'keypoint': 50,
                'visual': 100,
                'audio': 200
            }
        }
    }
    
    preprocessor = DataPreprocessor(config)
    
    # 加载原始数据
    input_path = "datasets/ac.pkl"
    output_path = "datasets/ac_processed.pkl"
    
    logger.info("开始数据预处理...")
    data = preprocessor.load_data(input_path)
    
    # 分析数据质量
    quality_report = preprocessor.analyze_data_quality(data)
    
    # 保存质量报告
    with open("reports/data_quality_report.txt", "w") as f:
        for split, report in quality_report.items():
            f.write(f"=== {split.upper()} 数据质量报告 ===\n")
            f.write(str(report))
            f.write("\n\n")
    
    # 执行预处理
    preprocessor.handle_missing_values(data)
    preprocessor.normalize_features(data)
    preprocessor.apply_pca(data)
    preprocessor.handle_labels(data)
    
    # 保存处理后的数据
    preprocessor.save_processed_data(data, output_path)
    preprocessor.save_preprocessors("models/preprocessors/")
    
    logger.info("数据预处理完成！")
    logger.info(f"处理后的数据已保存到: {output_path}")
    logger.info("预处理模型已保存到: models/preprocessors/")

if __name__ == "__main__":
    main()