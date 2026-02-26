#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价数据集质量分析

详细分析EmotionTalk数据集的数据质量、特征分布、异常值等问题
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataQualityAnalyzer:
    def __init__(self, dataset_path):
        """
        初始化数据质量分析器
        
        Args:
            dataset_path: 数据集文件路径
        """
        self.dataset_path = dataset_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.feature_dims = {
            'visual': 163,
            'audio': 768,
            'keypoint': 1404,
            'au': 17
        }
        
    def load_data(self):
        """加载数据集"""
        print("正在加载数据集...")
        try:
            with open(self.dataset_path, 'rb') as f:
                self.data = pickle.load(f)
            
            # 处理不同的数据格式
            if isinstance(self.data, dict):
                # 标准格式：包含train/val/test键的字典
                self.train_data = self.data.get('train', [])
                self.val_data = self.data.get('val', [])
                self.test_data = self.data.get('test', [])
            elif isinstance(self.data, list):
                # 缓存格式：包含所有样本的列表，每个样本有split字段
                self.train_data = [item for item in self.data if item.get('split') == 'train']
                self.val_data = [item for item in self.data if item.get('split') == 'valid']  # 注意是'valid'不是'val'
                self.test_data = [item for item in self.data if item.get('split') == 'test']
            else:
                raise ValueError(f"不支持的數據格式: {type(self.data)}")
            
            print(f"数据加载成功！")
            print(f"训练集: {len(self.train_data)} 样本")
            print(f"验证集: {len(self.val_data)} 样本")
            print(f"测试集: {len(self.test_data)} 样本")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def analyze_label_distribution(self):
        """分析标签分布"""
        print("\n=== 标签分布分析 ===")
        
        # 收集所有标签数据
        all_labels = []
        for split_data, split_name in [(self.train_data, 'train'), 
                                      (self.val_data, 'val'), 
                                      (self.test_data, 'test')]:
            if split_data:
                for sample in split_data:
                    if 'labels' in sample:
                        labels = sample['labels'].copy()
                        labels['split'] = split_name
                        all_labels.append(labels)
        
        if not all_labels:
            print("未找到标签数据")
            return
        
        labels_df = pd.DataFrame(all_labels)
        
        # 标签统计
        score_columns = ['lip_sync_score', 'expression_score', 'audio_quality_score', 
                        'cross_modal_score', 'overall_score']
        
        print(f"总样本数: {len(labels_df)}")
        
        for col in score_columns:
            if col in labels_df.columns:
                valid_mask = labels_df[col] != -1.0
                valid_count = valid_mask.sum()
                invalid_count = (~valid_mask).sum()
                
                print(f"\n{col}:")
                print(f"  有效样本: {valid_count} ({valid_count/len(labels_df)*100:.1f}%)")
                print(f"  无效样本(-1.0): {invalid_count} ({invalid_count/len(labels_df)*100:.1f}%)")
                
                if valid_count > 0:
                    valid_values = labels_df[col][valid_mask]
                    print(f"  均值: {valid_values.mean():.3f}")
                    print(f"  标准差: {valid_values.std():.3f}")
                    print(f"  最小值: {valid_values.min():.3f}")
                    print(f"  最大值: {valid_values.max():.3f}")
                    print(f"  中位数: {valid_values.median():.3f}")
        
        return labels_df
    
    def analyze_feature_quality(self):
        """分析特征质量"""
        print("\n=== 特征质量分析 ===")
        
        feature_stats = {}
        
        # 分析每个特征类型
        for feature_type in ['visual', 'audio', 'keypoint', 'au']:
            print(f"\n--- {feature_type.upper()} 特征分析 ---")
            
            all_features = []
            nan_count = 0
            inf_count = 0
            zero_var_count = 0
            
            # 收集特征数据
            for sample in self.train_data[:100]:  # 限制样本数以节省内存
                if 'features' in sample and feature_type in sample['features']:
                    features = np.array(sample['features'][feature_type])
                    
                    # 检查NaN和Inf
                    nan_count += np.isnan(features).sum()
                    inf_count += np.isinf(features).sum()
                    
                    all_features.append(features)
            
            if not all_features:
                print(f"  未找到{feature_type}特征数据")
                continue
            
            # 堆叠特征数组
            try:
                features_array = np.stack(all_features)
                print(f"  特征形状: {features_array.shape}")
                print(f"  NaN值数量: {nan_count}")
                print(f"  Inf值数量: {inf_count}")
                
                # 计算基本统计信息
                flat_features = features_array.reshape(-1, features_array.shape[-1])
                
                # 检查零方差特征
                feature_vars = np.var(flat_features, axis=0)
                zero_var_features = np.sum(feature_vars == 0)
                print(f"  零方差特征数: {zero_var_features}/{len(feature_vars)}")
                
                # 计算统计信息
                feature_stats[feature_type] = {
                    'mean': np.mean(flat_features, axis=0),
                    'std': np.std(flat_features, axis=0),
                    'min': np.min(flat_features, axis=0),
                    'max': np.max(flat_features, axis=0),
                    'median': np.median(flat_features, axis=0),
                    'q25': np.percentile(flat_features, 25, axis=0),
                    'q75': np.percentile(flat_features, 75, axis=0)
                }
                
                # 检查异常值（使用IQR方法）
                q1 = np.percentile(flat_features, 25, axis=0)
                q3 = np.percentile(flat_features, 75, axis=0)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_count = 0
                for i in range(flat_features.shape[1]):
                    outliers = ((flat_features[:, i] < lower_bound[i]) | 
                               (flat_features[:, i] > upper_bound[i])).sum()
                    outliers_count += outliers
                
                print(f"  异常值数量: {outliers_count} (基于IQR方法)")
                
                # 特征值范围
                print(f"  特征值范围: [{np.min(flat_features):.3f}, {np.max(flat_features):.3f}]")
                
            except Exception as e:
                print(f"  特征分析出错: {e}")
        
        return feature_stats
    
    def check_feature_scaling_issues(self):
        """检查特征缩放问题"""
        print("\n=== 特征缩放问题分析 ===")
        
        scaling_issues = {}
        
        for feature_type in ['visual', 'audio', 'keypoint', 'au']:
            print(f"\n--- {feature_type.upper()} 特征缩放分析 ---")
            
            sample_features = []
            for sample in self.train_data[:50]:  # 限制样本数
                if 'features' in sample and feature_type in sample['features']:
                    features = np.array(sample['features'][feature_type])
                    # 取时间维度的平均值来简化分析
                    if len(features.shape) == 2:
                        features = np.mean(features, axis=0)
                    sample_features.append(features)
            
            if not sample_features:
                continue
            
            try:
                features_matrix = np.stack(sample_features)
                
                # 计算每个特征维度的统计信息
                feature_means = np.mean(features_matrix, axis=0)
                feature_stds = np.std(features_matrix, axis=0)
                feature_ranges = np.max(features_matrix, axis=0) - np.min(features_matrix, axis=0)
                
                print(f"  特征维度: {features_matrix.shape[1]}")
                print(f"  均值范围: [{np.min(feature_means):.3f}, {np.max(feature_means):.3f}]")
                print(f"  标准差范围: [{np.min(feature_stds):.3f}, {np.max(feature_stds):.3f}]")
                print(f"  值域范围: [{np.min(feature_ranges):.3f}, {np.max(feature_ranges):.3f}]")
                
                # 检查是否存在明显的缩放问题
                mean_range_ratio = np.max(np.abs(feature_means)) / (np.min(feature_means) + 1e-8)
                std_range_ratio = np.max(feature_stds) / (np.min(feature_stds) + 1e-8)
                range_ratio = np.max(feature_ranges) / (np.min(feature_ranges) + 1e-8)
                
                print(f"  均值范围比例: {mean_range_ratio:.1f}")
                print(f"  标准差范围比例: {std_range_ratio:.1f}")
                print(f"  值域范围比例: {range_ratio:.1f}")
                
                # 判断是否需要缩放
                needs_scaling = (mean_range_ratio > 100 or std_range_ratio > 100 or range_ratio > 1000)
                scaling_issues[feature_type] = {
                    'needs_scaling': needs_scaling,
                    'mean_range_ratio': mean_range_ratio,
                    'std_range_ratio': std_range_ratio,
                    'range_ratio': range_ratio
                }
                
                if needs_scaling:
                    print(f"  ⚠️  建议进行特征缩放！")
                else:
                    print(f"  ✅  特征缩放看起来良好")
                    
            except Exception as e:
                print(f"  缩放分析出错: {e}")
        
        return scaling_issues
    
    def analyze_temporal_consistency(self):
        """分析时间序列一致性"""
        print("\n=== 时间序列一致性分析 ===")
        
        temporal_stats = {}
        
        for feature_type in ['visual', 'audio', 'keypoint', 'au']:
            print(f"\n--- {feature_type.upper()} 时间一致性分析 ---")
            
            temporal_variations = []
            
            for sample in self.train_data[:20]:  # 限制样本数以节省计算
                if 'features' in sample and feature_type in sample['features']:
                    features = np.array(sample['features'][feature_type])
                    
                    if len(features.shape) == 2 and features.shape[0] > 1:  # 确保是时间序列
                        # 计算时间维度上的变化
                        temporal_diff = np.diff(features, axis=0)
                        temporal_variations.extend(np.std(temporal_diff, axis=0))
            
            if temporal_variations:
                temporal_variations = np.array(temporal_variations)
                print(f"  时间变化标准差 - 均值: {np.mean(temporal_variations):.3f}")
                print(f"  时间变化标准差 - 标准差: {np.std(temporal_variations):.3f}")
                
                # 检查是否存在时间上的突变（可能的异常）
                high_variation_ratio = np.sum(temporal_variations > np.percentile(temporal_variations, 95)) / len(temporal_variations)
                print(f"  高变化特征比例 (>95百分位): {high_variation_ratio:.3f}")
                
                temporal_stats[feature_type] = {
                    'mean_temporal_variation': np.mean(temporal_variations),
                    'std_temporal_variation': np.std(temporal_variations),
                    'high_variation_ratio': high_variation_ratio
                }
            else:
                print(f"  未找到足够的时间序列数据")
        
        return temporal_stats
    
    def generate_preprocessing_recommendations(self, analysis_results):
        """生成预处理建议"""
        print("\n=== 预处理建议 ===")
        
        recommendations = []
        
        # 基于标签分析的建议
        if 'labels_df' in analysis_results:
            labels_df = analysis_results['labels_df']
            score_columns = ['lip_sync_score', 'expression_score', 'audio_quality_score', 
                           'cross_modal_score', 'overall_score']
            
            for col in score_columns:
                if col in labels_df.columns:
                    invalid_ratio = (labels_df[col] == -1.0).mean()
                    if invalid_ratio > 0.1:  # 超过10%的无效标签
                        recommendations.append({
                            'type': '标签处理',
                            'issue': f'{col}有{invalid_ratio*100:.1f}%的无效标签',
                            'recommendation': f'考虑移除无效标签样本或使用插值方法填充'
                        })
        
        # 基于特征质量的建议
        if 'scaling_issues' in analysis_results:
            scaling_issues = analysis_results['scaling_issues']
            for feature_type, issues in scaling_issues.items():
                if issues['needs_scaling']:
                    recommendations.append({
                        'type': '特征缩放',
                        'issue': f'{feature_type}特征存在严重的缩放问题',
                        'recommendation': f'使用StandardScaler或MinMaxScaler进行特征标准化'
                    })
        
        # 基于NaN和Inf检查的建议
        # 这里可以添加更多的检查逻辑
        
        # 通用建议
        recommendations.extend([
            {
                'type': '特征选择',
                'issue': '特征维度差异很大(17-1404维)',
                'recommendation': '考虑使用PCA或特征选择方法降低高维特征维度'
            },
            {
                'type': '数据增强',
                'issue': '模型表现不佳(R²=0.174)',
                'recommendation': '考虑时间序列数据增强技术，如时间扭曲、噪声添加等'
            },
            {
                'type': '特征融合',
                'issue': 'Cross Modal任务表现特别差',
                'recommendation': '考虑使用注意力机制或专门的跨模态融合策略'
            }
        ])
        
        # 打印建议
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. 【{rec['type']}】{rec['issue']}")
            print(f"   建议: {rec['recommendation']}")
            print()
        
        return recommendations
    
    def run_complete_analysis(self):
        """运行完整的数据质量分析"""
        print("=" * 60)
        print("AI生成说话人脸视频评价数据集质量分析报告")
        print("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 运行各项分析
        analysis_results = {}
        
        # 1. 标签分布分析
        labels_df = self.analyze_label_distribution()
        analysis_results['labels_df'] = labels_df
        
        # 2. 特征质量分析
        feature_stats = self.analyze_feature_quality()
        analysis_results['feature_stats'] = feature_stats
        
        # 3. 特征缩放问题分析
        scaling_issues = self.check_feature_scaling_issues()
        analysis_results['scaling_issues'] = scaling_issues
        
        # 4. 时间序列一致性分析
        temporal_stats = self.analyze_temporal_consistency()
        analysis_results['temporal_stats'] = temporal_stats
        
        # 5. 生成预处理建议
        recommendations = self.generate_preprocessing_recommendations(analysis_results)
        analysis_results['recommendations'] = recommendations
        
        print("\n" + "=" * 60)
        print("数据质量分析完成！")
        print("=" * 60)
        
        return analysis_results

def main():
    """主函数"""
    # 数据集路径
    dataset_path = "f:/bs/datasets/ch_sims_final_dataset.pkl"
    
    if not os.path.exists(dataset_path):
        print(f"数据集文件不存在: {dataset_path}")
        # 尝试寻找其他可能的数据集文件
        possible_files = [
            "f:/bs/datasets/ch_sims_processed_data_cache_1985.pkl",
            "f:/bs/datasets/ac.pkl"
        ]
        for file_path in possible_files:
            if os.path.exists(file_path):
                dataset_path = file_path
                print(f"使用替代数据集: {dataset_path}")
                break
        else:
            print("未找到可用的数据集文件")
            return
    
    # 创建分析器并运行分析
    analyzer = DataQualityAnalyzer(dataset_path)
    results = analyzer.run_complete_analysis()
    
    # 保存分析结果
    output_path = "f:/bs/data_quality_analysis_results.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n分析结果已保存至: {output_path}")

if __name__ == "__main__":
    main()