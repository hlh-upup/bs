#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频选取和实验管理系统
用于从数据集中选取代表性视频进行主观评价实验
"""

import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_config
from models import MultiTaskTalkingFaceEvaluator, ImprovedMultiTaskTalkingFaceEvaluator
from data import create_dataloaders_from_pkl
from evaluation import Evaluator

class VideoSelector:
    """视频选取器 - 基于模型评分选取代表性视频"""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'subjective_experiment'))
        self.videos_dir = self.output_dir / 'videos'
        self.selection_log = self.output_dir / 'selection_log.json'
        
    def load_model_predictions(self, model_path: str, test_loader, device) -> Dict:
        """加载模型预测结果"""
        
        print(f"📂 加载模型预测结果: {model_path}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint['model_state_dict']
        
        # 创建模型实例
        model_config = self.config['model']
        if 'improved' in model_path.lower():
            model = ImprovedMultiTaskTalkingFaceEvaluator(model_config)
        else:
            model = MultiTaskTalkingFaceEvaluator(model_config)
        
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        # 获取预测结果
        all_predictions = []
        all_targets = []
        video_indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 模型预测
                predictions, losses = model(
                    visual_features=batch['visual_features'].to(device),
                    audio_features=batch['audio_features'].to(device),
                    keypoint_features=batch.get('keypoints', torch.zeros(batch['visual_features'].size(0), 0).to(device)),
                    au_features=batch.get('au_features', torch.zeros(batch['visual_features'].size(0), 0).to(device)),
                    targets={k: v for k, v in batch.items() if k.endswith('_score')}
                )
                
                # 收集结果
                all_predictions.append(predictions)
                all_targets.append({k: v for k, v in batch.items() if k.endswith('_score')})
                
                # 假设每个batch对应不同的视频
                if 'video_indices' in batch:
                    video_indices.extend(batch['video_indices'].tolist())
                else:
                    video_indices.extend(range(batch_idx * batch['visual_features'].size(0), 
                                            (batch_idx + 1) * batch['visual_features'].size(0)))
        
        # 整理结果
        results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'video_indices': video_indices,
            'model_path': model_path
        }
        
        return results
    
    def select_videos_by_quality_stratification(self, results: Dict, n_videos: int = 20) -> Dict:
        """基于质量分层选取视频"""
        
        print("🎯 基于质量分层选取视频...")
        
        # 提取整体评分
        overall_scores = []
        for pred_dict in results['predictions']:
            if 'overall' in pred_dict:
                overall_scores.extend(pred_dict['overall'].cpu().numpy().flatten())
            else:
                # 如果没有整体评分，计算各任务平均
                task_scores = []
                for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal']:
                    if task in pred_dict:
                        task_scores.append(pred_dict[task].cpu().numpy().flatten())
                if task_scores:
                    overall_scores.extend(np.mean(task_scores, axis=0))
        
        overall_scores = np.array(overall_scores)
        
        # 按质量分层
        percentiles = np.percentile(overall_scores, [25, 75])
        low_quality_mask = overall_scores <= percentiles[0]
        high_quality_mask = overall_scores >= percentiles[1]
        medium_quality_mask = ~low_quality_mask & ~high_quality_mask
        
        # 分层选取
        n_high = n_videos // 3
        n_medium = n_videos // 3
        n_low = n_videos - n_high - n_medium
        
        selected_indices = []
        
        # 高质量视频
        high_indices = np.where(high_quality_mask)[0]
        selected_high = np.random.choice(high_indices, min(n_high, len(high_indices)), replace=False)
        selected_indices.extend(selected_high)
        
        # 中等质量视频
        medium_indices = np.where(medium_quality_mask)[0]
        selected_medium = np.random.choice(medium_indices, min(n_medium, len(medium_indices)), replace=False)
        selected_indices.extend(selected_medium)
        
        # 低质量视频
        low_indices = np.where(low_quality_mask)[0]
        selected_low = np.random.choice(low_indices, min(n_low, len(low_indices)), replace=False)
        selected_indices.extend(selected_low)
        
        selection_info = {
            'method': 'quality_stratification',
            'total_videos': len(overall_scores),
            'selected_count': len(selected_indices),
            'quality_distribution': {
                'high': len(selected_high),
                'medium': len(selected_medium),
                'low': len(selected_low)
            },
            'quality_thresholds': {
                'low_threshold': percentiles[0],
                'high_threshold': percentiles[1]
            },
            'selected_indices': selected_indices.tolist(),
            'scores': overall_scores.tolist()
        }
        
        return selection_info
    
    def select_videos_by_diversity_sampling(self, results: Dict, n_videos: int = 20) -> Dict:
        """基于多样性采样选取视频"""
        
        print("🎭 基于多样性采样选取视频...")
        
        # 提取多维度特征
        features_list = []
        for pred_dict in results['predictions']:
            features = []
            for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal']:
                if task in pred_dict:
                    features.append(pred_dict[task].cpu().numpy().flatten())
            
            if features:
                # 合并所有任务特征
                video_features = np.column_stack(features)
                features_list.append(video_features)
        
        if not features_list:
            print("⚠️ 无法提取特征，使用随机选取")
            return self.select_videos_randomly(results, n_videos)
        
        all_features = np.vstack(features_list)
        
        # 标准化特征
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(all_features)
        
        # 使用K-means聚类
        n_clusters = min(n_videos, len(all_features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # 从每个簇中选取代表性样本
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # 选择距离簇中心最近的样本
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(features_normalized[cluster_mask] - cluster_center, axis=1)
                representative_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(representative_idx)
        
        # 如果选取数量不足，随机补充
        if len(selected_indices) < n_videos:
            remaining_indices = [i for i in range(len(all_features)) if i not in selected_indices]
            additional_needed = n_videos - len(selected_indices)
            additional_indices = np.random.choice(remaining_indices, min(additional_needed, len(remaining_indices)), replace=False)
            selected_indices.extend(additional_indices)
        
        selection_info = {
            'method': 'diversity_sampling',
            'total_videos': len(all_features),
            'selected_count': len(selected_indices),
            'n_clusters': n_clusters,
            'selected_indices': selected_indices,
            'cluster_distribution': {
                str(cluster_id): int(np.sum(cluster_labels == cluster_id))
                for cluster_id in range(n_clusters)
            }
        }
        
        return selection_info
    
    def select_videos_randomly(self, results: Dict, n_videos: int = 20) -> Dict:
        """随机选取视频（基线方法）"""
        
        print("🎲 随机选取视频...")
        
        total_videos = len(results['video_indices'])
        selected_indices = np.random.choice(total_videos, min(n_videos, total_videos), replace=False)
        
        selection_info = {
            'method': 'random_sampling',
            'total_videos': total_videos,
            'selected_count': len(selected_indices),
            'selected_indices': selected_indices.tolist()
        }
        
        return selection_info
    
    def create_video_pairs(self, original_selection: Dict, improved_selection: Dict) -> Dict:
        """创建视频配对用于对比评价"""
        
        print("🔗 创建视频配对...")
        
        # 确保两个模型选取相同的视频索引
        common_indices = list(set(original_selection['selected_indices']) & 
                            set(improved_selection['selected_indices']))
        
        if not common_indices:
            print("⚠️ 没有共同视频索引，使用改进模型的选取结果")
            common_indices = improved_selection['selected_indices'][:20]
        
        # 创建配对
        video_pairs = []
        for idx in common_indices:
            pair = {
                'video_index': idx,
                'original_video_path': f'videos/original/video_{idx:04d}.mp4',
                'improved_video_path': f'videos/improved/video_{idx:04d}.mp4',
                'pair_id': len(video_pairs)
            }
            video_pairs.append(pair)
        
        # 随机化呈现顺序
        random.shuffle(video_pairs)
        
        pairing_info = {
            'total_pairs': len(video_pairs),
            'pairing_method': 'matched_pairs',
            'video_pairs': video_pairs,
            'randomization_seed': 42
        }
        
        return pairing_info
    
    def generate_selection_report(self, original_results: Dict, improved_results: Dict,
                                original_selection: Dict, improved_selection: Dict,
                                pairing_info: Dict) -> str:
        """生成视频选取报告"""
        
        report = f"""# 视频选取报告

## 📊 选取概览

### 模型性能对比
- **原始模型视频数量**: {original_selection['selected_count']}
- **优化模型视频数量**: {improved_selection['selected_count']}
- **最终配对数量**: {pairing_info['total_pairs']}

### 选取方法
- **原始模型**: {original_selection['method']}
- **优化模型**: {improved_selection['method']}
- **配对方法**: {pairing_info['pairing_method']}

## 📈 质量分布分析

### 原始模型
"""
        
        if 'quality_distribution' in original_selection:
            report += f"""
- 高质量视频: {original_selection['quality_distribution']['high']}
- 中等质量视频: {original_selection['quality_distribution']['medium']}
- 低质量视频: {original_selection['quality_distribution']['low']}
"""
        
        report += "\n### 优化模型\n"
        
        if 'quality_distribution' in improved_selection:
            report += f"""
- 高质量视频: {improved_selection['quality_distribution']['high']}
- 中等质量视频: {improved_selection['quality_distribution']['medium']}
- 低质量视频: {improved_selection['quality_distribution']['low']}
"""
        
        report += f"""
## 📋 选取的视频索引

### 原始模型选取的视频
{', '.join(map(str, original_selection['selected_indices'][:10]))}{'...' if len(original_selection['selected_indices']) > 10 else ''}

### 优化模型选取的视频
{', '.join(map(str, improved_selection['selected_indices'][:10]))}{'...' if len(improved_selection['selected_indices']) > 10 else ''}

### 配对信息
总共创建了 {pairing_info['total_pairs']} 个视频对，用于主观评价实验。

## 🎯 实验设计建议

1. **视频呈现顺序**: 已随机化，减少顺序效应
2. **盲化设计**: 使用A/B标签替代原始/优化标签
3. **平衡设计**: 每个参与者评价所有视频对
4. **质量控制**: 包含高质量、中等质量和低质量视频

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_selection_results(self, results: Dict):
        """保存选取结果"""
        
        # 保存详细结果
        results_path = self.output_dir / 'selection_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成并保存报告
        report_path = self.output_dir / 'selection_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results['report'])
        
        print(f"✅ 选取结果已保存: {results_path}")
        print(f"📄 选取报告已保存: {report_path}")
        
        return results_path, report_path

class ExperimentManager:
    """实验管理器 - 管理主观评价实验的完整流程"""
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'subjective_experiment'))
        self.experiment_data = self.output_dir / 'experiment_data.json'
        
    def create_participant_info(self, participant_id: str) -> Dict:
        """创建参与者信息"""
        
        participant_info = {
            'participant_id': participant_id,
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'trial_results': [],
            'completed_trials': 0,
            'total_trials': 20,
            'quality_check': {
                'attention_passed': False,
                'time_consistent': False,
                'rating_consistent': False
            }
        }
        
        return participant_info
    
    def create_trial_structure(self, video_pairs: List[Dict]) -> List[Dict]:
        """创建试验结构"""
        
        trials = []
        for pair_idx, pair in enumerate(video_pairs):
            trial = {
                'trial_id': pair_idx,
                'video_pair': pair,
                'presentation_order': random.choice(['AB', 'BA']),  # 随机呈现顺序
                'condition_labels': {
                    'A': random.choice(['original', 'improved']),
                    'B': 'improved' if random.choice(['original', 'improved']) == 'original' else 'original'
                },
                'time_limit': 300,  # 5分钟时间限制
                'required_ratings': [
                    'lip_sync_A', 'expression_A', 'audio_quality_A', 'visual_clarity_A', 'overall_quality_A',
                    'lip_sync_B', 'expression_B', 'audio_quality_B', 'visual_clarity_B', 'overall_quality_B',
                    'preference'
                ]
            }
            trials.append(trial)
        
        return trials
    
    def setup_experiment_database(self) -> Dict:
        """设置实验数据库"""
        
        experiment_db = {
            'experiment_id': f"subj_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            'created_time': pd.Timestamp.now().isoformat(),
            'status': 'setup',
            'participants': {},
            'trials': [],
            'config': self.config,
            'statistics': {
                'total_participants': 0,
                'completed_participants': 0,
                'average_duration': 0,
                'completion_rate': 0
            }
        }
        
        # 保存数据库
        with open(self.experiment_data, 'w', encoding='utf-8') as f:
            json.dump(experiment_db, f, ensure_ascii=False, indent=2, default=str)
        
        return experiment_db
    
    def generate_experiment_summary(self, experiment_db: Dict) -> str:
        """生成实验总结"""
        
        summary = f"""# 主观评价实验总结

## 📊 实验概览

- **实验ID**: {experiment_db['experiment_id']}
- **创建时间**: {experiment_db['created_time']}
- **实验状态**: {experiment_db['status']}

## 👥 参与者统计

- **总参与者**: {experiment_db['statistics']['total_participants']}
- **完成实验**: {experiment_db['statistics']['completed_participants']}
- **完成率**: {experiment_db['statistics']['completion_rate']:.1f}%
- **平均时长**: {experiment_db['statistics']['average_duration']:.1f}分钟

## 📈 初步结果

### 完成度分析
"""
        
        if experiment_db['statistics']['completed_participants'] > 0:
            completion_rate = (experiment_db['statistics']['completed_participants'] / 
                             experiment_db['statistics']['total_participants']) * 100
            summary += f"- 实验完成率: {completion_rate:.1f}%\n"
        
        summary += f"""
## 📋 后续步骤

1. **数据收集**: 继续招募参与者直至达到目标数量
2. **质量控制**: 检查数据质量，排除无效数据
3. **统计分析**: 进行详细的统计分析
4. **结果可视化**: 生成图表和报告

---
*总结生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary

def main():
    """主函数"""
    print("🎬 开始视频选取和实验管理...")
    
    # 加载配置
    config = load_config("config/optimized_config.yaml")
    config['output_dir'] = 'subjective_experiment'
    
    # 创建视频选取器
    selector = VideoSelector(config)
    
    # 加载数据
    print("📂 加载数据集...")
    try:
        _, _, test_loader = create_dataloaders_from_pkl("datasets/ac.pkl", config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {device}")
    except Exception as e:
        print(f"❌ 无法加载数据集: {e}")
        return
    
    # 检查模型文件
    original_model_path = "experiments_original/checkpoints/best_model.pth"
    improved_model_path = "experiments_improved/checkpoints/best_model.pth"
    
    if not os.path.exists(original_model_path):
        print(f"⚠️ 原始模型不存在: {original_model_path}")
        return
    
    if not os.path.exists(improved_model_path):
        print(f"⚠️ 优化模型不存在: {improved_model_path}")
        return
    
    try:
        # 加载模型预测结果
        original_results = selector.load_model_predictions(original_model_path, test_loader, device)
        improved_results = selector.load_model_predictions(improved_model_path, test_loader, device)
        
        # 选取视频
        original_selection = selector.select_videos_by_quality_stratification(original_results, n_videos=20)
        improved_selection = selector.select_videos_by_diversity_sampling(improved_results, n_videos=20)
        
        # 创建视频配对
        pairing_info = selector.create_video_pairs(original_selection, improved_selection)
        
        # 生成报告
        report = selector.generate_selection_report(
            original_results, improved_results, 
            original_selection, improved_selection, 
            pairing_info
        )
        
        # 保存结果
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'original_selection': original_selection,
            'improved_selection': improved_selection,
            'pairing_info': pairing_info,
            'report': report
        }
        
        selector.save_selection_results(results)
        
        # 设置实验管理
        print("🧪 设置实验管理系统...")
        manager = ExperimentManager(config)
        experiment_db = manager.setup_experiment_database()
        
        # 创建试验结构
        trials = manager.create_trial_structure(pairing_info['video_pairs'])
        experiment_db['trials'] = trials
        
        # 保存更新的数据库
        with open(manager.experiment_data, 'w', encoding='utf-8') as f:
            json.dump(experiment_db, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 实验数据库已创建: {manager.experiment_data}")
        
        # 生成实验总结
        summary = manager.generate_experiment_summary(experiment_db)
        summary_path = selector.output_dir / 'experiment_summary.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"📄 实验总结已保存: {summary_path}")
        
        print(f"\n🎉 视频选取和实验管理设置完成！")
        print(f"📁 所有文件已保存到: {selector.output_dir}")
        print(f"\n📋 下一步操作:")
        print(f"1. 准备对应的视频文件")
        print(f"2. 部署评价界面到Web服务器")
        print(f"3. 开始招募参与者")
        print(f"4. 监控实验进展和数据质量")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()