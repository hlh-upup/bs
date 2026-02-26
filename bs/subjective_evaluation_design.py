#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主观评价实验管理系统
用于对比原始模型和优化模型的主观评价效果
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubjectiveEvaluationDesign:
    """主观评价实验设计"""
    
    def __init__(self, config: dict):
        self.config = config
        self.experiment_dir = Path(config.get('experiment_dir', 'subjective_experiment'))
        self.videos_dir = self.experiment_dir / 'videos'
        self.results_dir = self.experiment_dir / 'results'
        self.setup_directories()
        
    def setup_directories(self):
        """创建实验目录结构"""
        for dir_path in [self.experiment_dir, self.videos_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
            
        # 创建子目录
        (self.videos_dir / 'original').mkdir(exist_ok=True)
        (self.videos_dir / 'improved').mkdir(exist_ok=True)
        (self.videos_dir / 'ground_truth').mkdir(exist_ok=True)
        
    def design_experiment(self) -> Dict:
        """设计实验方案"""
        
        # 实验基本信息
        experiment_design = {
            'experiment_name': f"subjective_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'description': 'AI生成说话人脸视频质量主观评价实验',
            'version': '1.0',
            'created_date': datetime.now().isoformat(),
            
            # 实验设计
            'experimental_design': {
                'type': 'paired_comparison',  # 配对比较
                'design': 'within_subjects',  # 被试内设计
                'counterbalancing': True,     # 平衡设计
            },
            
            # 评价维度
            'evaluation_dimensions': [
                {
                    'name': 'lip_sync',
                    'description': '唇音同步质量',
                    'scale': '1-5分',
                    'anchors': {
                        1: '完全不同步',
                        3: '基本同步',
                        5: '完美同步'
                    }
                },
                {
                    'name': 'expression_naturalness',
                    'description': '表情自然度',
                    'scale': '1-5分',
                    'anchors': {
                        1: '非常不自然',
                        3: '比较自然',
                        5: '非常自然'
                    }
                },
                {
                    'name': 'audio_quality',
                    'description': '音频质量',
                    'scale': '1-5分',
                    'anchors': {
                        1: '质量很差',
                        3: '质量一般',
                        5: '质量很好'
                    }
                },
                {
                    'name': 'visual_clarity',
                    'description': '视觉清晰度',
                    'scale': '1-5分',
                    'anchors': {
                        1: '模糊不清',
                        3: '基本清晰',
                        5: '非常清晰'
                    }
                },
                {
                    'name': 'overall_quality',
                    'description': '整体质量',
                    'scale': '1-5分',
                    'anchors': {
                        1: '质量很差',
                        3: '质量一般',
                        5: '质量很好'
                    }
                }
            ],
            
            # 被试信息
            'participants': {
                'target_count': 30,
                'demographics': {
                    'age_range': '18-45',
                    'gender': 'balanced',
                    'background': '包含专业和非专业背景'
                },
                'screening': {
                    'vision': '正常或矫正正常',
                    'hearing': '正常',
                    'experience': '有无AI视频观看经验均可'
                }
            },
            
            # 实验流程
            'procedure': {
                'duration_per_participant': '20-30分钟',
                'training_samples': 3,
                'practice_trials': 5,
                'main_trials': 20,
                'break_intervals': '每10个评价后休息'
            },
            
            # 视频选取策略
            'video_selection_strategy': {
                'method': 'stratified_sampling',
                'criteria': [
                    'quality_variation',
                    'content_diversity',
                    'speaker_diversity',
                    'emotional_variation'
                ],
                'sample_size': 20,
                'distribution': {
                    'high_quality': 7,
                    'medium_quality': 6,
                    'low_quality': 7
                }
            }
        }
        
        return experiment_design
    
    def generate_video_selection_plan(self, dataset_info: Dict) -> Dict:
        """生成视频选取方案"""
        
        selection_plan = {
            'total_videos_needed': 20,
            'selection_criteria': {
                'quality_based': {
                    'high_quality': {
                        'count': 7,
                        'criteria': '模型评分前25%'
                    },
                    'medium_quality': {
                        'count': 6,
                        'criteria': '模型评分25%-75%'
                    },
                    'low_quality': {
                        'count': 7,
                        'criteria': '模型评分后25%'
                    }
                },
                'content_based': {
                    'speaker_variation': '至少包含5个不同说话人',
                    'emotion_variation': '包含多种情感状态',
                    'duration_variation': '视频长度在3-10秒之间',
                    'background_variation': '不同背景环境'
                }
            },
            
            # 配对设计
            'pairing_design': {
                'method': 'balanced_pairing',
                'each_video_shown': 2,  # 每个视频显示2次（原始vs改进）
                'presentation_order': 'randomized',
                'condition_labels': ['A', 'B'],  # 盲化标签
            },
            
            # 实验条件
            'experimental_conditions': [
                {
                    'name': 'original_model',
                    'label': 'A',
                    'description': '原始模型生成'
                },
                {
                    'name': 'improved_model', 
                    'label': 'B',
                    'description': '优化模型生成'
                }
            ]
        }
        
        return selection_plan
    
    def create_evaluation_interface(self) -> str:
        """创建评价界面HTML"""
        
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>主观评价实验</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        .video-section { margin: 20px 0; }
        .video-container { display: flex; gap: 20px; margin: 20px 0; }
        .video-player { flex: 1; text-align: center; }
        .video-player video { width: 100%; max-width: 500px; border: 2px solid #ddd; border-radius: 8px; }
        .evaluation-form { margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 8px; }
        .scale-container { margin: 15px 0; }
        .scale-labels { display: flex; justify-content: space-between; margin: 10px 0; font-size: 14px; }
        .radio-group { display: flex; justify-content: space-between; margin: 10px 0; }
        .radio-group label { display: flex; align-items: center; gap: 5px; }
        .submit-btn { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .submit-btn:hover { background: #0056b3; }
        .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; margin: 20px 0; }
        .progress-fill { height: 100%; background: #28a745; border-radius: 10px; transition: width 0.3s; }
        .instructions { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI生成说话人脸视频质量评价</h1>
        
        <div class="instructions">
            <h3>评价说明：</h3>
            <ul>
                <li>您将看到一对AI生成的说话人脸视频</li>
                <li>请仔细观察并比较两个视频的质量差异</li>
                <li>按照以下维度对每个视频进行独立评价</li>
                <li>您的真实反馈对我们非常重要</li>
            </ul>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
        </div>
        
        <div class="video-section">
            <h3>视频对评价</h3>
            <div class="video-container">
                <div class="video-player">
                    <h4>视频 A</h4>
                    <video controls id="videoA">
                        <source src="" type="video/mp4">
                    </video>
                </div>
                <div class="video-player">
                    <h4>视频 B</h4>
                    <video controls id="videoB">
                        <source src="" type="video/mp4">
                    </video>
                </div>
            </div>
        </div>
        
        <div class="evaluation-form">
            <h3>请对视频A进行评价</h3>
            
            <div class="scale-container">
                <label><strong>唇音同步质量</strong></label>
                <p>视频中嘴唇动作与语音的同步程度</p>
                <div class="radio-group">
                    <label><input type="radio" name="lip_sync_A" value="1"> 1分</label>
                    <label><input type="radio" name="lip_sync_A" value="2"> 2分</label>
                    <label><input type="radio" name="lip_sync_A" value="3"> 3分</label>
                    <label><input type="radio" name="lip_sync_A" value="4"> 4分</label>
                    <label><input type="radio" name="lip_sync_A" value="5"> 5分</label>
                </div>
                <div class="scale-labels">
                    <span>完全不同步</span>
                    <span>基本同步</span>
                    <span>完美同步</span>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>表情自然度</strong></label>
                <p>面部表情的自然和真实程度</p>
                <div class="radio-group">
                    <label><input type="radio" name="expression_A" value="1"> 1分</label>
                    <label><input type="radio" name="expression_A" value="2"> 2分</label>
                    <label><input type="radio" name="expression_A" value="3"> 3分</label>
                    <label><input type="radio" name="expression_A" value="4"> 4分</label>
                    <label><input type="radio" name="expression_A" value="5"> 5分</label>
                </div>
                <div class="scale-labels">
                    <span>非常不自然</span>
                    <span>比较自然</span>
                    <span>非常自然</span>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>音频质量</strong></label>
                <p>音频的清晰度和质量</p>
                <div class="radio-group">
                    <label><input type="radio" name="audio_A" value="1"> 1分</label>
                    <label><input type="radio" name="audio_A" value="2"> 2分</label>
                    <label><input type="radio" name="audio_A" value="3"> 3分</label>
                    <label><input type="radio" name="audio_A" value="4"> 4分</label>
                    <label><input type="radio" name="audio_A" value="5"> 5分</label>
                </div>
                <div class="scale-labels">
                    <span>质量很差</span>
                    <span>质量一般</span>
                    <span>质量很好</span>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>视觉清晰度</strong></label>
                <p>视频的清晰度和视觉质量</p>
                <div class="radio-group">
                    <label><input type="radio" name="visual_A" value="1"> 1分</label>
                    <label><input type="radio" name="visual_A" value="2"> 2分</label>
                    <label><input type="radio" name="visual_A" value="3"> 3分</label>
                    <label><input type="radio" name="visual_A" value="4"> 4分</label>
                    <label><input type="radio" name="visual_A" value="5"> 5分</label>
                </div>
                <div class="scale-labels">
                    <span>模糊不清</span>
                    <span>基本清晰</span>
                    <span>非常清晰</span>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>整体质量</strong></label>
                <p>对视频的整体印象</p>
                <div class="radio-group">
                    <label><input type="radio" name="overall_A" value="1"> 1分</label>
                    <label><input type="radio" name="overall_A" value="2"> 2分</label>
                    <label><input type="radio" name="overall_A" value="3"> 3分</label>
                    <label><input type="radio" name="overall_A" value="4"> 4分</label>
                    <label><input type="radio" name="overall_A" value="5"> 5分</label>
                </div>
                <div class="scale-labels">
                    <span>质量很差</span>
                    <span>质量一般</span>
                    <span>质量很好</span>
                </div>
            </div>
        </div>
        
        <div class="evaluation-form">
            <h3>请对视频B进行评价</h3>
            <!-- 类似的评价表单，只改变name属性 -->
            <div class="scale-container">
                <label><strong>唇音同步质量</strong></label>
                <div class="radio-group">
                    <label><input type="radio" name="lip_sync_B" value="1"> 1分</label>
                    <label><input type="radio" name="lip_sync_B" value="2"> 2分</label>
                    <label><input type="radio" name="lip_sync_B" value="3"> 3分</label>
                    <label><input type="radio" name="lip_sync_B" value="4"> 4分</label>
                    <label><input type="radio" name="lip_sync_B" value="5"> 5分</label>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>表情自然度</strong></label>
                <div class="radio-group">
                    <label><input type="radio" name="expression_B" value="1"> 1分</label>
                    <label><input type="radio" name="expression_B" value="2"> 2分</label>
                                       <label><input type="radio" name="expression_B" value="3"> 3分</label>
                    <label><input type="radio" name="expression_B" value="4"> 4分</label>
                    <label><input type="radio" name="expression_B" value="5"> 5分</label>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>音频质量</strong></label>
                <div class="radio-group">
                    <label><input type="radio" name="audio_B" value="1"> 1分</label>
                    <label><input type="radio" name="audio_B" value="2"> 2分</label>
                    <label><input type="radio" name="audio_B" value="3"> 3分</label>
                    <label><input type="radio" name="audio_B" value="4"> 4分</label>
                    <label><input type="radio" name="audio_B" value="5"> 5分</label>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>视觉清晰度</strong></label>
                <div class="radio-group">
                    <label><input type="radio" name="visual_B" value="1"> 1分</label>
                    <label><input type="radio" name="visual_B" value="2"> 2分</label>
                    <label><input type="radio" name="visual_B" value="3"> 3分</label>
                    <label><input type="radio" name="visual_B" value="4"> 4分</label>
                    <label><input type="radio" name="visual_B" value="5"> 5分</label>
                </div>
            </div>
            
            <div class="scale-container">
                <label><strong>整体质量</strong></label>
                <div class="radio-group">
                    <label><input type="radio" name="overall_B" value="1"> 1分</label>
                    <label><input type="radio" name="overall_B" value="2"> 2分</label>
                    <label><input type="radio" name="overall_B" value="3"> 3分</label>
                    <label><input type="radio" name="overall_B" value="4"> 4分</label>
                    <label><input type="radio" name="overall_B" value="5"> 5分</label>
                </div>
            </div>
        </div>
        
        <div class="evaluation-form">
            <h3>偏好比较</h3>
            <p>在两个视频中，您更偏好哪一个？</p>
            <div class="radio-group">
                <label><input type="radio" name="preference" value="A"> 更偏好视频A</label>
                <label><input type="radio" name="preference" value="B"> 更偏好视频B</label>
                <label><input type="radio" name="preference" value="equal"> 两者无明显差异</label>
            </div>
            
            <div class="scale-container">
                <label><strong>评论 (可选)</strong></label>
                <textarea name="comments" rows="4" style="width: 100%; margin-top: 10px;" 
                         placeholder="请描述您选择的原因或任何其他观察..."></textarea>
            </div>
        </div>
        
        <button type="button" class="submit-btn" onclick="submitEvaluation()">提交评价</button>
    </div>
    
    <script>
        let currentTrial = 0;
        const totalTrials = 20;
        
        function updateProgress() {
            const progress = (currentTrial / totalTrials) * 100;
            document.querySelector('.progress-fill').style.width = progress + '%';
        }
        
        function submitEvaluation() {
            // 收集评价数据
            const formData = new FormData();
            const ratings = {};
            
            // 收集A视频评分
            ['lip_sync_A', 'expression_A', 'audio_A', 'visual_A', 'overall_A'].forEach(name => {
                const value = document.querySelector(`input[name="${name}"]:checked`);
                ratings[name] = value ? value.value : null;
            });
            
            // 收集B视频评分
            ['lip_sync_B', 'expression_B', 'audio_B', 'visual_B', 'overall_B'].forEach(name => {
                const value = document.querySelector(`input[name="${name}"]:checked`);
                ratings[name] = value ? value.value : null;
            });
            
            // 收集偏好
            const preference = document.querySelector('input[name="preference"]:checked');
            ratings['preference'] = preference ? preference.value : null;
            
            // 收集评论
            const comments = document.querySelector('textarea[name="comments"]').value;
            ratings['comments'] = comments;
            
            // 验证数据完整性
            const requiredFields = ['lip_sync_A', 'expression_A', 'audio_A', 'visual_A', 'overall_A',
                                 'lip_sync_B', 'expression_B', 'audio_B', 'visual_B', 'overall_B', 'preference'];
            
            const missingFields = requiredFields.filter(field => !ratings[field]);
            if (missingFields.length > 0) {
                alert('请完成所有必填项评分！');
                return;
            }
            
            // 发送数据到服务器
            fetch('/submit_evaluation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trial_id: currentTrial,
                    participant_id: getParticipantId(),
                    ratings: ratings,
                    timestamp: new Date().toISOString()
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentTrial++;
                    updateProgress();
                    if (currentTrial >= totalTrials) {
                        alert('实验完成！感谢您的参与。');
                        window.location.href = '/completion';
                    } else {
                        loadNextTrial();
                    }
                } else {
                    alert('提交失败，请重试。');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('网络错误，请重试。');
            });
        }
        
        function getParticipantId() {
            // 从URL参数或本地存储获取参与者ID
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get('participant_id') || localStorage.getItem('participant_id');
        }
        
        function loadNextTrial() {
            // 加载下一个评价任务
            window.location.reload();
        }
        
        // 初始化
        updateProgress();
    </script>
</body>
</html>
        """
        
        return html_template

def main():
    """主函数"""
    print("🧪 设计主观评价实验方案...")
    
    # 基础配置
    config = {
        'experiment_dir': 'subjective_experiment',
        'target_participants': 30,
        'video_count': 20
    }
    
    # 创建实验设计器
    designer = SubjectiveEvaluationDesign(config)
    
    # 1. 设计实验方案
    print("📋 生成实验设计方案...")
    experiment_design = designer.design_experiment()
    
    # 保存实验设计
    design_path = designer.experiment_dir / 'experiment_design.json'
    with open(design_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_design, f, ensure_ascii=False, indent=2)
    print(f"✅ 实验设计方案已保存: {design_path}")
    
    # 2. 生成视频选取方案
    print("🎬 生成视频选取方案...")
    selection_plan = designer.generate_video_selection_plan({'total_samples': 1000})
    
    # 保存选取方案
    selection_path = designer.experiment_dir / 'video_selection_plan.json'
    with open(selection_path, 'w', encoding='utf-8') as f:
        json.dump(selection_plan, f, ensure_ascii=False, indent=2)
    print(f"✅ 视频选取方案已保存: {selection_path}")
    
    # 3. 创建评价界面
    print("🖥️  创建评价界面...")
    html_content = designer.create_evaluation_interface()
    
    # 保存HTML界面
    html_path = designer.experiment_dir / 'evaluation_interface.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 评价界面已保存: {html_path}")
    
    # 4. 生成实验指南
    print("📖 生成实验指南...")
    experiment_guide = generate_experiment_guide(experiment_design)
    
    # 保存指南
    guide_path = designer.experiment_dir / 'experiment_guide.md'
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(experiment_guide)
    print(f"✅ 实验指南已保存: {guide_path}")
    
    print(f"\n🎉 主观评价实验设计完成！")
    print(f"📁 所有文件已保存到: {designer.experiment_dir}")
    print(f"\n📋 下一步操作:")
    print(f"1. 根据视频选取方案选择实验视频")
    print(f"2. 部署评价界面到Web服务器")
    print(f"3. 招募参与者进行实验")
    print(f"4. 收集和分析评价数据")

def generate_experiment_guide(design: Dict) -> str:
    """生成实验指南"""
    
    guide = f"""# AI生成说话人脸视频主观评价实验指南

## 🎯 实验概述

本实验旨在通过主观评价方法对比原始模型和优化模型生成的说话人脸视频质量差异。

## 📋 实验设计

### 实验类型
- **设计类型**: 被试内设计 (Within-subjects design)
- **比较方法**: 配对比较 (Paired comparison)
- **平衡设计**: 是，采用 counterbalancing 消除顺序效应

### 评价维度
{chr(10).join([f"- **{dim['name']}**: {dim['description']} (1-5分量表)" for dim in design['evaluation_dimensions']])}

### 实验流程
- **训练阶段**: 3个样本练习
- **练习阶段**: 5个测试评价
- **正式实验**: 20个视频对评价
- **预计时长**: 20-30分钟/人

## 👥 参与者招募

### 目标人数
- **总人数**: {design['participants']['target_count']}人
- **年龄范围**: {design['participants']['demographics']['age_range']}
- **背景要求**: {design['participants']['demographics']['background']}

### 筛选标准
- {design['participants']['screening']['vision']}
- {design['participants']['screening']['hearing']}
- {design['participants']['screening']['experience']}

## 🎬 视频选取策略

### 选取方法
- **策略**: {design['video_selection_strategy']['method']}
- **样本量**: {design['video_selection_strategy']['sample_size']}个视频

### 质量分布
{chr(10).join([f"- **category}**: {count}个视频" for category, count in design['video_selection_strategy']['distribution'].items()])}

### 选取标准
{chr(10).join([f"- **criterion}**" for criterion in design['video_selection_strategy']['criteria']])}

## 📊 数据收集

### 评价数据
- 5个维度的1-5分评分
- 偏好比较结果 (A/B/无差异)
- 开放式评论

### 元数据
- 参与者基本信息
- 评价时间戳
- 评价时长

## 🔄 实验流程

### 1. 准备阶段
- [ ] 准备实验视频材料
- [ ] 部署评价系统
- [ ] 测试评价流程

### 2. 执行阶段
- [ ] 参与者知情同意
- [ ] 基本信息收集
- [ ] 训练和练习
- [ ] 正式评价实验

### 3. 数据处理
- [ ] 数据质量检查
- [ ] 统计分析
- [ ] 结果可视化

## 📈 统计分析计划

### 主要分析
1. **描述性统计**: 各维度平均分、标准差
2. **配对t检验**: 原始vs优化模型差异
3. **Wilcoxon检验**: 非参数检验
4. **效应量计算**: Cohen's d

### 高级分析
1. **多维度分析**: PCA降维分析
2. **一致性检验**: 评价者间一致性
3. **相关性分析**: 各维度间相关性

## ⚠️ 注意事项

### 实验控制
- 保持实验环境一致
- 确保评价设备标准化
- 避免干扰因素

### 数据质量
- 检查异常值
- 监控评价时间
- 评估参与者注意力

## 📋 实验材料清单

- [ ] 实验视频 (原始模型)
- [ ] 实验视频 (优化模型)
- [ ] 评价界面系统
- [ ] 知情同意书
- [ ] 评价指南
- [ ] 数据收集表格

---
*实验设计版本: {design['version']}*  
*创建时间: {design['created_date']}*
"""
    
    return guide

if __name__ == "__main__":
    main()