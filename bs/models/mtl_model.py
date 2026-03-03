#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多任务学习模型 - 用于AI生成说话人脸视频评价

实现口型同步、表情自然度、音频质量和跨模态一致性的多任务评估。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEncoder(nn.Module):
    """特征编码器
    
    将不同模态的特征编码到统一的特征空间。
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultiModalTransformer(nn.Module):
    """多模态Transformer融合模块"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 4, d_model))  # 4个模态
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, num_modalities, d_model]
        Returns:
            融合后的特征: [batch_size, num_modalities, d_model]
        """
        # 添加位置编码
        features = features + self.pos_encoding
        
        # Transformer编码
        output = self.transformer(features)
        
        return output


class TaskHead(nn.Module):
    """任务特定的预测头"""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.head = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskTalkingFaceEvaluator(nn.Module):
    """多任务说话人脸视频评估模型
    
    实现四个评估任务：
    1. 口型同步 (lip_sync)
    2. 表情自然度 (expression) 
    3. 音频质量 (audio_quality)
    4. 跨模态一致性 (cross_modal)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 特征维度
        self.visual_dim = config['visual_dim']
        self.audio_dim = config['audio_dim']
        self.keypoint_dim = config['keypoint_dim']
        self.au_dim = config['au_dim']
        
        # 编码器输出维度
        self.encoder_dim = config['encoder_dim']
        
        # 特征编码器
        self.visual_encoder = FeatureEncoder(
            self.visual_dim, self.encoder_dim, config['dropout']
        )
        self.audio_encoder = FeatureEncoder(
            self.audio_dim, self.encoder_dim, config['dropout']
        )
        self.keypoint_encoder = FeatureEncoder(
            self.keypoint_dim, self.encoder_dim, config['dropout']
        )
        self.au_encoder = FeatureEncoder(
            self.au_dim, self.encoder_dim, config['dropout']
        )
        
        # 多模态融合
        transformer_config = config['transformer']
        self.multimodal_fusion = MultiModalTransformer(
            d_model=self.encoder_dim,
            nhead=transformer_config['num_heads'],
            num_layers=transformer_config['num_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout']
        )
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 任务头
        task_heads_config = config['task_heads']
        self.lip_sync_head = TaskHead(
            self.encoder_dim, 
            task_heads_config['lip_sync']['hidden_dims'],
            task_heads_config['lip_sync']['dropout']
        )
        self.expression_head = TaskHead(
            self.encoder_dim,
            task_heads_config['expression']['hidden_dims'],
            task_heads_config['expression']['dropout']
        )
        self.audio_quality_head = TaskHead(
            self.encoder_dim,
            task_heads_config['audio_quality']['hidden_dims'],
            task_heads_config['audio_quality']['dropout']
        )
        self.cross_modal_head = TaskHead(
            self.encoder_dim,
            task_heads_config['cross_modal']['hidden_dims'],
            task_heads_config['cross_modal']['dropout']
        )
        # 可选：总体评分头（与 Trainer/Evaluator 中的 overall 对齐；若配置缺失则跳过）
        self.overall_head = None
        if 'overall' in task_heads_config:
            self.overall_head = TaskHead(
                self.encoder_dim,
                task_heads_config['overall'].get('hidden_dims', task_heads_config['lip_sync']['hidden_dims']),
                task_heads_config['overall'].get('dropout', task_heads_config['lip_sync'].get('dropout', 0.2))
            )
        
        # 损失权重
        self.loss_weights = {
            'lip_sync': task_heads_config['lip_sync']['loss_weight'],
            'expression': task_heads_config['expression']['loss_weight'],
            'audio_quality': task_heads_config['audio_quality']['loss_weight'],
            'cross_modal': task_heads_config['cross_modal']['loss_weight']
        }
        
        # 损失函数
        self.criterion = nn.MSELoss()

        # 初始化日志
        logger.info(f"MultiTaskTalkingFaceEvaluator initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor,
                keypoint_features: torch.Tensor, au_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            visual_features: [batch_size, seq_len, visual_dim]
            audio_features: [batch_size, seq_len, audio_dim]
            keypoint_features: [batch_size, seq_len, keypoint_dim]
            au_features: [batch_size, seq_len, au_dim]
            
        Returns:
            Dict containing task predictions
        """
        batch_size, seq_len = visual_features.shape[:2]
        
        # 特征编码 - 对序列中的每一帧进行编码
        visual_encoded = self.visual_encoder(visual_features.view(-1, self.visual_dim))
        audio_encoded = self.audio_encoder(audio_features.view(-1, self.audio_dim))
        keypoint_encoded = self.keypoint_encoder(keypoint_features.view(-1, self.keypoint_dim))
        au_encoded = self.au_encoder(au_features.view(-1, self.au_dim))
        
        # 重塑为序列格式
        visual_encoded = visual_encoded.view(batch_size, seq_len, self.encoder_dim)
        audio_encoded = audio_encoded.view(batch_size, seq_len, self.encoder_dim)
        keypoint_encoded = keypoint_encoded.view(batch_size, seq_len, self.encoder_dim)
        au_encoded = au_encoded.view(batch_size, seq_len, self.encoder_dim)
        
        # 时间维度平均池化，得到固定长度特征
        visual_pooled = torch.mean(visual_encoded, dim=1)  # [batch_size, encoder_dim]
        audio_pooled = torch.mean(audio_encoded, dim=1)
        keypoint_pooled = torch.mean(keypoint_encoded, dim=1)
        au_pooled = torch.mean(au_encoded, dim=1)
        
        # 堆叠为多模态特征 [batch_size, 4, encoder_dim]
        multimodal_features = torch.stack([
            visual_pooled, audio_pooled, keypoint_pooled, au_pooled
        ], dim=1)
        
        # 多模态融合
        fused_features = self.multimodal_fusion(multimodal_features)
        
        # 全局池化得到最终特征向量
        pooled_features = torch.mean(fused_features, dim=1)  # [batch_size, encoder_dim]
        
        # 任务预测
        predictions = {
            'lip_sync': self.lip_sync_head(pooled_features),
            'expression': self.expression_head(pooled_features),
            'audio_quality': self.audio_quality_head(pooled_features),
            'cross_modal': self.cross_modal_head(pooled_features)
        }
        # 若启用 overall 头，则补充总体评分输出；否则以四任务平均作为回退，避免下游KeyError
        if self.overall_head is not None:
            predictions['overall'] = self.overall_head(pooled_features)
        else:
            predictions['overall'] = (
                predictions['lip_sync'] +
                predictions['expression'] +
                predictions['audio_quality'] +
                predictions['cross_modal']
            ) / 4.0
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            各任务的损失值
        """
        losses = {}

        for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']:
            if task in predictions and task in targets:
                task_loss = self.criterion(predictions[task], targets[task])
                # 对于 overall，如未在权重配置中定义，则默认与 lip_sync 相同或为 1.0
                weight = self.loss_weights.get(task, self.loss_weights.get('lip_sync', 1.0))
                losses[task] = task_loss * weight

        return losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """获取任务权重"""
        return self.loss_weights.copy()
    
    def set_task_weights(self, weights: Dict[str, float]):
        """设置任务权重"""
        for task, weight in weights.items():
            if task in self.loss_weights:
                self.loss_weights[task] = weight
                logger.info(f"Updated {task} loss weight to {weight}")