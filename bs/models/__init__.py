#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models module for AI-generated talking face video evaluation.

This module provides model classes for evaluating AI-generated talking face videos.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TalkingFaceEvaluationModel(nn.Module):
    """Talking Face Evaluation Model
    
    A simple model for evaluating talking face videos.
    This is a placeholder implementation that can be extended.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # 校验关键配置项，缺失时立即抛出异常以便快速定位问题
        required_keys = ['visual_dim', 'audio_dim', 'hidden_dim']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise KeyError(f"Model config missing required keys: {missing}")
        self.config = config
        
        # Define model architecture based on config
        self.visual_dim = config.get('visual_dim', 2048)
        self.audio_dim = config.get('audio_dim', 768)
        self.keypoint_dim = config.get('keypoint_dim', 0)
        self.au_dim = config.get('au_dim', 0)
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_classes = config.get('num_classes', 1)
        
        # Visual feature processor
        self.visual_fc = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # optional input projection layers (created lazily in forward if needed)
        self.visual_input_proj = None
        
        # Audio feature processor
        self.audio_fc = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.audio_input_proj = None

        # Keypoint feature processor (optional)
        if self.keypoint_dim and self.keypoint_dim > 0:
            self.keypoint_fc = nn.Sequential(
                nn.Linear(self.keypoint_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.keypoint_fc = None

        # AU feature processor (optional)
        if self.au_dim and self.au_dim > 0:
            self.au_fc = nn.Sequential(
                nn.Linear(self.au_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.au_fc = None
        
        # Fusion layer
        # fusion input size depends on available modalities
        fusion_input_dim = self.hidden_dim * 2
        if self.keypoint_fc is not None:
            fusion_input_dim += self.hidden_dim
        if self.au_fc is not None:
            fusion_input_dim += self.hidden_dim

        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific prediction heads (scalar outputs)
        self.lip_sync_head = nn.Linear(self.hidden_dim, 1)
        self.expression_head = nn.Linear(self.hidden_dim, 1)
        self.audio_quality_head = nn.Linear(self.hidden_dim, 1)
        self.cross_modal_head = nn.Linear(self.hidden_dim, 1)
        self.overall_head = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor,
                keypoint_features: Optional[torch.Tensor] = None, au_features: Optional[torch.Tensor] = None,
                syncnet: Optional[torch.Tensor] = None, consistency: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass that supports multiple modalities and returns a dict of task predictions.

        Returns dict with keys: 'lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall'
        """
        # Process visual features
        # 如果输入是时序数据 (batch, seq_len, feat_dim)，先对时间维度做平均池化
        if visual_features is not None and visual_features.dim() == 3:
            visual_features = visual_features.float().mean(dim=1)
        else:
            visual_features = visual_features.float()

        # 若输入维度与初始化期望不同，创建并使用投影层将其映射到 self.visual_dim
        if visual_features is not None and visual_features.size(-1) != self.visual_dim:
            if self.visual_input_proj is None:
                proj = nn.Linear(visual_features.size(-1), self.visual_dim)
                # 初始化为近似恒等映射：对角线为1，其余为0，bias为0，尽量减少对训练初期的扰动
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                min_dim = min(visual_features.size(-1), self.visual_dim)
                for k in range(min_dim):
                    proj.weight.data[k, k] = 1.0
                # 确保与输入在同一设备
                proj = proj.to(visual_features.device)
                setattr(self, 'visual_input_proj', proj)
            visual_features = self.visual_input_proj(visual_features)

        visual_out = self.visual_fc(visual_features)

        # Process audio features
        if audio_features is not None and audio_features.dim() == 3:
            audio_features = audio_features.float().mean(dim=1)
        else:
            audio_features = audio_features.float()

        if audio_features is not None and audio_features.size(-1) != self.audio_dim:
            if self.audio_input_proj is None:
                proj = nn.Linear(audio_features.size(-1), self.audio_dim)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                min_dim = min(audio_features.size(-1), self.audio_dim)
                for k in range(min_dim):
                    proj.weight.data[k, k] = 1.0
                proj = proj.to(audio_features.device)
                setattr(self, 'audio_input_proj', proj)
            audio_features = self.audio_input_proj(audio_features)

        audio_out = self.audio_fc(audio_features)

        feats = [visual_out, audio_out]

        # Process optional keypoint features
        if keypoint_features is not None and self.keypoint_fc is not None:
            if keypoint_features.dim() == 3:
                keypoint_features = keypoint_features.float().mean(dim=1)
            else:
                keypoint_features = keypoint_features.float()
            # lazy projection for keypoint
            if keypoint_features.size(-1) != self.keypoint_dim:
                if not hasattr(self, 'keypoint_input_proj') or self.keypoint_input_proj is None:
                    proj = nn.Linear(keypoint_features.size(-1), self.keypoint_dim)
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    min_dim = min(keypoint_features.size(-1), self.keypoint_dim)
                    for k in range(min_dim):
                        proj.weight.data[k, k] = 1.0
                    proj = proj.to(keypoint_features.device)
                    setattr(self, 'keypoint_input_proj', proj)
                keypoint_features = self.keypoint_input_proj(keypoint_features)
            kp_out = self.keypoint_fc(keypoint_features)
            feats.append(kp_out)

        # Process optional AU features
        if au_features is not None and self.au_fc is not None:
            if au_features.dim() == 3:
                au_features = au_features.float().mean(dim=1)
            else:
                au_features = au_features.float()
            # lazy projection for au
            if au_features.size(-1) != self.au_dim:
                if not hasattr(self, 'au_input_proj') or self.au_input_proj is None:
                    proj = nn.Linear(au_features.size(-1), self.au_dim)
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    min_dim = min(au_features.size(-1), self.au_dim)
                    for k in range(min_dim):
                        proj.weight.data[k, k] = 1.0
                    proj = proj.to(au_features.device)
                    setattr(self, 'au_input_proj', proj)
                au_features = self.au_input_proj(au_features)
            au_out = self.au_fc(au_features)
            feats.append(au_out)

        # If syncnet or consistency are provided (low-dim), append after linear projection
        if syncnet is not None:
            try:
                sync_tensor = syncnet.view(syncnet.size(0), -1).float()
                # project to hidden if sizes mismatch
                if sync_tensor.size(-1) != self.hidden_dim:
                    proj = nn.Linear(sync_tensor.size(-1), self.hidden_dim).to(sync_tensor.device)
                    sync_out = proj(sync_tensor)
                else:
                    sync_out = sync_tensor
                feats.append(sync_out)
            except Exception:
                pass

        if consistency is not None:
            try:
                cons = consistency.view(consistency.size(0), -1).float()
                if cons.size(-1) != self.hidden_dim:
                    proj2 = nn.Linear(cons.size(-1), self.hidden_dim).to(cons.device)
                    cons_out = proj2(cons)
                else:
                    cons_out = cons
                feats.append(cons_out)
            except Exception:
                pass

        # Fuse all available modality outputs
        fused = torch.cat(feats, dim=-1)
        fused_features = self.fusion_fc(fused)

        # Task predictions
        lip_sync_pred = self.lip_sync_head(fused_features).squeeze(-1)
        expression_pred = self.expression_head(fused_features).squeeze(-1)
        audio_quality_pred = self.audio_quality_head(fused_features).squeeze(-1)
        cross_modal_pred = self.cross_modal_head(fused_features).squeeze(-1)
        overall_pred = self.overall_head(fused_features).squeeze(-1)

        return {
            'lip_sync': lip_sync_pred,
            'expression': expression_pred,
            'audio_quality': audio_quality_pred,
            'cross_modal': cross_modal_pred,
            'overall': overall_pred
        }

class MTLTalkingFaceEvaluator(nn.Module):
    """Multi-Task Learning Talking Face Evaluator
    
    A more advanced model for multi-task evaluation of talking face videos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # 校验关键配置项，缺失时立即抛出异常以便快速定位问题
        required_keys = ['visual_dim', 'audio_dim', 'hidden_dim']
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise KeyError(f"Model config missing required keys: {missing}")
        self.config = config
        
        # Model dimensions
        self.visual_dim = config.get('visual_dim', 2048)
        self.audio_dim = config.get('audio_dim', 768)
        self.hidden_dim = config.get('hidden_dim', 512)
        
        # Task-specific output dimensions
        self.quality_classes = config.get('quality_classes', 5)
        self.sync_classes = config.get('sync_classes', 1)
        self.emotion_classes = config.get('emotion_classes', 7)
        
        # Shared feature extractors
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.quality_head = nn.Linear(self.hidden_dim, self.quality_classes)
        self.sync_head = nn.Linear(self.hidden_dim, self.sync_classes)
        self.emotion_head = nn.Linear(self.hidden_dim, self.emotion_classes)
        
    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            visual_features: Visual features tensor
            audio_features: Audio features tensor
            
        Returns:
            Dictionary containing task-specific predictions
        """
        # Encode features
        visual_encoded = self.visual_encoder(visual_features)
        audio_encoded = self.audio_encoder(audio_features)
        
        # Fuse features
        fused = torch.cat([visual_encoded, audio_encoded], dim=-1)
        fused_features = self.fusion_layer(fused)
        
        # Task-specific predictions
        quality_pred = self.quality_head(fused_features)
        sync_pred = self.sync_head(fused_features)
        emotion_pred = self.emotion_head(fused_features)
        
        return {
            'quality': quality_pred,
            'sync': sync_pred,
            'emotion': emotion_pred
        }

# 导入多任务学习模型
from .mtl_model import MultiTaskTalkingFaceEvaluator

# 导入改进版多任务学习模型
try:
    from .improved_mtl_model import ImprovedMultiTaskTalkingFaceEvaluator
except ImportError:
    ImprovedMultiTaskTalkingFaceEvaluator = None

# 高级模型（可选）
try:
    from .advanced_mtl_model import AdvancedMTLTalkingFaceEvaluator
except ImportError:
    AdvancedMTLTalkingFaceEvaluator = None

# Export classes
def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """Factory: create model instance from config.

    Supports names:
      - 'TalkingFaceEvaluationModel' (baseline)
      - 'MTLTalkingFaceEvaluator' (prefers improved architecture if available)
      - 'ImprovedMultiTaskTalkingFaceEvaluator'
      - 'OptimizedMTLTalkingFaceEvaluator' (alias to improved)
    Fallbacks to baseline if name unknown.
    """
    name = (model_config.get('name') or 'TalkingFaceEvaluationModel').strip()
    lname = name.lower()

    # Prefer improved model when requested or when generic MTL name provided
    if lname in {
        'improvedmultitasktalkingfaceevaluator',
        'optimizedmtltalkingfaceevaluator',
        'improved_mtl_talking_face_evaluator',
        'optimized_mtl_talking_face_evaluator'
    } or lname == 'mtltalkingfaceevaluator':
        if ImprovedMultiTaskTalkingFaceEvaluator is not None:
            logger.info(f"Creating model: {ImprovedMultiTaskTalkingFaceEvaluator.__name__} (from name='{name}')")
            return ImprovedMultiTaskTalkingFaceEvaluator(model_config)
        else:
            logger.warning("ImprovedMultiTaskTalkingFaceEvaluator not available, falling back to MultiTaskTalkingFaceEvaluator")
            try:
                return MultiTaskTalkingFaceEvaluator(model_config)
            except Exception as e:
                logger.warning(f"Fallback to baseline due to error constructing MTL: {e}")
                return TalkingFaceEvaluationModel(model_config)

    if lname in {
        'advancedmtltalkingfaceevaluator',
        'advancedmultitasktalkingfaceevaluator',
        'advanced_mtl_talking_face_evaluator',
        'advanced_mtl_multitask_talking_face_evaluator',
        'advancedmtlmodel', 'advanced'
    }:
        if AdvancedMTLTalkingFaceEvaluator is not None:
            logger.info("Creating model: AdvancedMultiTaskTalkingFaceEvaluator")
            return AdvancedMTLTalkingFaceEvaluator(model_config)
        else:
            logger.warning("AdvancedMTLTalkingFaceEvaluator not available, falling back to Improved if possible")
            if ImprovedMultiTaskTalkingFaceEvaluator is not None:
                return ImprovedMultiTaskTalkingFaceEvaluator(model_config)
            return MultiTaskTalkingFaceEvaluator(model_config)

    if lname in {'talkingfaceevaluationmodel', 'baseline', 'simple'}:
        logger.info("Creating model: TalkingFaceEvaluationModel (baseline)")
        return TalkingFaceEvaluationModel(model_config)

    # Unknown name: try to import by mapping, else fallback
    try:
        if lname == 'multitasktalkingfaceevaluator':
            return MultiTaskTalkingFaceEvaluator(model_config)
    except Exception as e:
        logger.warning(f"Failed to construct requested model '{name}': {e}")

    logger.warning(f"Unknown model name '{name}', using baseline TalkingFaceEvaluationModel")
    return TalkingFaceEvaluationModel(model_config)

__all__ = ['TalkingFaceEvaluationModel', 'MTLTalkingFaceEvaluator', 'MultiTaskTalkingFaceEvaluator', 'create_model']
if ImprovedMultiTaskTalkingFaceEvaluator is not None:
    __all__.extend(['ImprovedMultiTaskTalkingFaceEvaluator'])
if AdvancedMTLTalkingFaceEvaluator is not None:
    __all__.extend(['AdvancedMTLTalkingFaceEvaluator'])