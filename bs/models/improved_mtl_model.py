#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedFeatureEncoder(nn.Module):

    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 4),
            nn.LayerNorm(output_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
   
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        main_path = self.encoder(x)
  
        residual = self.residual_proj(x)
        return main_path + residual


class TemporalAttention(nn.Module):

    
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class MultiScaleFusion(nn.Module):
    """"""
    
    def __init__(self, feature_dims: list, fusion_dim: int):
        super().__init__()
        
        # 
        self.projections = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feature_dims
        ])
        

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * len(feature_dims), fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
    def forward(self, features_list: list) -> torch.Tensor:
 
        projected_features = [
            proj(features) for proj, features in zip(self.projections, features_list)
        ]
        

        concatenated = torch.cat(projected_features, dim=-1)
        

        fused = self.fusion_network(concatenated)
        
        return fused


class ImprovedTaskHead(nn.Module):


    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.3,
        task_name: str = "",
    score_min: Optional[float] = None,
    score_max: Optional[float] = None,
        out_activation: str = "sigmoid",  # 'sigmoid' | 'tanh' | 'none'
        clamp_output: bool = True,
    ):
        super().__init__()
        self.task_name = task_name

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.head = nn.Sequential(*layers)

        # 可学习特征缩放（保持原有行为）
        self.feature_scale = nn.Parameter(torch.ones(1))

        # 分数区间映射配置
        self.out_activation = (out_activation or "none").lower()
        self.apply_range = (score_min is not None) and (score_max is not None) and (score_max > score_min)
        if self.apply_range:
            self.register_buffer("_score_min", torch.tensor(float(score_min)))
            self.register_buffer("_score_max", torch.tensor(float(score_max)))
            self.register_buffer("_score_range", torch.tensor(float(score_max - score_min)))
        else:
            self.register_buffer("_score_min", torch.tensor(0.0))
            self.register_buffer("_score_max", torch.tensor(1.0))
            self.register_buffer("_score_range", torch.tensor(1.0))
        self.clamp_output = clamp_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = x * self.feature_scale
        out = self.head(scaled_x)

        # 将输出映射到指定分数区间（例如 1~5）
        if self.apply_range:
            if self.out_activation == "sigmoid":
                base = torch.sigmoid(out)
            elif self.out_activation == "tanh":
                base = 0.5 * (torch.tanh(out) + 1.0)  # 映射到 [0,1]
            else:
                base = out  # 未指定归一化，直接用线性输出（不建议）
                # 若未进行激活，则仅做仿射映射
            out = base * self._score_range + self._score_min
            if self.clamp_output:
                out = torch.clamp(out, min=float(self._score_min.item()), max=float(self._score_max.item()))

        return out


class ImprovedMultiTaskTalkingFaceEvaluator(nn.Module):

    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        # Visual feature dimensions
        self.visual_dim = config['visual_dim']
        self.audio_dim = config['audio_dim']
        self.keypoint_dim = config['keypoint_dim']
        self.au_dim = config['au_dim']
        

        self.encoder_dim = config.get('encoder_dim', 512)  # Encoder dimension

        # Improved feature encoders
        self.visual_encoder = ImprovedFeatureEncoder(
            self.visual_dim, self.encoder_dim, config.get('dropout', 0.3)
        )
        self.audio_encoder = ImprovedFeatureEncoder(
            self.audio_dim, self.encoder_dim, config.get('dropout', 0.3)
        )
        self.keypoint_encoder = ImprovedFeatureEncoder(
            self.keypoint_dim, self.encoder_dim, config.get('dropout', 0.3)
        )
        self.au_encoder = ImprovedFeatureEncoder(
            self.au_dim, self.encoder_dim, config.get('dropout', 0.3)
        )
        

        self.temporal_attention = TemporalAttention(
            feature_dim=self.encoder_dim,
            num_heads=config.get('num_heads', 8),
            dropout=config.get('temporal_dropout', 0.1)
        )
        

        transformer_config = config.get('transformer', {})
        self.multimodal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=transformer_config.get('num_heads', 8),
                dim_feedforward=transformer_config.get('dim_feedforward', 2048),
                dropout=transformer_config.get('dropout', 0.1),
                batch_first=True
            ),
            num_layers=transformer_config.get('num_layers', 4)  
        )
        

        self.multi_scale_fusion = MultiScaleFusion(
            feature_dims=[self.encoder_dim] * 4,
            fusion_dim=self.encoder_dim
        )
        
        # 任务头配置与输出映射（分数区间默认 1~5）
        task_heads_config = config.get('task_heads', {})
        default_score_min = float(task_heads_config.get('score_min', 1.0))
        default_score_max = float(task_heads_config.get('score_max', 5.0))
        out_activation = task_heads_config.get('out_activation', 'sigmoid')
        clamp_output = bool(task_heads_config.get('clamp_output', True))

        self.lip_sync_head = ImprovedTaskHead(
            self.encoder_dim,
            task_heads_config.get('lip_sync', {}).get('hidden_dims', [256, 128]),
            task_heads_config.get('lip_sync', {}).get('dropout', 0.3),
            task_name="lip_sync",
            score_min=task_heads_config.get('lip_sync', {}).get('score_min', default_score_min),
            score_max=task_heads_config.get('lip_sync', {}).get('score_max', default_score_max),
            out_activation=task_heads_config.get('lip_sync', {}).get('out_activation', out_activation),
            clamp_output=task_heads_config.get('lip_sync', {}).get('clamp_output', clamp_output),
        )
        self.expression_head = ImprovedTaskHead(
            self.encoder_dim,
            task_heads_config.get('expression', {}).get('hidden_dims', [256, 128]),
            task_heads_config.get('expression', {}).get('dropout', 0.3),
            task_name="expression",
            score_min=task_heads_config.get('expression', {}).get('score_min', default_score_min),
            score_max=task_heads_config.get('expression', {}).get('score_max', default_score_max),
            out_activation=task_heads_config.get('expression', {}).get('out_activation', out_activation),
            clamp_output=task_heads_config.get('expression', {}).get('clamp_output', clamp_output),
        )
        self.audio_quality_head = ImprovedTaskHead(
            self.encoder_dim,
            task_heads_config.get('audio_quality', {}).get('hidden_dims', [256, 128]),
            task_heads_config.get('audio_quality', {}).get('dropout', 0.3),
            task_name="audio_quality",
            score_min=task_heads_config.get('audio_quality', {}).get('score_min', default_score_min),
            score_max=task_heads_config.get('audio_quality', {}).get('score_max', default_score_max),
            out_activation=task_heads_config.get('audio_quality', {}).get('out_activation', out_activation),
            clamp_output=task_heads_config.get('audio_quality', {}).get('clamp_output', clamp_output),
        )
        self.cross_modal_head = ImprovedTaskHead(
            self.encoder_dim,
            task_heads_config.get('cross_modal', {}).get('hidden_dims', [256, 128]),
            task_heads_config.get('cross_modal', {}).get('dropout', 0.3),
            task_name="cross_modal",
            score_min=task_heads_config.get('cross_modal', {}).get('score_min', default_score_min),
            score_max=task_heads_config.get('cross_modal', {}).get('score_max', default_score_max),
            out_activation=task_heads_config.get('cross_modal', {}).get('out_activation', out_activation),
            clamp_output=task_heads_config.get('cross_modal', {}).get('clamp_output', clamp_output),
        )
        
        self.overall_head = ImprovedTaskHead(
            self.encoder_dim,
            [256, 128],
            dropout=0.2,
            task_name="overall",
            score_min=task_heads_config.get('overall', {}).get('score_min', default_score_min),
            score_max=task_heads_config.get('overall', {}).get('score_max', default_score_max),
            out_activation=task_heads_config.get('overall', {}).get('out_activation', out_activation),
            clamp_output=task_heads_config.get('overall', {}).get('clamp_output', clamp_output),
        )
        
        self.loss_weights = nn.ParameterDict({
            'lip_sync': nn.Parameter(torch.tensor(1.0)),
            'expression': nn.Parameter(torch.tensor(1.0)),
            'audio_quality': nn.Parameter(torch.tensor(1.0)),
            'cross_modal': nn.Parameter(torch.tensor(1.0)),
            'overall': nn.Parameter(torch.tensor(1.0))
        })
        
        self.criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()  
        self.temperature = nn.Parameter(torch.ones(1))
        # 将 mean+max 拼接 (2*encoder_dim) 投影回 encoder_dim
        self.pool_proj = nn.Linear(self.encoder_dim * 2, self.encoder_dim)
        # 自适应输入维度投影缓存
        self._adaptive_proj = nn.ModuleDict()
        
        logger.info(f"ImprovedMultiTaskTalkingFaceEvaluator initialized with {self._count_parameters():,} parameters")
    
    def _count_parameters(self) -> int:

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor,
                keypoint_features: torch.Tensor, au_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        
        Args:
            visual_features: [batch_size, seq_len, visual_dim]
            audio_features: [batch_size, seq_len, audio_dim]
            keypoint_features: [batch_size, seq_len, keypoint_dim]
            au_features: [batch_size, seq_len, au_dim]
            
        Returns:
            Dict containing task predictions
        """
        def ensure_3d(x: torch.Tensor):
            if x.dim() == 2:  # (seq, dim) -> (1, seq, dim)
                return x.unsqueeze(0)
            return x

        visual_features = ensure_3d(visual_features)
        audio_features = ensure_3d(audio_features)
        keypoint_features = ensure_3d(keypoint_features)
        au_features = ensure_3d(au_features)

        # 记录原始形状
        logger.debug(
            f"Forward input (after ensure_3d) shapes - visual:{tuple(visual_features.shape)} "
            f"audio:{tuple(audio_features.shape)} keypoint:{tuple(keypoint_features.shape)} au:{tuple(au_features.shape)}"
        )

        batch_size = visual_features.shape[0]
        seq_lengths = [visual_features.shape[1], audio_features.shape[1], keypoint_features.shape[1], au_features.shape[1]]
        min_seq = min(seq_lengths)
        if len(set(seq_lengths)) > 1:
            logger.warning(f"Modal sequence length mismatch {seq_lengths}, aligning to min={min_seq}")
            visual_features = visual_features[:, :min_seq]
            audio_features = audio_features[:, :min_seq]
            keypoint_features = keypoint_features[:, :min_seq]
            au_features = au_features[:, :min_seq]
        seq_len = min_seq

        # 编码：分别处理防止形状不一致
        def encode_seq(x, declared_in_dim, encoder, name):
            B, T, D = x.shape
            if D != declared_in_dim:
                key = f"{name}_proj_{D}_to_{declared_in_dim}"
                if key not in self._adaptive_proj:
                    logger.warning(f"[AdaptiveProj] {name}: expected {declared_in_dim}, got {D}; creating Linear({D}->{declared_in_dim})")
                    self._adaptive_proj[key] = nn.Linear(D, declared_in_dim).to(x.device)
                x = self._adaptive_proj[key](x)
                D = declared_in_dim
            x_flat = x.reshape(B * T, D)
            encoded = encoder(x_flat).view(B, T, self.encoder_dim)
            return encoded

        visual_encoded = encode_seq(visual_features, self.visual_dim, self.visual_encoder, 'visual')
        audio_encoded = encode_seq(audio_features, self.audio_dim, self.audio_encoder, 'audio')
        keypoint_encoded = encode_seq(keypoint_features, self.keypoint_dim, self.keypoint_encoder, 'keypoint')
        au_encoded = encode_seq(au_features, self.au_dim, self.au_encoder, 'au')
        
        visual_temporal = self.temporal_attention(visual_encoded)
        audio_temporal = self.temporal_attention(audio_encoded)
        keypoint_temporal = self.temporal_attention(keypoint_encoded)
        au_temporal = self.temporal_attention(au_encoded)
        
        visual_avg = torch.mean(visual_temporal, dim=1)
        visual_max = torch.max(visual_temporal, dim=1)[0]
        visual_fused = torch.cat([visual_avg, visual_max], dim=-1)
        
        audio_avg = torch.mean(audio_temporal, dim=1)
        audio_max = torch.max(audio_temporal, dim=1)[0]
        audio_fused = torch.cat([audio_avg, audio_max], dim=-1)
        
        keypoint_avg = torch.mean(keypoint_temporal, dim=1)
        keypoint_max = torch.max(keypoint_temporal, dim=1)[0]
        keypoint_fused = torch.cat([keypoint_avg, keypoint_max], dim=-1)
        
        au_avg = torch.mean(au_temporal, dim=1)
        au_max = torch.max(au_temporal, dim=1)[0]
        au_fused = torch.cat([au_avg, au_max], dim=-1)
        
        visual_final = self.pool_proj(visual_fused)
        audio_final = self.pool_proj(audio_fused)
        keypoint_final = self.pool_proj(keypoint_fused)
        au_final = self.pool_proj(au_fused)

        multimodal_features = torch.stack([
            visual_final, audio_final, keypoint_final, au_final
        ], dim=1)

        fused_features = self.multimodal_fusion(multimodal_features)

        multi_scale_features = self.multi_scale_fusion([
            visual_final, audio_final, keypoint_final, au_final
        ])

        final_features = (fused_features.mean(dim=1) + multi_scale_features) / 2.0

        predictions = {
            'lip_sync': self.lip_sync_head(final_features),
            'expression': self.expression_head(final_features),
            'audio_quality': self.audio_quality_head(final_features),
            'cross_modal': self.cross_modal_head(final_features),
            'overall': self.overall_head(final_features)
        }
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
     
        losses = {}
        

        for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']:
            if task in predictions and task in targets:
                # MSE_1
                mse_loss = self.criterion(predictions[task], targets[task])
  	
                l1_loss = self.l1_criterion(predictions[task], targets[task])

                task_loss = mse_loss + 0.1 * l1_loss
                
   
                weight = torch.sigmoid(self.loss_weights[task])
                losses[task] = task_loss * weight
        
     
        if len(predictions) > 1:
            consistency_loss = 0
            task_predictions = torch.stack([predictions[task] for task in ['lip_sync', 'expression', 'audio_quality', 'cross_modal']], dim=-1)
            
            consistency_loss = torch.std(task_predictions, dim=-1).mean()
            losses['consistency'] = 0.1 * consistency_loss
        
        return losses
    
    def get_task_weights(self) -> Dict[str, float]:

        return {task: torch.sigmoid(weight).item() for task, weight in self.loss_weights.items()}
    
    def set_task_weights(self, weights: Dict[str, float]):
     
        for task, weight in weights.items():
            if task in self.loss_weights:
                self.loss_weights[task].data = torch.tensor(weight)
                logger.info(f"Updated {task} loss weight to {weight}")



MTLTalkingFaceEvaluator = ImprovedMultiTaskTalkingFaceEvaluator