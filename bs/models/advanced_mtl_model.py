#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced multi-task talking face evaluation model.
特性 (满足需求 6 点并可扩展):
1. 更深层次特征编码器 (沿用改进版 3 层 + 残差)
2. 时序注意力 (Multi-Head 自注意 + FFN 残差层)
3. 多尺度特征融合 (mean + max 以及 multimodal transformer + 融合网络)
4. 增强正则化 (LayerNorm + Dropout + 梯度裁剪由训练脚本执行)
5. 动态任务权重 (支持: learned, uncertainty, gradnorm 三种策略)
6. 一致性损失 (支持: std, corr, cov 三种模式)

保持与 ImprovedMultiTaskTalkingFaceEvaluator 接口兼容: forward 返回 predictions；compute_loss 返回 losses dict。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import math

# ------------------ 基础模块 ------------------ #

class ResidualMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(depth):
            out_dim = hidden_dim if i < depth - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.net(x)
        res = self.residual_proj(x)
        return main + res

class TemporalBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float = 0.3, act: str = 'gelu',
                 score_min: float = 1.0, score_max: float = 5.0, activation: str = 'sigmoid', clamp: bool = True):
        super().__init__()
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU() if act == 'gelu' else nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(hidden) - 1 else dropout * 0.5))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.feature_scale = nn.Parameter(torch.ones(1))
        self.score_min = score_min
        self.score_max = score_max
        self.activation = activation
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.feature_scale
        out = self.net(x)
        if self.activation == 'sigmoid':
            base = torch.sigmoid(out)
        elif self.activation == 'tanh':
            base = 0.5 * (torch.tanh(out) + 1)
        else:
            base = out
        if self.activation in ('sigmoid', 'tanh'):
            out = base * (self.score_max - self.score_min) + self.score_min
            if self.clamp:
                out = torch.clamp(out, self.score_min, self.score_max)
        return out

# ------------------ 动态任务权重策略 ------------------ #
class DynamicWeighting(nn.Module):
    def __init__(self, tasks: List[str], strategy: str = 'learned'):
        super().__init__()
        self.tasks = tasks
        self.strategy = strategy.lower()
        if self.strategy == 'learned':
            self.params = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in tasks})  # sigmoid -> (0,1)
        elif self.strategy == 'uncertainty':
            # Kendall & Gal 2018: weight = 1/(2 sigma^2) + log sigma
            self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in tasks})
        elif self.strategy == 'gradnorm':
            # Will store initial losses externally
            self.params = nn.ParameterDict({t: nn.Parameter(torch.tensor(1.0)) for t in tasks})
            self._initial_losses = None
        else:
            raise ValueError(f"Unknown dynamic weighting strategy: {strategy}")

    def forward(self, losses: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        weighted = {}
        if self.strategy == 'learned':
            for t, l in losses.items():
                w = torch.sigmoid(self.params[t]) + 1e-4
                weighted[t] = l * w
        elif self.strategy == 'uncertainty':
            for t, l in losses.items():
                log_var = self.log_vars[t]
                weighted[t] = torch.exp(-log_var) * l + log_var  # (1/(2σ^2))*2 省略常数
        elif self.strategy == 'gradnorm':
            # gradnorm 需要在训练步骤中基于梯度再调整，此处只先乘当前参数
            for t, l in losses.items():
                weighted[t] = l * self.params[t]
        return weighted

    def gradnorm_update(self, losses: Dict[str, torch.Tensor], model: nn.Module, alpha: float = 1.5):
        if self.strategy != 'gradnorm':
            return
        # 初始化基线
        with torch.no_grad():
            if self._initial_losses is None:
                self._initial_losses = {t: l.detach() for t, l in losses.items()}
        # 选一个共享参数 (第一个可训练参数)
        shared_params = None
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                shared_params = p
                break
        if shared_params is None:
            return
        # 当前梯度范数
        grad_norms = {}
        for t, l in losses.items():
            g = torch.autograd.grad(l, shared_params, retain_graph=True, create_graph=True)[0]
            grad_norms[t] = g.norm(2)
        with torch.no_grad():
            avg_grad = torch.stack(list(grad_norms.values())).mean()
            # 计算相对训练进度比率 (ri)
            rel_losses = {t: (l / self._initial_losses[t]) for t, l in losses.items()}
            avg_rel = torch.stack([v for v in rel_losses.values()]).mean()
            target = {t: (rel_losses[t] / avg_rel) ** alpha * avg_grad for t in losses}
            # L1 损失更新参数
            for t in losses:
                diff = grad_norms[t] - target[t]
                self.params[t].data -= 0.01 * diff  # 简单学习率
                self.params[t].data = self.params[t].data.clamp(1e-3, 100)

# ------------------ 一致性损失 ------------------ #
class ConsistencyLoss(nn.Module):
    def __init__(self, mode: str = 'std', weight: float = 0.1, tasks: Optional[List[str]] = None):
        super().__init__()
        self.mode = mode.lower()
        self.weight = weight
        self.tasks = tasks

    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.tasks is None:
            keys = list(predictions.keys())
        else:
            keys = [k for k in self.tasks if k in predictions]
        if len(keys) < 2:
            return predictions[keys[0]].new_tensor(0.0)
        preds = torch.stack([predictions[k].squeeze(-1) for k in keys], dim=-1)  # [B, T? -> assume B, tasks]
        # 模式
        if self.mode == 'std':
            val = preds.std(dim=-1).mean()
        elif self.mode == 'corr':
            # 相关性: 1 - 平均皮尔逊相关
            preds_centered = preds - preds.mean(dim=0, keepdim=True)
            cov = preds_centered.transpose(0,1) @ preds_centered / (preds.shape[0] - 1 + 1e-6)
            diag = torch.diag(cov)
            std = torch.sqrt(diag + 1e-6)
            corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-6)
            # 取下三角非对角平均
            idx = torch.tril_indices(len(keys), len(keys), offset=-1)
            mean_corr = corr[idx[0], idx[1]].mean()
            val = 1 - mean_corr  # 相关性越高越小
        elif self.mode == 'cov':
            preds_centered = preds - preds.mean(dim=0, keepdim=True)
            cov = (preds_centered ** 2).mean()  # 简化: 方差和
            val = cov
        else:
            val = preds.std(dim=-1).mean()
        return val * self.weight

# ------------------ 主模型 ------------------ #
class AdvancedMultiTaskTalkingFaceEvaluator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.visual_dim = config['visual_dim']
        self.audio_dim = config['audio_dim']
        self.keypoint_dim = config['keypoint_dim']
        self.au_dim = config['au_dim']
        self.encoder_dim = config.get('encoder_dim', 512)
        dropout = config.get('dropout', 0.3)

        # Encoders
        self.visual_encoder = ResidualMLPEncoder(self.visual_dim, self.encoder_dim, depth=3, dropout=dropout)
        self.audio_encoder = ResidualMLPEncoder(self.audio_dim, self.encoder_dim, depth=3, dropout=dropout)
        self.keypoint_encoder = ResidualMLPEncoder(self.keypoint_dim, self.encoder_dim, depth=3, dropout=dropout)
        self.au_encoder = ResidualMLPEncoder(self.au_dim, self.encoder_dim, depth=3, dropout=dropout)

        # Temporal attention block (shared)
        attn_heads = config.get('num_heads', 8)
        attn_dropout = config.get('temporal_dropout', 0.1)
        self.temporal_block = TemporalBlock(self.encoder_dim, num_heads=attn_heads, dropout=attn_dropout)

        # Multimodal fusion transformer
        tr_cfg = config.get('transformer', {})
        self.multimodal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.encoder_dim,
                nhead=tr_cfg.get('num_heads', 8),
                dim_feedforward=tr_cfg.get('dim_feedforward', 2048),
                dropout=tr_cfg.get('dropout', 0.1),
                batch_first=True
            ),
            num_layers=tr_cfg.get('num_layers', 4)
        )

        # mean+max -> projection
        self.pool_proj = nn.Linear(self.encoder_dim * 2, self.encoder_dim)

        # multi-scale fusion (reuse approach: concat and MLP)
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(self.encoder_dim * 4, self.encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.encoder_dim * 2, self.encoder_dim)
        )

        # Task heads
        task_cfg = config.get('task_heads', {})
        def build_head(name: str):
            cfg = task_cfg.get(name, {})
            return TaskHead(
                input_dim=self.encoder_dim,
                hidden=cfg.get('hidden_dims', [256, 128]),
                dropout=cfg.get('dropout', 0.3),
                act=cfg.get('act', 'gelu'),
                score_min=cfg.get('score_min', task_cfg.get('score_min', 1.0)),
                score_max=cfg.get('score_max', task_cfg.get('score_max', 5.0)),
                activation=cfg.get('out_activation', task_cfg.get('out_activation', 'sigmoid')),
                clamp=cfg.get('clamp_output', task_cfg.get('clamp_output', True))
            )
        self.tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
        self.heads = nn.ModuleDict({t: build_head(t) for t in self.tasks})

        # Loss functions
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        # Dynamic weighting
        adv_cfg = config.get('advanced', {})
        dw_strategy = adv_cfg.get('task_weighting', 'learned')
        self.dynamic_weighting = DynamicWeighting(self.tasks, strategy=dw_strategy)

        # Consistency loss
        cons_cfg = adv_cfg.get('consistency', {})
        self.consistency_loss = ConsistencyLoss(
            mode=cons_cfg.get('mode', 'std'),
            weight=cons_cfg.get('weight', 0.1),
            tasks=cons_cfg.get('tasks', ['lip_sync', 'expression', 'audio_quality', 'cross_modal'])
        )

        # Adaptive projection cache (handle mismatched dims)
        self._adaptive_proj = nn.ModuleDict()

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0)
        return x

    def _align_seq(self, modalities: List[torch.Tensor]) -> List[torch.Tensor]:
        lens = [m.shape[1] for m in modalities]
        if len(set(lens)) > 1:
            min_len = min(lens)
            modalities = [m[:, :min_len] for m in modalities]
        return modalities

    def _encode(self, x: torch.Tensor, expected_dim: int, encoder: nn.Module, key: str) -> torch.Tensor:
        B, T, D = x.shape
        if D != expected_dim:
            proj_key = f"{key}_proj_{D}_to_{expected_dim}"
            if proj_key not in self._adaptive_proj:
                self._adaptive_proj[proj_key] = nn.Linear(D, expected_dim).to(x.device)
            x = self._adaptive_proj[proj_key](x)
        x_flat = x.view(B * T, -1)
        encoded = encoder(x_flat).view(B, T, self.encoder_dim)
        return encoded

    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor,
                keypoint_features: torch.Tensor, au_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure 3D
        visual_features = self._ensure_3d(visual_features)
        audio_features = self._ensure_3d(audio_features)
        keypoint_features = self._ensure_3d(keypoint_features)
        au_features = self._ensure_3d(au_features)

        # Align length
        visual_features, audio_features, keypoint_features, au_features = self._align_seq([
            visual_features, audio_features, keypoint_features, au_features
        ])

        # Encode each modality
        v_enc = self._encode(visual_features, self.visual_dim, self.visual_encoder, 'visual')
        a_enc = self._encode(audio_features, self.audio_dim, self.audio_encoder, 'audio')
        k_enc = self._encode(keypoint_features, self.keypoint_dim, self.keypoint_encoder, 'keypoint')
        au_enc = self._encode(au_features, self.au_dim, self.au_encoder, 'au')

        # Temporal modeling
        v_temp = self.temporal_block(v_enc)
        a_temp = self.temporal_block(a_enc)
        k_temp = self.temporal_block(k_enc)
        au_temp = self.temporal_block(au_enc)

        def pool_and_project(t: torch.Tensor):
            avg = t.mean(dim=1)
            mx = t.max(dim=1)[0]
            fused = torch.cat([avg, mx], dim=-1)
            return self.pool_proj(fused)

        v_final = pool_and_project(v_temp)
        a_final = pool_and_project(a_temp)
        k_final = pool_and_project(k_temp)
        au_final = pool_and_project(au_temp)

        # Multimodal fusion
        multi_stack = torch.stack([v_final, a_final, k_final, au_final], dim=1)  # [B,4,D]
        fused_seq = self.multimodal_fusion(multi_stack)
        fused_mean = fused_seq.mean(dim=1)

        # Multi-scale fusion
        multi_scale = self.multi_scale_fusion(torch.cat([v_final, a_final, k_final, au_final], dim=-1))

        final_features = (fused_mean + multi_scale) / 2.0

        predictions = {t: self.heads[t](final_features) for t in self.tasks}
        return predictions

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], epoch: int = 0) -> Dict[str, torch.Tensor]:
        base_losses = {}
        # 基础 MSE + L1 混合
        for t in self.tasks:
            if t in predictions and f"{t}_score" in targets:
                mse = self.mse(predictions[t], targets[f"{t}_score"])  # targets 中字段名假设为 <task>_score
                l1 = self.l1(predictions[t], targets[f"{t}_score"]) * 0.1
                base_losses[t] = mse + l1
        # 动态权重
        weighted = self.dynamic_weighting(base_losses, epoch=epoch)
        # 一致性
        cons = self.consistency_loss({k: predictions[k] for k in predictions if k != 'overall'})
        if cons is not None:
            weighted['consistency'] = cons
        return weighted


AdvancedMTLTalkingFaceEvaluator = AdvancedMultiTaskTalkingFaceEvaluator
