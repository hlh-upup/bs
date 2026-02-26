#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存优化的多任务学习训练器

基于2024年最新性能优化技术：
1. 梯度检查点（Gradient Checkpointing）
2. 混合精度训练优化（AMP with dynamic scaling）
3. 内存高效的梯度累积
4. 动态批次大小调整
5. 智能内存管理
6. 多任务损失平衡优化
"""

import os
import time
import gc
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from contextlib import contextmanager
import psutil
import GPUtil

from .trainer import Trainer

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """GPU内存分析器"""
    
    def __init__(self, device):
        self.device = device
        self.memory_stats = []
        
    def get_memory_usage(self):
        """获取当前GPU内存使用情况"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved(self.device) / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,  # GB
            }
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
    
    def log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        memory_info = self.get_memory_usage()
        self.memory_stats.append({
            'stage': stage,
            'timestamp': time.time(),
            **memory_info
        })
        
        logger.info(f"[Memory] {stage}: "
                   f"Allocated: {memory_info['allocated']:.2f}GB, "
                   f"Cached: {memory_info['cached']:.2f}GB, "
                   f"Max: {memory_info['max_allocated']:.2f}GB")


class GradientCheckpointedModel(nn.Module):
    """支持梯度检查点的模型包装器"""
    
    def __init__(self, model: nn.Module, checkpoint_segments: int = 4):
        super().__init__()
        self.model = model
        self.checkpoint_segments = checkpoint_segments
        
    def forward(self, *args, **kwargs):
        """前向传播，使用梯度检查点"""
        if self.training and self.checkpoint_segments > 1:
            # 将输入分割为多个检查点段
            visual_features = kwargs.get('visual_features')
            batch_size = visual_features.shape[0]
            segment_size = max(1, batch_size // self.checkpoint_segments)
            
            outputs_list = []
            
            for i in range(0, batch_size, segment_size):
                end_idx = min(i + segment_size, batch_size)
                
                # 提取当前段的输入
                segment_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                        segment_kwargs[key] = value[i:end_idx]
                    else:
                        segment_kwargs[key] = value
                
                # 使用检查点
                segment_output = checkpoint(
                    self.model.forward,
                    **segment_kwargs
                )
                
                outputs_list.append(segment_output)
            
            # 合并输出
            merged_outputs = {}
            for task in outputs_list[0].keys():
                merged_outputs[task] = torch.cat([out[task] for out in outputs_list], dim=0)
            
            return merged_outputs
        else:
            return self.model(*args, **kwargs)


class DynamicLossScaler:
    """动态损失缩放器，优化混合精度训练"""
    
    def __init__(self, init_scale: float = 2.**16, growth_factor: float = 2.0, 
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.init_scale = init_scale
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._consecutive_successes = 0
        
    def update(self, has_overflow: bool):
        """根据是否溢出更新缩放因子"""
        if has_overflow:
            self.scale *= self.backoff_factor
            self._consecutive_successes = 0
            logger.info(f"Loss scale decreased to {self.scale}")
        else:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.growth_interval:
                self.scale *= self.growth_factor
                self._consecutive_successes = 0
                logger.info(f"Loss scale increased to {self.scale}")
                
    def state_dict(self):
        return {
            'scale': self.scale,
            'growth_tracker': self._growth_tracker,
            'consecutive_successes': self._consecutive_successes
        }
        
    def load_state_dict(self, state_dict):
        self.scale = state_dict['scale']
        self._growth_tracker = state_dict['growth_tracker']
        self._consecutive_successes = state_dict['consecutive_successes']


class MultiTaskLossBalancer:
    """多任务损失平衡器"""
    
    def __init__(self, tasks: list, balance_method: str = 'uncertainty'):
        self.tasks = tasks
        self.balance_method = balance_method
        
        if balance_method == 'uncertainty':
            # 基于不确定性的损失平衡
            self.log_vars = nn.Parameter(torch.zeros(len(tasks)))
        elif balance_method == 'gradnorm':
            # 基于梯度范数的平衡
            self.task_weights = nn.Parameter(torch.ones(len(tasks)))
        else:
            # 固定权重
            self.task_weights = torch.ones(len(tasks))
    
    def compute_balanced_loss(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算平衡后的总损失"""
        if self.balance_method == 'uncertainty':
            # 不确定性加权
            total_loss = 0
            for i, (task, loss) in enumerate(task_losses.items()):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]
            return total_loss
        elif self.balance_method == 'gradnorm':
            # 梯度范数平衡
            weighted_losses = []
            for i, (task, loss) in enumerate(task_losses.items()):
                weighted_losses.append(self.task_weights[i] * loss)
            return torch.sum(torch.stack(weighted_losses))
        else:
            # 固定权重
            return torch.sum(torch.stack(list(task_losses.values())))


class MemoryOptimizedTrainer(Trainer):
    """内存优化的训练器"""
    
    def __init__(self, model, config, train_loader, val_loader, test_loader, device=None):
        # 初始化父类
        super().__init__(model, config, train_loader, val_loader, test_loader, device)
        
        # 内存优化配置
        self.memory_config = config.get('memory_optimization', {})
        
        # 梯度检查点
        self.use_gradient_checkpointing = self.memory_config.get('gradient_checkpointing', True)
        if self.use_gradient_checkpointing:
            self.model = GradientCheckpointedModel(
                self.model, 
                checkpoint_segments=self.memory_config.get('checkpoint_segments', 4)
            )
        
        # 内存分析器
        self.memory_profiler = MemoryProfiler(self.device)
        
        # 动态损失缩放
        self.use_dynamic_scaling = self.memory_config.get('dynamic_loss_scaling', True)
        if self.use_dynamic_scaling:
            self.loss_scaler = DynamicLossScaler()
        else:
            self.loss_scaler = GradScaler()
        
        # 多任务损失平衡
        self.loss_balancer = MultiTaskLossBalancer(
            tasks=['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall'],
            balance_method=self.memory_config.get('loss_balance_method', 'uncertainty')
        )
        
        # 动态批次大小
        self.dynamic_batch_size = self.memory_config.get('dynamic_batch_size', True)
        self.max_batch_size = config['data']['batch_size']
        self.current_batch_size = self.max_batch_size
        
        # 内存阈值
        self.memory_threshold = self.memory_config.get('memory_threshold_gb', 14.0)  # 16GB显存的安全阈值
        
        logger.info(f"MemoryOptimizedTrainer initialized with config: {self.memory_config}")
    
    @contextmanager
    def memory_efficient_forward(self):
        """内存高效的前向传播上下文"""
        try:
            # 清空缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # 清理未使用的内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def adjust_batch_size(self, memory_usage: float) -> int:
        """根据内存使用情况调整批次大小"""
        if not self.dynamic_batch_size:
            return self.current_batch_size
            
        if memory_usage > self.memory_threshold:
            # 减少批次大小
            new_batch_size = max(1, int(self.current_batch_size * 0.8))
            if new_batch_size != self.current_batch_size:
                logger.info(f"Reducing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
        elif memory_usage < self.memory_threshold * 0.7:
            # 增加批次大小
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            if new_batch_size != self.current_batch_size:
                logger.info(f"Increasing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
        
        return self.current_batch_size
    
    def _train_epoch(self, epoch):
        """内存优化的训练epoch"""
        self.model.train()
        total_loss = 0
        task_losses = {
            'lip_sync': 0, 'expression': 0, 'audio_quality': 0, 
            'cross_modal': 0, 'overall': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['train']['epochs']} [Train]")
        
        for batch_idx, (features, labels, video_ids) in enumerate(pbar):
            # 记录内存使用情况
            self.memory_profiler.log_memory_usage(f"epoch_{epoch}_batch_{batch_idx}_start")
            
            # 动态调整批次大小
            current_memory = self.memory_profiler.get_memory_usage()['allocated']
            adjusted_batch_size = self.adjust_batch_size(current_memory)
            
            # 如果批次大小为1，使用特殊处理
            if adjusted_batch_size == 1:
                loss, batch_task_losses = self._process_single_batch(features, labels)
            else:
                loss, batch_task_losses = self._process_batch(features, labels)
            
            # 更新统计信息
            total_loss += loss.item()
            for task in task_losses:
                task_losses[task] += batch_task_losses[task]
            
            # 记录内存使用情况
            self.memory_profiler.log_memory_usage(f"epoch_{epoch}_batch_{batch_idx}_end")
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'batch_size': adjusted_batch_size,
                'memory_gb': f"{current_memory:.2f}"
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def _process_single_batch(self, features: Dict, labels: Dict) -> Tuple[torch.Tensor, Dict]:
        """处理单批次数据（内存优化版本）"""
        with self.memory_efficient_forward():
            # 将数据移动到设备
            visual_features = torch.nan_to_num(features['visual'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            audio_features = torch.nan_to_num(features['audio'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            keypoint_features = torch.nan_to_num(features['keypoint'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            au_features = torch.nan_to_num(features['au'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 保持为一维 [batch]
            targets = {
                'lip_sync': labels['lip_sync'].to(self.device).view(-1),
                'expression': labels['expression'].to(self.device).view(-1),
                'audio_quality': labels['audio_quality'].to(self.device).view(-1),
                'cross_modal': labels['cross_modal'].to(self.device).view(-1),
                'overall': labels['overall'].to(self.device).view(-1)
            }
            
            # 混合精度前向传播
            with autocast():
                outputs = self.model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoint_features=keypoint_features,
                    au_features=au_features
                )
                
                # 计算损失（只对有效标签）
                task_losses = {}
                for task, target_tensor in targets.items():
                    mask = target_tensor != -1.0
                    if mask.any():
                        task_losses[task] = self.criterion(outputs[task][mask], target_tensor[mask])
                    else:
                        task_losses[task] = torch.tensor(0.0, device=self.device)
                
                # 使用损失平衡器
                loss = self.loss_balancer.compute_balanced_loss(task_losses)
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.use_dynamic_scaling:
                scaled_loss = self.loss_scaler.scale * loss
                scaled_loss.backward()
                
                # 检查梯度溢出
                has_overflow = False
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_overflow = True
                        break
                
                # 更新损失缩放器
                self.loss_scaler.update(has_overflow)
                
                if not has_overflow:
                    # 反缩放梯度并更新
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data /= self.loss_scaler.scale
                    self.optimizer.step()
            else:
                loss.backward()
                self.optimizer.step()
            
            return loss, task_losses
    
    def _process_batch(self, features: Dict, labels: Dict) -> Tuple[torch.Tensor, Dict]:
        """处理批次数据（标准处理）"""
        return self._process_single_batch(features, labels)
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点（包含内存优化状态）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'memory_config': self.memory_config,
            'loss_scaler_state': self.loss_scaler.state_dict() if hasattr(self.loss_scaler, 'state_dict') else None,
            'memory_stats': self.memory_profiler.memory_stats
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = (
            os.path.join(self.checkpoint_dir, 'best_model_optimized.pth')
            if is_best else os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}_optimized.pth')
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Memory-optimized checkpoint saved to {checkpoint_path}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存使用报告"""
        return {
            'memory_stats': self.memory_profiler.memory_stats,
            'peak_memory_gb': max(stat['allocated'] for stat in self.memory_profiler.memory_stats) if self.memory_profiler.memory_stats else 0,
            'average_memory_gb': sum(stat['allocated'] for stat in self.memory_profiler.memory_stats) / len(self.memory_profiler.memory_stats) if self.memory_profiler.memory_stats else 0,
            'optimization_enabled': {
                'gradient_checkpointing': self.use_gradient_checkpointing,
                'dynamic_loss_scaling': self.use_dynamic_scaling,
                'dynamic_batch_size': self.dynamic_batch_size,
                'loss_balancing': self.loss_balancer.balance_method
            }
        }