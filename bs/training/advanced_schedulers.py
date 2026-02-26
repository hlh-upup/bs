#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级学习率调度器

实现2024年最新的学习率调度策略：
1. 自适应余弦退火（Adaptive Cosine Annealing）
2. 内存感知调度（Memory-Aware Scheduling）
3. 多任务学习率调度（Multi-Task LR Scheduling）
4. 早停与恢复（Early Stopping with Recovery）
5. 梯度信息感知的调度（Gradient-Aware Scheduling）
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Any, Tuple
import logging
import math
from collections import deque
import json

logger = logging.getLogger(__name__)


class AdaptiveCosineAnnealingLR(_LRScheduler):
    """自适应余弦退火学习率调度器
    
    根据训练进度和验证指标动态调整退火参数
    """
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0, 
                 adapt_factor: float = 0.1, metric_patience: int = 5,
                 last_epoch: int = -1, verbose: bool = False):
        """
        Args:
            optimizer: 优化器
            T_max: 最大迭代次数
            eta_min: 最小学习率
            adapt_factor: 自适应调整因子
            metric_patience: 指标耐心值
            last_epoch: 上一个epoch
            verbose: 是否输出详细信息
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.adapt_factor = adapt_factor
        self.metric_patience = metric_patience
        
        # 性能跟踪
        self.metric_history = deque(maxlen=metric_patience)
        self.base_T_max = T_max
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """计算学习率"""
        if not self._get_lr_called_within_step:
            logger.warning("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.")
        
        # 标准余弦退火
        cos_term = math.cos(math.pi * self.last_epoch / self.T_max)
        lrs = [self.eta_min + (base_lr - self.eta_min) * (1 + cos_term) / 2
               for base_lr in self.base_lrs]
        
        return lrs
    
    def step(self, metrics: Optional[float] = None, epoch: Optional[int] = None):
        """执行调度步骤"""
        if metrics is not None:
            self.metric_history.append(metrics)
            
            # 如果指标没有改善，延长退火周期
            if len(self.metric_history) >= self.metric_patience:
                recent_improvement = self._check_improvement()
                if not recent_improvement:
                    self.T_max = int(self.T_max * (1 + self.adapt_factor))
                    logger.info(f"Extended T_max to {self.T_max} due to plateau")
        
        super().step(epoch)
    
    def _check_improvement(self) -> bool:
        """检查近期是否有改善"""
        if len(self.metric_history) < 2:
            return True
        
        # 计算近期指标的变化趋势
        recent_values = list(self.metric_history)[-self.metric_patience:]
        best_value = min(recent_values)
        
        # 如果最近的值不是最好的，认为没有改善
        return recent_values[-1] <= best_value * 1.01


class MultiTaskLRScheduler:
    """多任务学习率调度器
    
    为不同任务维护独立的学习率调度
    """
    
    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer], 
                 scheduler_configs: Dict[str, Dict], tasks: List[str]):
        """
        Args:
            optimizers: 任务优化器字典
            scheduler_configs: 调度器配置字典
            tasks: 任务列表
        """
        self.tasks = tasks
        self.schedulers = {}
        
        # 为每个任务创建调度器
        for task in tasks:
            if task in optimizers:
                config = scheduler_configs.get(task, scheduler_configs.get('default', {}))
                self.schedulers[task] = self._create_scheduler(
                    optimizers[task], config
                )
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer, 
                         config: Dict) -> _LRScheduler:
        """创建单个调度器"""
        scheduler_type = config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('T_max', 100),
                eta_min=config.get('eta_min', 0)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'min'),
                factor=config.get('factor', 0.1),
                patience=config.get('patience', 10)
            )
        else:
            # 默认使用余弦退火
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=0
            )
    
    def step(self, task_metrics: Optional[Dict[str, float]] = None, 
             epoch: Optional[int] = None):
        """执行所有调度器的步骤"""
        for task, scheduler in self.schedulers.items():
            if task_metrics and task in task_metrics:
                # 如果调度器支持指标输入
                if hasattr(scheduler, 'step') and callable(getattr(scheduler, 'step')):
                    try:
                        scheduler.step(task_metrics[task], epoch)
                    except TypeError:
                        scheduler.step(epoch)
            else:
                scheduler.step(epoch)
    
    def get_lr(self, task: str) -> float:
        """获取指定任务的学习率"""
        if task in self.schedulers:
            return self.schedulers[task].get_last_lr()[0]
        return 0.0
    
    def get_all_lrs(self) -> Dict[str, float]:
        """获取所有任务的学习率"""
        lrs = {}
        for task, scheduler in self.schedulers.items():
            lrs[task] = scheduler.get_last_lr()[0]
        return lrs


class MemoryAwareScheduler:
    """内存感知的学习率调度器
    
    根据GPU内存使用情况动态调整学习率
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 base_lr: float, memory_threshold: float = 0.85,
                 reduction_factor: float = 0.5, recovery_patience: int = 5):
        """
        Args:
            optimizer: 优化器
            base_lr: 基础学习率
            memory_threshold: 内存使用率阈值
            reduction_factor: 学习率减少因子
            recovery_patience: 恢复耐心值
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.memory_threshold = memory_threshold
        self.reduction_factor = reduction_factor
        self.recovery_patience = recovery_patience
        
        self.current_lr = base_lr
        self.memory_high_counter = 0
        self.memory_low_counter = 0
        self.memory_history = deque(maxlen=10)
        
        # 设置初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
    
    def step(self, memory_usage: float, epoch: int = None):
        """根据内存使用情况调整学习率"""
        self.memory_history.append(memory_usage)
        
        # 计算平均内存使用率
        avg_memory = np.mean(list(self.memory_history))
        
        if avg_memory > self.memory_threshold:
            self.memory_high_counter += 1
            self.memory_low_counter = 0
            
            # 如果持续高内存使用，降低学习率
            if self.memory_high_counter >= 3:
                self.current_lr *= self.reduction_factor
                logger.info(f"High memory usage ({avg_memory:.2%}), "
                           f"reducing LR to {self.current_lr:.6f}")
                self._update_lr()
                self.memory_high_counter = 0
        else:
            self.memory_low_counter += 1
            self.memory_high_counter = 0
            
            # 如果内存使用正常，尝试恢复学习率
            if (self.memory_low_counter >= self.recovery_patience and 
                self.current_lr < self.base_lr):
                self.current_lr = min(
                    self.base_lr, 
                    self.current_lr / self.reduction_factor
                )
                logger.info(f"Memory usage normal ({avg_memory:.2%}), "
                           f"recovering LR to {self.current_lr:.6f}")
                self._update_lr()
                self.memory_low_counter = 0
    
    def _update_lr(self):
        """更新优化器学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr


class GradientAwareScheduler:
    """梯度信息感知的学习率调度器
    
    根据梯度信息（如梯度范数、梯度噪声等）调整学习率
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 base_lr: float, gradient_window: int = 100,
                 noise_threshold: float = 1.0, plateau_patience: int = 10):
        """
        Args:
            optimizer: 优化器
            base_lr: 基础学习率
            gradient_window: 梯度统计窗口大小
            noise_threshold: 梯度噪声阈值
            plateau_patience: 平台期耐心值
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.gradient_window = gradient_window
        self.noise_threshold = noise_threshold
        self.plateau_patience = plateau_patience
        
        self.current_lr = base_lr
        self.gradient_norms = deque(maxlen=gradient_window)
        self.gradient_variances = deque(maxlen=gradient_window)
        self.plateau_counter = 0
        
        # 设置初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr
    
    def step(self, gradients: Optional[List[torch.Tensor]] = None, 
             loss: Optional[float] = None, epoch: int = None):
        """根据梯度信息调整学习率"""
        if gradients is not None:
            # 计算梯度范数
            total_norm = 0
            for grad in gradients:
                if grad is not None:
                    total_norm += grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            self.gradient_norms.append(total_norm)
            
            # 计算梯度方差（噪声指标）
            if len(self.gradient_norms) >= 10:
                recent_norms = list(self.gradient_norms)[-10:]
                variance = np.var(recent_norms)
                self.gradient_variances.append(variance)
                
                # 如果梯度噪声过大，降低学习率
                if len(self.gradient_variances) > 0:
                    avg_variance = np.mean(list(self.gradient_variances))
                    if avg_variance > self.noise_threshold:
                        self.current_lr *= 0.9
                        logger.info(f"High gradient noise (var: {avg_variance:.4f}), "
                                   f"reducing LR to {self.current_lr:.6f}")
                        self._update_lr()
        
        # 基于损失的平台期检测
        if loss is not None and len(self.gradient_norms) > 0:
            recent_norms = list(self.gradient_norms)[-self.plateau_patience:]
            
            # 如果梯度范数持续下降，可能进入平台期
            if len(recent_norms) >= self.plateau_patience:
                if all(recent_norms[i] >= recent_norms[i+1] 
                      for i in range(len(recent_norms)-1)):
                    self.plateau_counter += 1
                    
                    if self.plateau_counter >= self.plateau_patience:
                        self.current_lr *= 0.5
                        logger.info(f"Gradient plateau detected, "
                                   f"reducing LR to {self.current_lr:.6f}")
                        self._update_lr()
                        self.plateau_counter = 0
                else:
                    self.plateau_counter = 0
    
    def _update_lr(self):
        """更新优化器学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.current_lr


class EarlyStoppingWithRecovery:
    """带恢复功能的早停机制
    
    在检测到过拟合时保存模型状态，并在适当时机恢复
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 recovery_patience: int = 3, memory_aware: bool = True):
        """
        Args:
            patience: 耐心值
            min_delta: 最小改善阈值
            recovery_patience: 恢复耐心值
            memory_aware: 是否考虑内存使用
        """
        self.patience = patience
        self.min_delta = min_delta
        self.recovery_patience = recovery_patience
        self.memory_aware = memory_aware
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.best_epoch = 0
        self.recovery_counter = 0
        self.memory_threshold = 0.9  # 内存使用率阈值
        
        self.history = {
            'losses': [],
            'memory_usage': [],
            'epochs': []
        }
    
    def __call__(self, val_loss: float, model: nn.Module, 
                 memory_usage: Optional[float] = None) -> Tuple[bool, bool]:
        """
        检查是否应该早停或恢复
        
        Returns:
            (should_stop, should_recover)
        """
        self.history['losses'].append(val_loss)
        self.history['memory_usage'].append(memory_usage)
        self.history['epochs'].append(len(self.history['losses']))
        
        # 检查是否应该恢复
        should_recover = False
        if (self.best_state is not None and 
            self.recovery_counter >= self.recovery_patience):
            # 如果损失开始上升且内存使用正常，考虑恢复
            if (len(self.history['losses']) >= 3 and 
                self.history['losses'][-1] > self.history['losses'][-2] > self.history['losses'][-3]):
                if not self.memory_aware or (memory_usage and memory_usage < self.memory_threshold):
                    should_recover = True
                    self.recovery_counter = 0
                    logger.info("Triggering model recovery due to loss increase")
        
        # 检查损失改善
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.recovery_counter = 0
            
            # 保存最佳模型状态
            self.best_state = model.state_dict()
            self.best_epoch = len(self.history['losses'])
        else:
            self.counter += 1
            self.recovery_counter += 1
        
        # 检查是否应该早停
        should_stop = self.counter >= self.patience
        
        if should_stop:
            logger.info(f"Early stopping triggered at epoch {len(self.history['losses'])}")
        
        return should_stop, should_recover
    
    def recover_model(self, model: nn.Module):
        """恢复模型到最佳状态"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"Model recovered to epoch {self.best_epoch}")
    
    def get_history(self) -> Dict:
        """获取历史记录"""
        return self.history.copy()


def create_advanced_scheduler(optimizer: torch.optim.Optimizer, 
                            config: Dict[str, Any]) -> _LRScheduler:
    """创建高级调度器"""
    scheduler_type = config.get('type', 'adaptive_cosine')
    
    if scheduler_type == 'adaptive_cosine':
        return AdaptiveCosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', 100),
            eta_min=config.get('eta_min', 0),
            adapt_factor=config.get('adapt_factor', 0.1),
            metric_patience=config.get('metric_patience', 5)
        )
    elif scheduler_type == 'memory_aware':
        return MemoryAwareScheduler(
            optimizer,
            base_lr=config.get('lr', 0.001),
            memory_threshold=config.get('memory_threshold', 0.85),
            reduction_factor=config.get('reduction_factor', 0.5),
            recovery_patience=config.get('recovery_patience', 5)
        )
    elif scheduler_type == 'gradient_aware':
        return GradientAwareScheduler(
            optimizer,
            base_lr=config.get('lr', 0.001),
            gradient_window=config.get('gradient_window', 100),
            noise_threshold=config.get('noise_threshold', 1.0),
            plateau_patience=config.get('plateau_patience', 10)
        )
    else:
        # 默认使用标准余弦退火
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get('T_max', 100), eta_min=config.get('eta_min', 0)
        )