#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 训练模块

实现模型训练、验证和测试功能。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # 可选
except Exception:
    sns = None
# 优先使用 torch.utils.tensorboard，回退到 tensorboardX
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# 设置中文字体回退，避免中文缺字告警；并允许负号正常显示
try:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial'
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False
    if sns is not None:
        sns.set_theme(style='whitegrid')
        # 尝试在 seaborn 中也设置中文字体
        sns.set(font='Microsoft YaHei')
except Exception:
    pass


class Trainer:
    """模型训练器
    
    实现模型训练、验证和测试功能。
    
    Args:
        model (nn.Module): 模型
        config (dict): 配置字典
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
    """
    
    def __init__(self, model, config, train_loader, val_loader, test_loader, device=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 损失函数（若模型自带 compute_loss 则优先使用）
        self.criterion = nn.MSELoss()
        # 强制禁用模型内置 compute_loss，统一回退到按任务MSE+固定权重
        self.model_has_compute_loss = False
        # 标签范围（用于预测裁剪/映射）
        th_cfg = self.config.get('model', {}).get('task_heads', {})
        self.score_min = float(th_cfg.get('score_min', 1.0))
        self.score_max = float(th_cfg.get('score_max', 5.0))
        
        # 任务权重
        self.task_weights = {
            'lip_sync': config['model']['task_weights']['lip_sync'],
            'expression': config['model']['task_weights']['expression'],
            'audio_quality': config['model']['task_weights']['audio_quality'],
            'cross_modal': config['model']['task_weights']['cross_modal'],
            'overall': config['model']['task_weights']['overall']
        }
        
        # 创建输出目录
        self.output_dir = config['train']['output_dir']
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.result_dir = os.path.join(self.output_dir, 'results')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # 训练曲线历史：记录每个 epoch 的损失与指标，便于最终绘制曲线图
        self._tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_task_losses': {t: [] for t in self._tasks},
            'val_task_losses': {t: [] for t in self._tasks},
            'val_rmse': {t: [] for t in self._tasks},
            'val_r2': {t: [] for t in self._tasks},
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")

        # 分数区间配置（用于裁剪与映射）
        th_cfg = self.config.get('model', {}).get('task_heads', {})
        try:
            self.score_min = float(th_cfg.get('score_min', 1.0))
            self.score_max = float(th_cfg.get('score_max', 5.0))
        except Exception:
            self.score_min, self.score_max = 1.0, 5.0
        if self.score_max <= self.score_min:
            self.score_min, self.score_max = 0.0, 1.0
        self._enable_clamp = bool(th_cfg.get('clamp_output', True))
        # 输出激活策略：支持 'sigmoid' / 'tanh' / 'none'（或缺省）
        self._out_activation = str(th_cfg.get('out_activation', 'none')).lower()

        # 可选：是否对输入做批内标准化（默认关闭，避免与既有流程冲突）
        train_cfg = self.config.get('train', {})
        self._batch_norm_inputs = bool(train_cfg.get('batch_norm_inputs', False))

    def _apply_output_activation(self, outputs: dict) -> dict:
        """将模型原始输出映射到 [score_min, score_max] 区间。

        - sigmoid: y = smin + sigmoid(raw) * (smax - smin)
        - tanh:    y = smin + (tanh(raw)*0.5 + 0.5) * (smax - smin)
        - none:    不改动
        """
        if self._out_activation not in {'sigmoid', 'tanh'}:
            return outputs
        smin, smax = self.score_min, self.score_max
        rng = smax - smin
        act = self._out_activation
        mapped = {}
        for k, v in outputs.items():
            if not isinstance(v, torch.Tensor):
                mapped[k] = v
                continue
            x = v
            # 压缩 [B,1] -> [B]
            if x.dim() == 2 and x.size(1) == 1:
                x = x.squeeze(1)
            if act == 'sigmoid':
                x = torch.sigmoid(x) * rng + smin
            elif act == 'tanh':
                x = (torch.tanh(x) * 0.5 + 0.5) * rng + smin
            mapped[k] = x
        return mapped

    def _maybe_normalize_inputs(self, visual, audio, keypoint, au):
        """按批次对输入做简单标准化（可选）。
        以(B,T,D)或(B,D)张量为输入，对最后一维做均值/方差标准化。
        """
        if not self._batch_norm_inputs:
            return visual, audio, keypoint, au
        def _norm(t: torch.Tensor):
            if t is None:
                return None
            # 统一到 (N, D) 再标准化
            if t.dim() == 3:
                n, tlen, d = t.shape
                flat = t.reshape(n * tlen, d)
            else:
                flat = t
            mean = flat.mean(dim=0, keepdim=True)
            std = flat.std(dim=0, keepdim=True)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            flat = (flat - mean) / std
            if t.dim() == 3:
                flat = flat.reshape(n, tlen, d)
            return flat
        return _norm(visual), _norm(audio), _norm(keypoint), _norm(au)
    
    def _create_optimizer(self):
        """创建优化器
        
        Returns:
            torch.optim.Optimizer: 优化器
        """
        optimizer_config = self.config['train']['optimizer']
        optimizer_type = optimizer_config['type']
        lr = optimizer_config['lr']
        weight_decay = optimizer_config.get('weight_decay', 0)
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: 学习率调度器
        """
        scheduler_config = self.config['train'].get('scheduler', {'type': 'none'})
        scheduler_type = scheduler_config['type']
        # 调试日志: 输出 scheduler 配置及类型信息
        try:
            logger.debug(f"Scheduler config raw: {scheduler_config}")
            for k, v in scheduler_config.items():
                logger.debug(f"  - {k}: {v} (type={type(v)})")
        except Exception:
            pass
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', self.config['train']['epochs'])
            eta_min = scheduler_config.get('eta_min', 0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'cosine_with_warmup':
            """余弦退火 + 线性 warmup

            支持字段:
              warmup_steps: 线性升温步数 (默认总步数的 10%)
              T_max: 余弦周期（若未提供则使用 epochs）
              eta_min: 最低学习率 (默认 0)
              warmup_ratio: (可选) 代替 warmup_steps 指定比例
              by_epoch: (bool) 若为 True 则以 epoch 为单位, 否则以 iteration 为单位 (默认 False)
            """
            try:
                from transformers import get_cosine_schedule_with_warmup  # type: ignore
                use_transformers = True
            except Exception:
                use_transformers = False
            total_epochs = self.config['train']['epochs']
            by_epoch = scheduler_config.get('by_epoch', False)
            eta_min = scheduler_config.get('eta_min', 0.0)
            T_max = scheduler_config.get('T_max', total_epochs)

            if by_epoch:
                total_steps = total_epochs
            else:
                # 以 iteration 为单位: epoch * batches
                total_steps = max(1, total_epochs * len(self.train_loader))

            if 'warmup_ratio' in scheduler_config:
                warmup_steps = int(total_steps * float(scheduler_config['warmup_ratio']))
            else:
                warmup_steps = int(scheduler_config.get('warmup_steps', max(1, 0.1 * total_steps)))

            if use_transformers and not by_epoch:
                # 仅在以 iteration 计数时使用 transformers 版本
                scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                # transformers 的实现没有 eta_min, 需要包装一下在 step 结束后调整 (可选)
                # 强制类型转换，避免字符串数字比较
                eta_min = float(eta_min) if isinstance(eta_min, (int, float, str)) else 0.0
                if eta_min > 0:
                    class LrClampWrapper:
                        def __init__(self, inner, optimizer, eta_min):
                            self.inner = inner
                            self.optimizer = optimizer
                            self.eta_min = eta_min
                        def step(self, *args, **kwargs):
                            self.inner.step(*args, **kwargs)
                            for pg in self.optimizer.param_groups:
                                if pg['lr'] < self.eta_min:
                                    pg['lr'] = self.eta_min
                        def __getattr__(self, item):
                            return getattr(self.inner, item)
                    scheduler = LrClampWrapper(scheduler, self.optimizer, eta_min)
                return scheduler
            else:
                # 自定义实现: 支持 epoch 或 iteration 模式
                class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
                    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0, last_epoch=-1):
                        self.warmup_steps = max(1, warmup_steps)
                        self.total_steps = max(self.warmup_steps + 1, total_steps)
                        self.eta_min = eta_min
                        super().__init__(optimizer, last_epoch)

                    def get_lr(self):
                        step = self.last_epoch + 1
                        lrs = []
                        for base_lr in self.base_lrs:
                            if step <= self.warmup_steps:
                                lr = base_lr * step / self.warmup_steps
                            else:
                                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                                # 余弦部分
                                lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                            lrs.append(lr)
                        return lrs

                import math
                total_for_scheduler = T_max if by_epoch else total_steps
                return WarmupCosineScheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=total_for_scheduler, eta_min=eta_min)
        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 5)
            factor = scheduler_config.get('factor', 0.1)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=patience, factor=factor)
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def train(self):
        """训练模型
        
        Returns:
            dict: 训练结果
        """
        epochs = self.config['train']['epochs']
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss, train_task_losses = self._train_epoch(epoch)
            
            # 验证
            val_loss, val_task_losses, val_metrics = self._validate(epoch)
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录日志
            self._log_epoch(epoch, train_loss, train_task_losses, val_loss, val_task_losses, val_metrics)
            
            # 保存模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)
            
            if epoch % self.config['train']['save_interval'] == 0:
                self._save_checkpoint(epoch)
        
        # 训练结束后测试最佳模型
        self._load_checkpoint(self.best_epoch, is_best=True)
        test_loss, test_task_losses, test_metrics = self.test()
        
        # 记录最终结果
        final_results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'test_loss': test_loss,
            'test_task_losses': test_task_losses,
            'test_metrics': test_metrics
        }
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Test metrics: {test_metrics}")

        # 训练完成后，保存训练曲线图到结果目录
        try:
            self._save_training_curves()
        except Exception as e:
            logger.warning(f"Failed to save training curves: {e}")
        
        # 关闭TensorBoard日志
        self.writer.close()
        
        return final_results
    
    def _train_epoch(self, epoch):
        """训练一个epoch
        
        Args:
            epoch (int): 当前epoch
        
        Returns:
            tuple: (总损失, 任务损失字典)
        """
        self.model.train()
        total_loss = 0
        task_losses = {
            'lip_sync': 0,
            'expression': 0,
            'audio_quality': 0,
            'cross_modal': 0,
            'overall': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['train']['epochs']} [Train]")
        for batch_idx, (features, labels, video_ids) in enumerate(pbar):
            # 将数据移动到设备
            visual_features = torch.nan_to_num(features['visual'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            audio_features = torch.nan_to_num(features['audio'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            keypoint_features = torch.nan_to_num(features['keypoint'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
            au_features = torch.nan_to_num(features['au'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)

            # 可选：批内标准化（仅当配置开启时）
            visual_features, audio_features, keypoint_features, au_features = self._maybe_normalize_inputs(
                visual_features, audio_features, keypoint_features, au_features
            )
            
            # 保持为一维 [batch]，避免 batch=1 时变成 0 维标量
            lip_sync_score = labels['lip_sync'].to(self.device).view(-1)
            expression_score = labels['expression'].to(self.device).view(-1)
            audio_quality_score = labels['audio_quality'].to(self.device).view(-1)
            cross_modal_score = labels['cross_modal'].to(self.device).view(-1)
            overall_score = labels['overall'].to(self.device).view(-1)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                visual_features=visual_features,
                audio_features=audio_features,
                keypoint_features=keypoint_features,
                au_features=au_features
            )
            # 先做输出激活映射（若配置指定），再裁剪兜底
            outputs = self._apply_output_activation(outputs)
            # 裁剪预测，避免越界影响损失与稳定性
            if self._enable_clamp:
                for k in outputs.keys():
                    t = outputs[k]
                    if t.dim() == 2 and t.size(1) == 1:
                        t = t.squeeze(1)
                    outputs[k] = torch.clamp(t, min=self.score_min, max=self.score_max)
            # 裁剪预测到标签范围，避免越界影响稳定性
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = torch.clamp(v, min=self.score_min, max=self.score_max)
            
            # 计算损失：统一使用逐任务MSE + 固定权重
            losses = {}
            for task, target_tensor in {
                'lip_sync': lip_sync_score,
                'expression': expression_score,
                'audio_quality': audio_quality_score,
                'cross_modal': cross_modal_score,
                'overall': overall_score
            }.items():
                mask = target_tensor != -1.0
                if mask.any():
                    pred_task = outputs[task]
                    if pred_task.dim() == 2 and pred_task.size(1) == 1:  # 压缩 [B,1] -> [B]
                        pred_task = pred_task.squeeze(1)
                    losses[task] = self.criterion(pred_task[mask], target_tensor[mask])
                else:
                    losses[task] = torch.tensor(0.0, device=self.device)
            lip_sync_loss = losses['lip_sync']
            expression_loss = losses['expression']
            audio_quality_loss = losses['audio_quality']
            cross_modal_loss = losses['cross_modal']
            overall_loss = losses['overall']
            # 固定权重聚合
            loss = (
                self.task_weights['lip_sync'] * lip_sync_loss +
                self.task_weights['expression'] * expression_loss +
                self.task_weights['audio_quality'] * audio_quality_loss +
                self.task_weights['cross_modal'] * cross_modal_loss +
                self.task_weights['overall'] * overall_loss
            )
            
            # 反向传播
            loss.backward()
            # 梯度裁剪，防止爆梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            task_losses['lip_sync'] += lip_sync_loss.item()
            task_losses['expression'] += expression_loss.item()
            task_losses['audio_quality'] += audio_quality_loss.item()
            task_losses['cross_modal'] += cross_modal_loss.item()
            task_losses['overall'] += overall_loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def _validate(self, epoch):
        """验证模型
        
        Args:
            epoch (int): 当前epoch
        
        Returns:
            tuple: (总损失, 任务损失字典, 评估指标字典)
        """
        self.model.eval()
        total_loss = 0
        task_losses = {
            'lip_sync': 0,
            'expression': 0,
            'audio_quality': 0,
            'cross_modal': 0,
            'overall': 0
        }
        
        # 收集预测和真实值
        all_preds = {
            'lip_sync': [],
            'expression': [],
            'audio_quality': [],
            'cross_modal': [],
            'overall': []
        }
        all_targets = {
            'lip_sync': [],
            'expression': [],
            'audio_quality': [],
            'cross_modal': [],
            'overall': []
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['train']['epochs']} [Val]")
            for batch_idx, (features, labels, video_ids) in enumerate(pbar):
                # 将数据移动到设备
                visual_features = torch.nan_to_num(features['visual'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                audio_features = torch.nan_to_num(features['audio'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                keypoint_features = torch.nan_to_num(features['keypoint'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                au_features = torch.nan_to_num(features['au'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 保持为一维 [batch]，避免 batch=1 时变成 0 维标量
                lip_sync_score = labels['lip_sync'].to(self.device).view(-1)
                expression_score = labels['expression'].to(self.device).view(-1)
                audio_quality_score = labels['audio_quality'].to(self.device).view(-1)
                cross_modal_score = labels['cross_modal'].to(self.device).view(-1)
                overall_score = labels['overall'].to(self.device).view(-1)
                
                # 前向传播
                outputs = self.model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoint_features=keypoint_features,
                    au_features=au_features
                )
                outputs = self._apply_output_activation(outputs)
                if self._enable_clamp:
                    for k in outputs.keys():
                        t = outputs[k]
                        if t.dim() == 2 and t.size(1) == 1:
                            t = t.squeeze(1)
                        outputs[k] = torch.clamp(t, min=self.score_min, max=self.score_max)
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        outputs[k] = torch.clamp(v, min=self.score_min, max=self.score_max)
                
                # 计算损失：与训练一致，统一使用逐任务MSE + 固定权重
                losses = {}
                for task, target_tensor in {
                    'lip_sync': lip_sync_score,
                    'expression': expression_score,
                    'audio_quality': audio_quality_score,
                    'cross_modal': cross_modal_score,
                    'overall': overall_score
                }.items():
                    mask = target_tensor != -1.0
                    if mask.any():
                        pred_task = outputs[task]
                        if pred_task.dim() == 2 and pred_task.size(1) == 1:
                            pred_task = pred_task.squeeze(1)
                        losses[task] = self.criterion(pred_task[mask], target_tensor[mask])
                    else:
                        losses[task] = torch.tensor(0.0, device=self.device)
                lip_sync_loss = losses['lip_sync']
                expression_loss = losses['expression']
                audio_quality_loss = losses['audio_quality']
                cross_modal_loss = losses['cross_modal']
                overall_loss = losses['overall']
                loss = (
                    self.task_weights['lip_sync'] * lip_sync_loss +
                    self.task_weights['expression'] * expression_loss +
                    self.task_weights['audio_quality'] * audio_quality_loss +
                    self.task_weights['cross_modal'] * cross_modal_loss +
                    self.task_weights['overall'] * overall_loss
                )
                
                # 更新统计信息
                total_loss += loss.item()
                task_losses['lip_sync'] += lip_sync_loss.item()
                task_losses['expression'] += expression_loss.item()
                task_losses['audio_quality'] += audio_quality_loss.item()
                task_losses['cross_modal'] += cross_modal_loss.item()
                task_losses['overall'] += overall_loss.item()
                
                # 收集预测和真实值 (统一展平为 1D，避免后续 concatenate 维度不匹配)
                def _flat(t: torch.Tensor):
                    if t.dim() == 2 and t.size(1) == 1:
                        t = t.squeeze(1)
                    return t.detach().cpu().view(-1).numpy()
                all_preds['lip_sync'].append(_flat(outputs['lip_sync']))
                all_preds['expression'].append(_flat(outputs['expression']))
                all_preds['audio_quality'].append(_flat(outputs['audio_quality']))
                all_preds['cross_modal'].append(_flat(outputs['cross_modal']))
                all_preds['overall'].append(_flat(outputs['overall']))

                all_targets['lip_sync'].append(_flat(lip_sync_score))
                all_targets['expression'].append(_flat(expression_score))
                all_targets['audio_quality'].append(_flat(audio_quality_score))
                all_targets['cross_modal'].append(_flat(cross_modal_score))
                all_targets['overall'].append(_flat(overall_score))
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        # 合并预测和真实值
        for task in all_preds.keys():
            all_preds[task] = np.concatenate(all_preds[task])
            all_targets[task] = np.concatenate(all_targets[task])
        try:
            rng = {t: (float(np.min(all_preds[t])), float(np.max(all_preds[t]))) for t in all_preds}
            logger.info(f"[Val] prediction ranges: " + ", ".join([f"{k}:[{v[0]:.2f},{v[1]:.2f}]" for k,v in rng.items()]))
        except Exception:
            pass
        
        # 计算评估指标（只对有效标签计算）
        metrics = {}
        for task in all_preds.keys():
            mask = all_targets[task] != -1.0
            if np.any(mask):
                metrics[task] = {
                    'mse': mean_squared_error(all_targets[task][mask], all_preds[task][mask]),
                    'rmse': np.sqrt(mean_squared_error(all_targets[task][mask], all_preds[task][mask])),
                    'mae': mean_absolute_error(all_targets[task][mask], all_preds[task][mask]),
                    'r2': r2_score(all_targets[task][mask], all_preds[task][mask])
                }
            else:
                metrics[task] = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
        
        # 可视化预测结果（仅对整体评分；仅在配置显式提供 vis_interval 时启用）
        vis_interval = self.config.get('train', {}).get('vis_interval', None)
        if vis_interval is not None and vis_interval > 0 and epoch % vis_interval == 0:
            self._visualize_predictions(epoch, all_preds['overall'], all_targets['overall'], 'val')
        
        # 保存各任务图片与误差直方图
        try:
            for task in all_preds.keys():
                self._visualize_task_predictions(epoch, task, all_preds[task], all_targets[task], 'val')
                self._visualize_task_error_hist(epoch, task, all_preds[task], all_targets[task], 'val')
            self._visualize_metrics_bar(epoch, metrics, 'val')
        except Exception as e:
            logger.warning(f"Failed to save per-task val plots: {e}")

        return avg_loss, avg_task_losses, metrics
    
    def test(self):
        """测试模型
        
        Returns:
            tuple: (总损失, 任务损失字典, 评估指标字典)
        """
        self.model.eval()
        total_loss = 0
        task_losses = {
            'lip_sync': 0,
            'expression': 0,
            'audio_quality': 0,
            'cross_modal': 0,
            'overall': 0
        }
        
        # 收集预测和真实值
        all_preds = {
            'lip_sync': [],
            'expression': [],
            'audio_quality': [],
            'cross_modal': [],
            'overall': []
        }
        all_targets = {
            'lip_sync': [],
            'expression': [],
            'audio_quality': [],
            'cross_modal': [],
            'overall': []
        }
        
        # 收集视频ID
        video_ids = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for batch_idx, (features, labels, batch_video_ids) in enumerate(pbar):
                # 收集视频ID
                video_ids.extend(batch_video_ids)
                
                # 将数据移动到设备
                visual_features = torch.nan_to_num(features['visual'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                audio_features = torch.nan_to_num(features['audio'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                keypoint_features = torch.nan_to_num(features['keypoint'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                au_features = torch.nan_to_num(features['au'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 保持为一维 [batch]，避免 batch=1 时变成 0 维标量
                lip_sync_score = labels['lip_sync'].to(self.device).view(-1)
                expression_score = labels['expression'].to(self.device).view(-1)
                audio_quality_score = labels['audio_quality'].to(self.device).view(-1)
                cross_modal_score = labels['cross_modal'].to(self.device).view(-1)
                overall_score = labels['overall'].to(self.device).view(-1)
                
                # 前向传播
                outputs = self.model(
                    visual_features=visual_features,
                    audio_features=audio_features,
                    keypoint_features=keypoint_features,
                    au_features=au_features
                )
                outputs = self._apply_output_activation(outputs)
                if self._enable_clamp:
                    for k in outputs.keys():
                        t = outputs[k]
                        if t.dim() == 2 and t.size(1) == 1:
                            t = t.squeeze(1)
                        outputs[k] = torch.clamp(t, min=self.score_min, max=self.score_max)
                for k, v in outputs.items():
                    if isinstance(v, torch.Tensor):
                        outputs[k] = torch.clamp(v, min=self.score_min, max=self.score_max)
                
                # 计算损失：与训练一致，统一使用逐任务MSE + 固定权重
                losses = {}
                for task, target_tensor in {
                    'lip_sync': lip_sync_score,
                    'expression': expression_score,
                    'audio_quality': audio_quality_score,
                    'cross_modal': cross_modal_score,
                    'overall': overall_score
                }.items():
                    mask = target_tensor != -1.0
                    if mask.any():
                        pred_task = outputs[task]
                        if pred_task.dim() == 2 and pred_task.size(1) == 1:
                            pred_task = pred_task.squeeze(1)
                        losses[task] = self.criterion(pred_task[mask], target_tensor[mask])
                    else:
                        losses[task] = torch.tensor(0.0, device=self.device)
                lip_sync_loss = losses['lip_sync']
                expression_loss = losses['expression']
                audio_quality_loss = losses['audio_quality']
                cross_modal_loss = losses['cross_modal']
                overall_loss = losses['overall']
                loss = (
                    self.task_weights['lip_sync'] * lip_sync_loss +
                    self.task_weights['expression'] * expression_loss +
                    self.task_weights['audio_quality'] * audio_quality_loss +
                    self.task_weights['cross_modal'] * cross_modal_loss +
                    self.task_weights['overall'] * overall_loss
                )
                
                # 更新统计信息
                total_loss += loss.item()
                task_losses['lip_sync'] += lip_sync_loss.item()
                task_losses['expression'] += expression_loss.item()
                task_losses['audio_quality'] += audio_quality_loss.item()
                task_losses['cross_modal'] += cross_modal_loss.item()
                task_losses['overall'] += overall_loss.item()
                
                # 收集预测和真实值 (统一展平为 1D)
                def _flat(t: torch.Tensor):
                    if t.dim() == 2 and t.size(1) == 1:
                        t = t.squeeze(1)
                    return t.detach().cpu().view(-1).numpy()
                all_preds['lip_sync'].append(_flat(outputs['lip_sync']))
                all_preds['expression'].append(_flat(outputs['expression']))
                all_preds['audio_quality'].append(_flat(outputs['audio_quality']))
                all_preds['cross_modal'].append(_flat(outputs['cross_modal']))
                all_preds['overall'].append(_flat(outputs['overall']))

                all_targets['lip_sync'].append(_flat(lip_sync_score))
                all_targets['expression'].append(_flat(expression_score))
                all_targets['audio_quality'].append(_flat(audio_quality_score))
                all_targets['cross_modal'].append(_flat(cross_modal_score))
                all_targets['overall'].append(_flat(overall_score))
                
                # 更新进度条
                pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        num_batches = len(self.test_loader)
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        # 合并预测和真实值
        for task in all_preds.keys():
            all_preds[task] = np.concatenate(all_preds[task])
            all_targets[task] = np.concatenate(all_targets[task])
        try:
            rng = {t: (float(np.min(all_preds[t])), float(np.max(all_preds[t]))) for t in all_preds}
            logger.info(f"[Test] prediction ranges: " + ", ".join([f"{k}:[{v[0]:.2f},{v[1]:.2f}]" for k,v in rng.items()]))
        except Exception:
            pass
        
        # 计算评估指标（只对有效标签计算）
        metrics = {}
        for task in all_preds.keys():
            mask = all_targets[task] != -1.0
            if np.any(mask):
                metrics[task] = {
                    'mse': mean_squared_error(all_targets[task][mask], all_preds[task][mask]),
                    'rmse': np.sqrt(mean_squared_error(all_targets[task][mask], all_preds[task][mask])),
                    'mae': mean_absolute_error(all_targets[task][mask], all_preds[task][mask]),
                    'r2': r2_score(all_targets[task][mask], all_preds[task][mask])
                }
            else:
                metrics[task] = {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}
        
        # 可视化预测结果（仅当配置存在 vis_interval 键时执行，以避免 KeyError）
        vis_interval = self.config.get('train', {}).get('vis_interval', None)
        if vis_interval is not None:
            self._visualize_predictions(0, all_preds['overall'], all_targets['overall'], 'test')
        
        # 保存各任务测试图片与误差直方图
        try:
            for task in all_preds.keys():
                self._visualize_task_predictions(0, task, all_preds[task], all_targets[task], 'test')
                self._visualize_task_error_hist(0, task, all_preds[task], all_targets[task], 'test')
            self._visualize_metrics_bar(0, metrics, 'test')
        except Exception as e:
            logger.warning(f"Failed to save per-task test plots: {e}")

        # 保存预测结果
        self._save_predictions(video_ids, all_preds, all_targets)
        
        return avg_loss, avg_task_losses, metrics
    
    def _log_epoch(self, epoch, train_loss, train_task_losses, val_loss, val_task_losses, val_metrics):
        """记录epoch日志
        
        Args:
            epoch (int): 当前epoch
            train_loss (float): 训练损失
            train_task_losses (dict): 训练任务损失字典
            val_loss (float): 验证损失
            val_task_losses (dict): 验证任务损失字典
            val_metrics (dict): 验证评估指标字典
        """
        # 打印日志
        logger.info(f"Epoch {epoch}/{self.config['train']['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 记录TensorBoard日志
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        
        for task, loss in train_task_losses.items():
            self.writer.add_scalar(f'Loss/train_{task}', loss, epoch)
        
        for task, loss in val_task_losses.items():
            self.writer.add_scalar(f'Loss/val_{task}', loss, epoch)
        
        for task, metrics in val_metrics.items():
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'Metrics/val_{task}_{metric_name}', metric_value, epoch)
        
        # 记录学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_rate', current_lr, epoch)

        # 累计历史（用于曲线图）
        try:
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            for t in self._tasks:
                if t in train_task_losses:
                    self.history['train_task_losses'][t].append(float(train_task_losses[t]))
                if t in val_task_losses:
                    self.history['val_task_losses'][t].append(float(val_task_losses[t]))
                if t in val_metrics:
                    self.history['val_rmse'][t].append(float(val_metrics[t].get('rmse', 0.0)))
                    self.history['val_r2'][t].append(float(val_metrics[t].get('r2', 0.0)))
        except Exception:
            pass

    def _save_training_curves(self):
        """保存训练曲线图（总损失、各任务 RMSE/R²、各任务训练/验证损失）到结果目录。"""
        if not self.history['epochs']:
            return
        epochs = self.history['epochs']
        # 1) 总损失曲线
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
            plt.plot(epochs, self.history['val_loss'], label='Val Loss', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training/Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            out = os.path.join(self.result_dir, 'training_loss_curves.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            try:
                self.writer.add_image('Curves/Loss', np.transpose(plt.imread(out), (2, 0, 1)), epochs[-1])
            except Exception:
                pass
        except Exception:
            pass

        # 2) 各任务 Val RMSE 曲线
        try:
            plt.figure(figsize=(10, 6))
            for t in self._tasks:
                vals = self.history['val_rmse'].get(t, [])
                if vals:
                    plt.plot(epochs, vals, marker='o', label=t)
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('Validation RMSE by Task')
            plt.legend()
            plt.grid(True, alpha=0.3)
            out = os.path.join(self.result_dir, 'training_val_rmse_curves.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            try:
                self.writer.add_image('Curves/Val_RMSE', np.transpose(plt.imread(out), (2, 0, 1)), epochs[-1])
            except Exception:
                pass
        except Exception:
            pass

        # 3) 各任务 Val R2 曲线
        try:
            plt.figure(figsize=(10, 6))
            for t in self._tasks:
                vals = self.history['val_r2'].get(t, [])
                if vals:
                    plt.plot(epochs, vals, marker='o', label=t)
            plt.xlabel('Epoch')
            plt.ylabel('R2')
            plt.title('Validation R2 by Task')
            plt.legend()
            plt.grid(True, alpha=0.3)
            out = os.path.join(self.result_dir, 'training_val_r2_curves.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            try:
                self.writer.add_image('Curves/Val_R2', np.transpose(plt.imread(out), (2, 0, 1)), epochs[-1])
            except Exception:
                pass
        except Exception:
            pass

        # 4) 各任务 Train/Val 损失曲线（同图对比）
        try:
            plt.figure(figsize=(11, 7))
            for t in self._tasks:
                tr = self.history['train_task_losses'].get(t, [])
                va = self.history['val_task_losses'].get(t, [])
                if tr and va:
                    plt.plot(epochs, tr, linestyle='--', marker='o', label=f'{t} Train Loss')
                    plt.plot(epochs, va, linestyle='-', marker='o', label=f'{t} Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Per-Task Train/Val Loss')
            plt.legend(ncol=2)
            plt.grid(True, alpha=0.3)
            out = os.path.join(self.result_dir, 'training_task_losses_curves.png')
            plt.savefig(out, bbox_inches='tight')
            plt.close()
            try:
                self.writer.add_image('Curves/Task_Losses', np.transpose(plt.imread(out), (2, 0, 1)), epochs[-1])
            except Exception:
                pass
        except Exception:
            pass
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点
        
        Args:
            epoch (int): 当前epoch
            is_best (bool): 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = (
            os.path.join(self.checkpoint_dir, 'best_model.pth')
            if is_best else os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        )
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, epoch, is_best=False):
        """加载检查点
        
        Args:
            epoch (int): 要加载的epoch
            is_best (bool): 是否加载最佳模型
        """
        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint {checkpoint_path} does not exist")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _visualize_predictions(self, epoch, predictions, targets, split):
        """可视化预测结果
        
        Args:
            epoch (int): 当前epoch
            predictions (np.ndarray): 预测值
            targets (np.ndarray): 真实值
            split (str): 数据集划分，可选值为 'train', 'val', 'test'
        """
        # 仅使用有效标签绘图
        mask = targets != -1.0
        if np.any(mask):
            x = targets[mask]
            y = predictions[mask]
        else:
            x = targets
            y = predictions
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5)
        if len(x) > 0:
            lo = min(x)
            hi = max(x)
            plt.plot([lo, hi], [lo, hi], 'r--')
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title(f'{split.capitalize()} Set Predictions vs Ground Truth (Epoch {epoch})')
        
        # 添加评估指标
        if len(x) > 0:
            mse = mean_squared_error(x, y)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(x, y)
            r2 = r2_score(x, y)
        else:
            mse = rmse = mae = r2 = 0.0
        
        plt.figtext(0.15, 0.85, f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}')
        
        # 保存图像
        if epoch == 0:  # 测试集
            fig_path = os.path.join(self.result_dir, f'{split}_predictions.png')
        else:
            fig_path = os.path.join(self.result_dir, f'{split}_predictions_epoch_{epoch}.png')
        
        plt.savefig(fig_path)
        plt.close()
        
        # 添加到TensorBoard
        self.writer.add_image(f'Predictions/{split}', np.transpose(plt.imread(fig_path), (2, 0, 1)), epoch)

    def _visualize_task_predictions(self, epoch, task, predictions, targets, split):
        """按任务保存散点图 (Pred vs True)
        Args:
            epoch (int): 当前 epoch（测试集可传 0）
            task (str): 任务名，如 'lip_sync'
            predictions (np.ndarray): 该任务预测
            targets (np.ndarray): 该任务真实值
            split (str): 'train' | 'val' | 'test'
        """
        mask = targets != -1.0
        if np.any(mask):
            x = targets[mask]
            y = predictions[mask]
        else:
            x = targets
            y = predictions

        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.scatterplot(x=x, y=y, alpha=0.5)
        else:
            plt.scatter(x, y, alpha=0.5)
        if len(x) > 0:
            lo = min(x)
            hi = max(x)
            plt.plot([lo, hi], [lo, hi], 'r--')
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        title_epoch = f" (Epoch {epoch})" if epoch else ""
        plt.title(f"{split.capitalize()} - {task} Pred vs True{title_epoch}")

        # 附加简单指标
        if len(x) > 0:
            mse = mean_squared_error(x, y)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(x, y)
            r2 = r2_score(x, y)
        else:
            mse = rmse = mae = r2 = 0.0
        plt.figtext(0.15, 0.85, f'MSE: {mse:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR2: {r2:.3f}')

        # 保存
        if epoch == 0:
            fig_path = os.path.join(self.result_dir, f'{split}_{task}_pred_vs_true.png')
        else:
            fig_path = os.path.join(self.result_dir, f'{split}_{task}_pred_vs_true_epoch_{epoch}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # 写入 TensorBoard
        try:
            self.writer.add_image(f'Predictions/{split}/{task}', np.transpose(plt.imread(fig_path), (2, 0, 1)), epoch)
        except Exception:
            pass

    def _visualize_task_error_hist(self, epoch, task, predictions, targets, split):
        """按任务保存误差直方图"""
        mask = targets != -1.0
        if np.any(mask):
            y_true = targets[mask]
            y_pred = predictions[mask]
        else:
            y_true = targets
            y_pred = predictions
        errors = y_pred - y_true
        plt.figure(figsize=(8, 5))
        if sns is not None:
            sns.histplot(errors, kde=True, bins=30)
        else:
            plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        title_epoch = f" (Epoch {epoch})" if epoch else ""
        plt.title(f"{split.capitalize()} - {task} Error Histogram{title_epoch}")
        if epoch == 0:
            fig_path = os.path.join(self.result_dir, f'{split}_{task}_error_hist.png')
        else:
            fig_path = os.path.join(self.result_dir, f'{split}_{task}_error_hist_epoch_{epoch}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        try:
            self.writer.add_image(f'Errors/{split}/{task}', np.transpose(plt.imread(fig_path), (2, 0, 1)), epoch)
        except Exception:
            pass

    def _visualize_metrics_bar(self, epoch, metrics: dict, split: str):
        """保存按任务的指标柱状图（MSE/MAE/R2）"""
        tasks = list(metrics.keys())
        mse_vals = [metrics[t].get('mse', 0) for t in tasks]
        mae_vals = [metrics[t].get('mae', 0) for t in tasks]
        r2_vals = [metrics[t].get('r2', 0) for t in tasks]

        # MSE/MAE
        x = np.arange(len(tasks))
        width = 0.35
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, mse_vals, width=width, label='MSE')
        plt.bar(x + width/2, mae_vals, width=width, label='MAE')
        plt.xticks(x, tasks, rotation=0)
        plt.ylabel('Value')
        title_epoch = f" (Epoch {epoch})" if epoch else ""
        plt.title(f"{split.capitalize()} - Metrics by Task{title_epoch}")
        plt.legend()
        if epoch == 0:
            fig_path = os.path.join(self.result_dir, f'{split}_metrics_bar.png')
        else:
            fig_path = os.path.join(self.result_dir, f'{split}_metrics_bar_epoch_{epoch}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        # R2 (单独一张)
        plt.figure(figsize=(10, 5))
        colors = ['#4daf4a' if r >= 0 else '#e41a1c' for r in r2_vals]
        plt.bar(tasks, r2_vals, color=colors)
        plt.ylabel('R2')
        plt.title(f"{split.capitalize()} - R2 by Task{title_epoch}")
        if epoch == 0:
            fig_path = os.path.join(self.result_dir, f'{split}_r2_bar.png')
        else:
            fig_path = os.path.join(self.result_dir, f'{split}_r2_bar_epoch_{epoch}.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
    
    def _save_predictions(self, video_ids, predictions, targets):
        """保存预测结果
        
        Args:
            video_ids (list): 视频ID列表
            predictions (dict): 预测值字典
            targets (dict): 真实值字典
        """
        results = []
        
        for i, video_id in enumerate(video_ids):
            result = {'video_id': video_id}
            
            for task in predictions.keys():
                result[f'{task}_pred'] = float(predictions[task][i])
                result[f'{task}_true'] = float(targets[task][i])
            
            results.append(result)
        
        # 保存为CSV文件
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.result_dir, 'test_predictions.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Predictions saved to {csv_path}")


def train_model(model, config, train_loader, val_loader, test_loader, device=None):
    """训练模型
    
    Args:
        model (nn.Module): 模型
        config (dict): 配置字典
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        test_loader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
    
    Returns:
        dict: 训练结果
    """
    # 创建训练器
    trainer = Trainer(model, config, train_loader, val_loader, test_loader, device)
    
    # 训练模型
    results = trainer.train()
    
    return results