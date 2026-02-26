#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 工具模块

实现日志记录、配置加载等辅助功能。
"""

import os
import yaml
import json
import logging
import random
import numpy as np
import torch
from datetime import datetime


def setup_logging(log_dir=None, log_level=logging.INFO):
    """设置日志记录
    
    Args:
        log_dir (str, optional): 日志目录
        log_level (int, optional): 日志级别
    
    Returns:
        logging.Logger: 日志记录器
    """
    # 创建日志目录
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    else:
        log_file = None
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        filename=log_file
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 添加控制台处理器到根日志记录器
    logging.getLogger().addHandler(console_handler)
    
    # 创建日志记录器
    logger = logging.getLogger("talking_face_evaluator")
    
    return logger


def load_config(config_path):
    """加载配置文件
    
    Args:
        config_path (str): 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, config_path):
    """保存配置文件
    
    Args:
        config (dict): 配置字典
        config_path (str): 配置文件路径
    """
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def set_seed(seed):
    """设置随机种子
    
    Args:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA的确定性选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cuda=True):
    """获取计算设备
    
    Args:
        use_cuda (bool, optional): 是否使用CUDA
    
    Returns:
        torch.device: 计算设备
    """
    if use_cuda and torch.cuda.is_available():
        # 对于PyTorch 1.2.8，确保使用兼容的CUDA设备
        device = torch.device('cuda')
        # 打印CUDA设备信息，帮助调试
        print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"CUDA设备能力: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU设备")
    
    return device


def count_parameters(model):
    """计算模型参数数量
    
    Args:
        model (torch.nn.Module): 模型
    
    Returns:
        int: 参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data, file_path):
    """保存JSON文件
    
    Args:
        data (dict): 数据字典
        file_path (str): 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(file_path):
    """加载JSON文件
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        dict: 数据字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_experiment_dir(base_dir, experiment_name=None):
    """创建实验目录
    
    Args:
        base_dir (str): 基础目录
        experiment_name (str, optional): 实验名称
    
    Returns:
        str: 实验目录路径
    """
    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 生成实验名称
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建实验目录
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    logs_dir = os.path.join(experiment_dir, 'logs')
    results_dir = os.path.join(experiment_dir, 'results')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return experiment_dir


def format_time(seconds):
    """格式化时间
    
    Args:
        seconds (float): 秒数
    
    Returns:
        str: 格式化后的时间字符串
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


def get_optimizer(config, model_parameters):
    """获取优化器
    
    Args:
        config (dict): 配置字典
        model_parameters: 模型参数
    
    Returns:
        torch.optim.Optimizer: 优化器
    """
    optimizer_config = config['training']['optimizer']
    optimizer_type = optimizer_config['type']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config.get('weight_decay', 0)
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def get_scheduler(config, optimizer):
    """获取学习率调度器
    
    Args:
        config (dict): 配置字典
        optimizer (torch.optim.Optimizer): 优化器
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: 学习率调度器
    """
    scheduler_config = config['training'].get('scheduler', {'type': 'none'})
    scheduler_type = scheduler_config['type']
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', config['training']['epochs'])
        eta_min = scheduler_config.get('eta_min', 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type == 'cosine_with_warmup':
        try:
            from transformers import get_cosine_schedule_with_warmup  # type: ignore
            use_transformers = True
        except Exception:
            use_transformers = False
        total_epochs = config['training']['epochs']
        eta_min = scheduler_config.get('eta_min', 0.0)
        T_max = scheduler_config.get('T_max', total_epochs)
        by_epoch = scheduler_config.get('by_epoch', False)
        if by_epoch:
            total_steps = total_epochs
        else:
            # 需要传入 DataLoader 长度才可精确，这里默认 T_max * steps_per_epoch, 如果没有则 fallback 为 epochs
            steps_per_epoch = scheduler_config.get('steps_per_epoch')
            if steps_per_epoch is None:
                steps_per_epoch = T_max  # 粗略假设
            total_steps = max(1, steps_per_epoch * total_epochs)
        if 'warmup_ratio' in scheduler_config:
            warmup_steps = int(total_steps * float(scheduler_config['warmup_ratio']))
        else:
            warmup_steps = int(scheduler_config.get('warmup_steps', max(1, 0.1 * total_steps)))
        if use_transformers and not by_epoch:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            if eta_min > 0:
                base_step = scheduler.step
                def _wrap_step(*args, **kwargs):
                    base_step(*args, **kwargs)
                    for pg in optimizer.param_groups:
                        pg['lr'] = max(pg['lr'], eta_min)
                scheduler.step = _wrap_step  # type: ignore
            return scheduler
        else:
            import math
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
                            lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                        lrs.append(lr)
                    return lrs
            total_for_scheduler = T_max if by_epoch else total_steps
            return WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_for_scheduler, eta_min=eta_min)
    elif scheduler_type == 'plateau':
        patience = scheduler_config.get('patience', 5)
        factor = scheduler_config.get('factor', 0.1)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def ensure_dir(directory):
    """确保目录存在
    
    Args:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def compute_metrics(preds, targets, metrics_list=None):
    """计算评估指标：支持 pearson, spearman, rmse, mae

    Args:
        preds (array-like): 预测值
        targets (array-like): 真实值
        metrics_list (list, optional): 要计算的指标列表，默认为 ['pearson','spearman','rmse','mae']

    Returns:
        dict: 指标名称到数值的映射
    """
    import math
    preds = np.asarray(preds).ravel()
    targets = np.asarray(targets).ravel()

    if metrics_list is None:
        metrics_list = ['pearson', 'spearman', 'rmse', 'mae']

    results = {}

    # 边界检查
    if preds.size == 0 or targets.size == 0 or preds.size != targets.size:
        for m in metrics_list:
            results[m] = float('nan')
        return results

    # 计算基础统计
    try:
        # Pearson
        if 'pearson' in metrics_list:
            if preds.size < 2:
                results['pearson'] = float('nan')
            else:
                # 使用 numpy 计算相关系数
                c = np.corrcoef(preds, targets)
                results['pearson'] = float(c[0, 1]) if c.shape == (2, 2) else float('nan')

        # Spearman
        if 'spearman' in metrics_list:
            try:
                from scipy.stats import rankdata
                p_rank = rankdata(preds)
                t_rank = rankdata(targets)
            except Exception:
                # 简单排名实现（无平均并列处理）
                p_rank = np.argsort(np.argsort(preds)).astype(float)
                t_rank = np.argsort(np.argsort(targets)).astype(float)

            if preds.size < 2:
                results['spearman'] = float('nan')
            else:
                c = np.corrcoef(p_rank, t_rank)
                results['spearman'] = float(c[0, 1]) if c.shape == (2, 2) else float('nan')

        # RMSE
        if 'rmse' in metrics_list:
            results['rmse'] = float(np.sqrt(np.mean((preds - targets) ** 2)))

        # MAE
        if 'mae' in metrics_list:
            results['mae'] = float(np.mean(np.abs(preds - targets)))

    except Exception:
        # 如果计算任何指标失败，返回 NaN
        for m in metrics_list:
            results[m] = float('nan')

    return results


def visualize_results(predictions, targets, video_ids=None, metrics=None, output_dir='results'):
    """生成并保存可视化结果（散点图、CSV、指标JSON）

    Args:
        predictions (dict): 每个任务的预测值列表（每项为多个 batch 的 numpy 数组）
        targets (dict): 每个任务的真实值列表（每项为多个 batch 的 numpy 数组）
        video_ids (list, optional): 视频ID列表，长度应等于样本数
        metrics (dict, optional): 已计算的指标字典
        output_dir (str, optional): 输出目录
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # 准备按任务保存的表格数据
    task_data = {}
    total_len = None

    for task in predictions.keys():
        try:
            preds = np.concatenate(predictions[task]) if len(predictions[task]) > 0 else np.array([])
        except Exception:
            preds = np.asarray(predictions[task]).ravel()
        try:
            tars = np.concatenate(targets[task]) if len(targets[task]) > 0 else np.array([])
        except Exception:
            tars = np.asarray(targets[task]).ravel()

        # 保证为一维数组
        preds = np.asarray(preds).ravel()
        tars = np.asarray(tars).ravel()

        task_data[task] = {'preds': preds, 'targets': tars}

        if total_len is None:
            total_len = max(len(preds), len(tars))

        # 绘制散点图
        try:
            if preds.size > 0 and tars.size > 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(tars, preds, alpha=0.6)
                mn = min(float(tars.min()), float(preds.min()))
                mx = max(float(tars.max()), float(preds.max()))
                ax.plot([mn, mx], [mn, mx], 'r--')
                ax.set_xlabel('Ground Truth')
                ax.set_ylabel('Prediction')
                ax.set_title(f'{task} Predictions vs Ground Truth')
                fig_path = os.path.join(output_dir, f'{task}_scatter.png')
                fig.savefig(fig_path)
                plt.close(fig)
        except Exception:
            # 忽略绘图失败
            pass

    # 构建 DataFrame 并保存为 CSV
    rows = []
    for i in range(total_len or 0):
        row = {}
        if video_ids is not None and i < len(video_ids):
            row['video_id'] = video_ids[i]
        else:
            row['video_id'] = f'sample_{i}'
        for task, d in task_data.items():
            row[f'{task}_pred'] = float(d['preds'][i]) if i < len(d['preds']) else float('nan')
            row[f'{task}_true'] = float(d['targets'][i]) if i < len(d['targets']) else float('nan')
        rows.append(row)

    try:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, 'predictions_vs_targets.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
    except Exception:
        # 如果 pandas 不可用或保存失败，尝试用简单的文本方式保存
        try:
            txt_path = os.path.join(output_dir, 'predictions_vs_targets.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(str(r) + '\n')
        except Exception:
            pass

    # 保存 summary metrics
    if metrics is not None:
        try:
            import json
            metrics_path = os.path.join(output_dir, 'metrics_summary.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return True