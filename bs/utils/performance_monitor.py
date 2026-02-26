#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
性能监控与剖析工具

功能：
1. GPU/CPU内存监控
2. 训练速度分析
3. 内存泄漏检测
4. 性能瓶颈识别
5. 实时性能报告
6. 历史性能数据存储
"""

import os
import time
import json
import psutil
import GPUtil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import threading
import queue
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, log_dir: str, monitoring_interval: float = 1.0):
        """
        初始化性能分析器
        
        Args:
            log_dir: 日志保存目录
            monitoring_interval: 监控间隔（秒）
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        
        # 性能数据存储
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.epoch_stats = defaultdict(dict)
        
        # 内存基准
        self.baseline_memory = None
        
        # 性能计数器
        self.counters = defaultdict(int)
        
        logger.info(f"PerformanceProfiler initialized, log dir: {log_dir}")
    
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                
                # 收集PyTorch指标
                pytorch_metrics = self._collect_pytorch_metrics()
                metrics.update(pytorch_metrics)
                
                # 收集GPU指标
                gpu_metrics = self._collect_gpu_metrics()
                metrics.update(gpu_metrics)
                
                # 存储指标
                timestamp = time.time()
                for key, value in metrics.items():
                    self.metrics_history[key].append((timestamp, value))
                
                # 检查内存泄漏
                self._check_memory_leak(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        metrics = {}
        
        # CPU使用率
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        metrics['cpu_count'] = psutil.cpu_count()
        
        # 内存使用
        memory = psutil.virtual_memory()
        metrics['memory_total_gb'] = memory.total / (1024**3)
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_percent'] = memory.percent
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        metrics['disk_total_gb'] = disk.total / (1024**3)
        metrics['disk_used_gb'] = disk.used / (1024**3)
        metrics['disk_percent'] = (disk.used / disk.total) * 100
        
        # 系统负载
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        metrics['load_avg_1min'] = load_avg[0]
        
        return metrics
    
    def _collect_pytorch_metrics(self) -> Dict[str, float]:
        """收集PyTorch指标"""
        metrics = {}
        
        if torch.cuda.is_available():
            # CUDA内存使用
            metrics['cuda_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['cuda_cached_gb'] = torch.cuda.memory_reserved() / (1024**3)
            metrics['cuda_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
            
            # 获取GPU属性
            if torch.cuda.device_count() > 0:
                device_props = torch.cuda.get_device_properties(0)
                metrics['cuda_total_gb'] = device_props.total_memory / (1024**3)
                metrics['cuda_memory_percent'] = (metrics['cuda_allocated_gb'] / 
                                                 metrics['cuda_total_gb']) * 100
        
        # 计算张量数量（近似）
        tensor_count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    tensor_count += 1
            except:
                pass
        
        metrics['active_tensors'] = tensor_count
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """收集GPU指标"""
        metrics = {}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 假设使用第一个GPU
                metrics['gpu_utilization'] = gpu.load * 100
                metrics['gpu_memory_used_gb'] = gpu.memoryUsed / 1024
                metrics['gpu_memory_total_gb'] = gpu.memoryTotal / 1024
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_temperature'] = gpu.temperature
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")
        
        return metrics
    
    def _check_memory_leak(self, metrics: Dict[str, float]):
        """检查内存泄漏"""
        if self.baseline_memory is None:
            # 设置内存基准
            self.baseline_memory = {
                'cpu_memory_gb': metrics.get('memory_used_gb', 0),
                'cuda_memory_gb': metrics.get('cuda_allocated_gb', 0),
                'timestamp': time.time()
            }
            return
        
        # 检查内存增长
        current_time = time.time()
        time_elapsed = current_time - self.baseline_memory['timestamp']
        
        if time_elapsed > 300:  # 5分钟后开始检查
            # CPU内存检查
            cpu_memory_growth = (metrics.get('memory_used_gb', 0) - 
                               self.baseline_memory['cpu_memory_gb'])
            
            # CUDA内存检查
            cuda_memory_growth = (metrics.get('cuda_allocated_gb', 0) - 
                                self.baseline_memory['cuda_memory_gb'])
            
            # 如果内存增长超过阈值，发出警告
            if cpu_memory_growth > 2.0:  # 2GB增长
                logger.warning(f"Potential CPU memory leak detected: "
                             f"{cpu_memory_growth:.2f} GB growth over {time_elapsed/60:.1f} minutes")
            
            if cuda_memory_growth > 1.0:  # 1GB增长
                logger.warning(f"Potential CUDA memory leak detected: "
                             f"{cuda_memory_growth:.2f} GB growth over {time_elapsed/60:.1f} minutes")
    
    @contextmanager
    def profile_section(self, section_name: str):
        """性能分析上下文管理器"""
        start_time = time.time()
        start_memory = self._get_current_memory_usage()
        
        logger.info(f"Starting profile section: {section_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_current_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 记录性能指标
            self.counters[f'{section_name}_count'] += 1
            self._record_metric(f'{section_name}_duration', duration)
            self._record_metric(f'{section_name}_memory_delta_gb', memory_delta)
            
            logger.info(f"Profile section {section_name} completed: "
                       f"duration={duration:.3f}s, memory_delta={memory_delta:.3f}GB")
    
    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用量"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return psutil.virtual_memory().used / (1024**3)
    
    def _record_metric(self, name: str, value: float):
        """记录指标"""
        timestamp = time.time()
        self.metrics_history[name].append((timestamp, value))
    
    def record_custom_metric(self, name: str, value: float, 
                           timestamp: Optional[float] = None):
        """记录自定义指标"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history[name].append((timestamp, value))
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        stats = {}
        
        # 最新指标
        for key, history in self.metrics_history.items():
            if history:
                latest_value = history[-1][1]
                stats[f'{key}_current'] = latest_value
                
                # 计算平均值（最近100个样本）
                recent_samples = list(history)[-100:]
                if recent_samples:
                    avg_value = np.mean([sample[1] for sample in recent_samples])
                    stats[f'{key}_avg'] = avg_value
        
        # 性能计数器
        stats.update(self.counters)
        
        return stats
    
    def get_performance_report(self, epoch: Optional[int] = None) -> str:
        """生成性能报告"""
        stats = self.get_current_stats()
        
        report_lines = [
            "=" * 60,
            "Performance Report",
            "=" * 60,
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if epoch is not None:
            report_lines.append(f"Epoch: {epoch}")
            report_lines.append("")
        
        # 系统资源
        report_lines.extend([
            "System Resources:",
            f"  CPU Usage: {stats.get('cpu_percent_current', 0):.1f}%",
            f"  Memory Usage: {stats.get('memory_percent_current', 0):.1f}% "
            f"({stats.get('memory_used_gb_current', 0):.2f}GB / "
            f"{stats.get('memory_total_gb_current', 0):.2f}GB)",
            f"  Disk Usage: {stats.get('disk_percent_current', 0):.1f}%",
            ""
        ])
        
        # GPU资源
        if 'cuda_memory_percent_current' in stats:
            report_lines.extend([
                "GPU Resources:",
                f"  GPU Utilization: {stats.get('gpu_utilization_current', 0):.1f}%",
                f"  GPU Memory: {stats.get('cuda_memory_percent_current', 0):.1f}% "
                f"({stats.get('cuda_allocated_gb_current', 0):.2f}GB / "
                f"{stats.get('cuda_total_gb_current', 0):.2f}GB)",
                f"  GPU Temperature: {stats.get('gpu_temperature_current', 0):.1f}°C",
                ""
            ])
        
        # PyTorch特定指标
        if 'active_tensors_current' in stats:
            report_lines.extend([
                "PyTorch Metrics:",
                f"  Active Tensors: {stats.get('active_tensors_current', 0):,}",
                f"  Max CUDA Memory: {stats.get('cuda_max_allocated_gb_current', 0):.2f}GB",
                ""
            ])
        
        # 性能分析
        report_lines.append("Performance Analysis:")
        for key, value in self.counters.items():
            if 'duration' in key:
                avg_duration = stats.get(f'{key}_avg', 0)
                report_lines.append(f"  {key}: count={value}, avg={avg_duration:.3f}s")
        
        report_lines.extend(["", "=" * 60])
        
        return "\n".join(report_lines)
    
    def save_metrics(self, filepath: str):
        """保存指标到文件"""
        # 转换数据格式
        data_to_save = {}
        for key, history in self.metrics_history.items():
            data_to_save[key] = [
                {'timestamp': ts, 'value': val} for ts, val in history
            ]
        
        # 保存为JSON
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def plot_metrics(self, save_dir: str, metrics: Optional[List[str]] = None):
        """绘制指标图表"""
        if metrics is None:
            # 默认绘制关键指标
            metrics = [
                'memory_percent', 'cpu_percent',
                'cuda_memory_percent', 'gpu_utilization',
                'cuda_allocated_gb'
            ]
        
        plt.style.use('seaborn-v0_8')
        
        for metric in metrics:
            if metric not in self.metrics_history:
                continue
            
            history = list(self.metrics_history[metric])
            if not history:
                continue
            
            timestamps = [item[0] for item in history]
            values = [item[1] for item in history]
            
            # 转换为相对时间（分钟）
            start_time = timestamps[0]
            relative_times = [(ts - start_time) / 60.0 for ts in timestamps]
            
            plt.figure(figsize=(12, 6))
            plt.plot(relative_times, values, linewidth=2)
            plt.xlabel('Time (minutes)')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Over Time')
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            save_path = os.path.join(save_dir, f'{metric}_plot.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Plot saved to {save_path}")
    
    def __del__(self):
        """清理资源"""
        self.stop_monitoring()


class BatchTimeProfiler:
    """批次时间分析器"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.batch_times = deque(maxlen=window_size)
        self.data_loading_times = deque(maxlen=window_size)
        self.forward_times = deque(maxlen=window_size)
        self.backward_times = deque(maxlen=window_size)
        self.optimization_times = deque(maxlen=window_size)
        
        self.current_batch_start = None
        self.current_stage_start = None
        self.current_stage = None
    
    def start_batch(self):
        """开始记录批次"""
        self.current_batch_start = time.time()
    
    def start_stage(self, stage: str):
        """开始记录阶段"""
        self.current_stage = stage
        self.current_stage_start = time.time()
    
    def end_stage(self, stage: str):
        """结束记录阶段"""
        if self.current_stage_start is None or self.current_stage != stage:
            return
        
        duration = time.time() - self.current_stage_start
        
        if stage == 'data_loading':
            self.data_loading_times.append(duration)
        elif stage == 'forward':
            self.forward_times.append(duration)
        elif stage == 'backward':
            self.backward_times.append(duration)
        elif stage == 'optimization':
            self.optimization_times.append(duration)
    
    def end_batch(self):
        """结束记录批次"""
        if self.current_batch_start is None:
            return
        
        duration = time.time() - self.current_batch_start
        self.batch_times.append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        
        if self.batch_times:
            stats['batch_time_mean'] = np.mean(self.batch_times)
            stats['batch_time_std'] = np.std(self.batch_times)
            stats['batch_time_recent_mean'] = np.mean(list(self.batch_times)[-10:])
        
        if self.data_loading_times:
            stats['data_loading_mean'] = np.mean(self.data_loading_times)
            stats['data_loading_ratio'] = np.mean(self.data_loading_times) / stats['batch_time_mean']
        
        if self.forward_times:
            stats['forward_mean'] = np.mean(self.forward_times)
            stats['forward_ratio'] = np.mean(self.forward_times) / stats['batch_time_mean']
        
        if self.backward_times:
            stats['backward_mean'] = np.mean(self.backward_times)
            stats['backward_ratio'] = np.mean(self.backward_times) / stats['batch_time_mean']
        
        if self.optimization_times:
            stats['optimization_mean'] = np.mean(self.optimization_times)
            stats['optimization_ratio'] = np.mean(self.optimization_times) / stats['batch_time_mean']
        
        # 计算吞吐量（样本/秒）
        if self.batch_times:
            stats['throughput_samples_per_sec'] = 1.0 / np.mean(self.batch_times)  # 假设batch_size=1
        
        return stats
    
    def get_performance_suggestions(self) -> List[str]:
        """获取性能优化建议"""
        suggestions = []
        stats = self.get_stats()
        
        # 数据加载瓶颈
        if stats.get('data_loading_ratio', 0) > 0.3:
            suggestions.append("Data loading appears to be a bottleneck. Consider:")
            suggestions.append("  - Increasing num_workers in DataLoader")
            suggestions.append("  - Using pin_memory=True")
            suggestions.append("  - Optimizing data preprocessing pipeline")
        
        # 前向传播瓶颈
        if stats.get('forward_ratio', 0) > 0.5:
            suggestions.append("Forward pass appears to be a bottleneck. Consider:")
            suggestions.append("  - Using mixed precision training (AMP)")
            suggestions.append("  - Optimizing model architecture")
            suggestions.append("  - Using gradient checkpointing")
        
        # 反向传播瓶颈
        if stats.get('backward_ratio', 0) > 0.4:
            suggestions.append("Backward pass appears to be a bottleneck. Consider:")
            suggestions.append("  - Reducing model complexity")
            suggestions.append("  - Using gradient accumulation")
            suggestions.append("  - Optimizing loss computation")
        
        return suggestions