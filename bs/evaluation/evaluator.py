#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 评估模块

实现模型评估和可视化功能。
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)


# ---- 字体与保存配置：确保中文显示与图片一致性 ----
def _setup_fonts_for_chinese():
    """在不同环境下尽可能启用中文字体，避免缺字告警，并统一导出风格。"""
    try:
        import matplotlib
        from matplotlib import font_manager
        import seaborn as sns  # noqa

        # 候选字体名称（优先顺序），尽量强制使用微软雅黑
        candidates = [
            'Microsoft YaHei',  # Windows 常见，优先且尽量强制
            'SimHei',            # 黑体
            'Arial Unicode MS',  # 全字库
            'Noto Sans CJK SC',  # Google Noto CJK
            'DejaVu Sans',       # 兜底
        ]
        # 已安装字体名称集合
        installed = {f.name for f in font_manager.fontManager.ttflist}
        # 优先并直接选择微软雅黑；若不存在再按候选顺序回退
        if 'Microsoft YaHei' in installed:
            chosen = 'Microsoft YaHei'
        else:
            chosen = next((n for n in candidates if n in installed), 'DejaVu Sans')

        # 全局 rcParams 设置
        matplotlib.rcParams['font.family'] = 'sans-serif'
        if chosen == 'Microsoft YaHei':
            # 直接强制只用微软雅黑，避免回退到其它英文字体（如 Arial）导致缺字告警
            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        else:
            matplotlib.rcParams['font.sans-serif'] = [chosen] + [n for n in candidates if n != chosen]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示

        # 图像保存与版式
        matplotlib.rcParams['savefig.dpi'] = 200
        matplotlib.rcParams['figure.dpi'] = 120
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['figure.autolayout'] = False  # 我们在各处调用 tight_layout()

        # 统一 seaborn 风格与字体
        try:
            sns.set_theme(style='whitegrid', font=chosen)
        except Exception:
            pass
        return chosen
    except Exception:
        return None


class Evaluator:
    """模型评估器
    
    实现模型评估和可视化功能。
    
    Args:
        model (torch.nn.Module): 模型
        config (dict): 配置字典
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        output_dir (str, optional): 评估输出目录，若未提供则从配置推断
    """
    
    def __init__(self, model, config, test_loader, device=None, output_dir=None):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 字体与输出风格（尽早设置，避免后续作图缺字）
        try:
            chosen_font = _setup_fonts_for_chinese()
            if chosen_font:
                logger.info(f"Using font for Chinese rendering: {chosen_font}")
        except Exception:
            pass

        # 创建输出目录（优先顺序：参数 > config.eval.output_dir > config.train.output_dir/results > 默认路径）
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            eval_cfg = self.config.get('evaluation') or self.config.get('eval') or {}
            cfg_out = eval_cfg.get('output_dir')
            if cfg_out:
                self.output_dir = cfg_out
            else:
                train_out = self.config.get('train', {}).get('output_dir')
                if train_out:
                    self.output_dir = os.path.join(train_out, 'evaluation_results')
                else:
                    self.output_dir = os.path.join('.', 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 任务列表及中文映射
        self.tasks = ['lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall']
        self.task_names_cn = {
            'lip_sync': '口型同步',
            'expression': '表情自然度',
            'audio_quality': '音频质量',
            'cross_modal': '跨模态一致性',
            'overall': '总体评分'
        }
        
        # 启用中文字体与负号正常显示（Windows 优先使用黑体/雅黑）
        # 已在 _setup_fonts_for_chinese 中全局设置为优先/直接使用微软雅黑（若可用）
        
        logger.info(f"Evaluator initialized with device: {self.device}")
    
    def evaluate(self):
        """评估模型
        
        Returns:
            dict: 评估结果
        """
        logger.info("Starting model evaluation")
        
        # 收集预测和真实值
        all_preds = {task: [] for task in self.tasks}
        all_targets = {task: [] for task in self.tasks}
        video_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # 兼容两种批次格式：
                # 1) (features, labels, video_ids)
                # 2) dict (包含 'video_id', 'visual_features' 等)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    features, labels, batch_video_ids = batch
                    video_ids.extend(batch_video_ids)

                    # 将数据移动到设备，并清理 NaN/Inf
                    visual_features = torch.nan_to_num(features['visual'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    audio_features = torch.nan_to_num(features['audio'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    keypoint_features = torch.nan_to_num(features['keypoint'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    au_features = torch.nan_to_num(features['au'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)

                    # 前向传播
                    outputs = self.model(
                        visual_features=visual_features,
                        audio_features=audio_features,
                        keypoint_features=keypoint_features,
                        au_features=au_features
                    )

                    # 统一展平为 1D
                    def _flat(t: torch.Tensor):
                        if t.dim() == 2 and t.size(1) == 1:
                            t = t.squeeze(1)
                        return t.detach().cpu().view(-1).numpy()

                    # 收集预测值
                    all_preds['lip_sync'].append(_flat(outputs['lip_sync']))
                    all_preds['expression'].append(_flat(outputs['expression']))
                    all_preds['audio_quality'].append(_flat(outputs['audio_quality']))
                    all_preds['cross_modal'].append(_flat(outputs['cross_modal']))
                    all_preds['overall'].append(_flat(outputs['overall']))

                    # 收集真实值（labels 中键为不带 _score 的版本）
                    all_targets['lip_sync'].append(_flat(labels['lip_sync'].view(-1)))
                    all_targets['expression'].append(_flat(labels['expression'].view(-1)))
                    all_targets['audio_quality'].append(_flat(labels['audio_quality'].view(-1)))
                    all_targets['cross_modal'].append(_flat(labels['cross_modal'].view(-1)))
                    all_targets['overall'].append(_flat(labels['overall'].view(-1)))
                elif isinstance(batch, dict):
                    # 旧版 Evaluator 期望的 dict 格式
                    video_ids.extend(batch['video_id'])

                    visual_features = torch.nan_to_num(batch['visual_features'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    audio_features = torch.nan_to_num(batch['audio_features'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    keypoint_features = torch.nan_to_num(batch['keypoint_features'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)
                    au_features = torch.nan_to_num(batch['au_features'].to(self.device), nan=0.0, posinf=1e6, neginf=-1e6)

                    # 收集真实值（带 _score 的版本），统一展平为 1D
                    def _flat(t: torch.Tensor):
                        if t.dim() == 2 and t.size(1) == 1:
                            t = t.squeeze(1)
                        return t.detach().cpu().view(-1).numpy()
                    all_targets['lip_sync'].append(_flat(batch['lip_sync_score'].view(-1)))
                    all_targets['expression'].append(_flat(batch['expression_score'].view(-1)))
                    all_targets['audio_quality'].append(_flat(batch['audio_quality_score'].view(-1)))
                    all_targets['cross_modal'].append(_flat(batch['cross_modal_score'].view(-1)))
                    all_targets['overall'].append(_flat(batch['overall_score'].view(-1)))

                    outputs = self.model(
                        visual_features=visual_features,
                        audio_features=audio_features,
                        keypoint_features=keypoint_features,
                        au_features=au_features
                    )

                    for task in self.tasks:
                        # 统一展平为 1D
                        pred = outputs[task]
                        if pred.dim() == 2 and pred.size(1) == 1:
                            pred = pred.squeeze(1)
                        all_preds[task].append(pred.detach().cpu().view(-1).numpy())
                else:
                    raise TypeError("Unsupported batch format from DataLoader.")
        
        # 合并预测和真实值
        for task in self.tasks:
            try:
                all_preds[task] = np.concatenate(all_preds[task])
            except Exception:
                all_preds[task] = np.concatenate([np.asarray(x).ravel() for x in all_preds[task]])
            try:
                all_targets[task] = np.concatenate(all_targets[task])
            except Exception:
                all_targets[task] = np.concatenate([np.asarray(x).ravel() for x in all_targets[task]])
        
        # 计算评估指标（对 -1 无效标签进行掩码）
        masked_preds = {}
        masked_targets = {}
        for task in self.tasks:
            mask = all_targets[task] != -1.0
            if np.any(mask):
                masked_preds[task] = all_preds[task][mask]
                masked_targets[task] = all_targets[task][mask]
            else:
                masked_preds[task] = np.array([], dtype=float)
                masked_targets[task] = np.array([], dtype=float)
        metrics = self._calculate_metrics(masked_preds, masked_targets)
        # 可选：自举置信区间
        try:
            boot_cfg = (self.config.get('evaluation') or self.config.get('eval') or {}).get('bootstrap', {})
            n_resamples = int(boot_cfg.get('n_resamples', 0))
            if n_resamples > 0:
                metrics_ci = self._bootstrap_metrics(masked_preds, masked_targets, n_resamples=n_resamples)
                self._save_bootstrap_cis(metrics_ci)
        except Exception as _:
            pass
        
        # 可视化结果（仅对有效标签数据绘图）
        self._visualize_results(masked_preds, masked_targets, video_ids)
        
        # 保存结果（保存原始含 -1 的结果，便于定位问题）
        self._save_results(metrics, all_preds, all_targets, video_ids)
        
        return metrics

    def _get_score_range(self):
        m = self.config.get('model', {})
        th = m.get('task_heads', {}) if isinstance(m.get('task_heads', {}), dict) else {}
        try:
            smin = float(th.get('score_min', 1.0))
            smax = float(th.get('score_max', 5.0))
        except Exception:
            smin, smax = 1.0, 5.0
        if smax <= smin:
            smin, smax = 0.0, 1.0
        return smin, smax

    def _calculate_metrics(self, predictions, targets):
        """计算评估指标
        
        Args:
            predictions (dict): 预测值字典
            targets (dict): 真实值字典
        
        Returns:
            dict: 评估指标字典
        """
        metrics = {}
        
        smin, smax = self._get_score_range()
        for task in self.tasks:
            preds = predictions[task]
            tgts = targets[task]
            
            if len(tgts) == 0 or len(preds) == 0:
                metrics[task] = {k: 0.0 for k in ['mse','rmse','mae','medae','r2','pearson','spearman','kendall','ccc','accuracy','qwk']}
                continue

            # 基本误差
            mse = mean_squared_error(tgts, preds)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(tgts, preds)
            medae = float(np.median(np.abs(preds - tgts)))
            r2 = r2_score(tgts, preds)

            # 相关性
            pearson = float(np.corrcoef(tgts, preds)[0, 1]) if len(tgts) > 1 else 0.0
            from scipy.stats import spearmanr, kendalltau
            spearman, _ = spearmanr(tgts, preds)
            kendall, _ = kendalltau(tgts, preds)

            # CCC (一致性相关系数)
            mu_t, mu_p = float(np.mean(tgts)), float(np.mean(preds))
            var_t, var_p = float(np.var(tgts)), float(np.var(preds))
            cov_tp = float(np.mean((tgts - mu_t) * (preds - mu_p)))
            ccc = (2 * cov_tp) / (var_t + var_p + (mu_t - mu_p) ** 2 + 1e-12)

            # 粗分类准确率（低/中/高）
            low = smin + (smax - smin) / 3
            mid = smin + 2 * (smax - smin) / 3
            tgts_cat = np.digitize(tgts, bins=[low, mid])
            preds_cat = np.digitize(preds, bins=[low, mid])
            acc = float(np.mean(tgts_cat == preds_cat))

            # QWK（5档离散，四舍五入到 [smin,smax] 区间的整数档）
            try:
                from sklearn.metrics import cohen_kappa_score
                # 将连续映射到 1..5 档（或依据 smin/smax 线性分桶）
                n_bins = 5
                bins = np.linspace(smin, smax, n_bins+1)
                def to_labels(x):
                    idx = np.digitize(np.clip(x, smin, smax), bins[:-1], right=False)
                    idx = np.clip(idx, 0, n_bins-1)
                    return idx
                y_true = to_labels(tgts)
                y_pred = to_labels(preds)
                qwk = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
            except Exception:
                qwk = 0.0

            metrics[task] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'medae': float(medae),
                'r2': float(r2),
                'pearson': float(pearson),
                'spearman': float(spearman if np.isfinite(spearman) else 0.0),
                'kendall': float(kendall if np.isfinite(kendall) else 0.0),
                'ccc': float(ccc),
                'accuracy': float(acc),
                'qwk': float(qwk)
            }
        
        # 计算平均指标
        avg_metrics = {}
        for metric in ['mse', 'rmse', 'mae', 'medae', 'r2', 'pearson', 'spearman', 'kendall', 'ccc', 'accuracy', 'qwk']:
            avg_metrics[metric] = np.mean([metrics[task][metric] for task in self.tasks])
        
        metrics['average'] = avg_metrics
        
        return metrics
    
    def _visualize_results(self, predictions, targets, video_ids):
        """可视化结果
        
        Args:
            predictions (dict): 预测值字典
            targets (dict): 真实值字典
            video_ids (list): 视频ID列表
        """
        # 设置风格与中文字体（再次强制，避免被后续调用覆盖为 Arial）
        try:
            import matplotlib as _mpl
            _mpl.rcParams['font.family'] = 'sans-serif'
            _mpl.rcParams['font.sans-serif'] = [
                'Microsoft YaHei',  # 优先微软雅黑
                'SimHei',            # 黑体
                'Arial Unicode MS',  # 全字库
                'Noto Sans CJK SC',  # Noto CJK
                'DejaVu Sans',       # 兜底
            ]
            _mpl.rcParams['axes.unicode_minus'] = False
            # 使用 rc 明确锁定中文字体，避免 seaborn 默认将字体设为 Arial
            sns.set_theme(
                style='whitegrid',
                rc={
                    'font.family': 'sans-serif',
                    'font.sans-serif': _mpl.rcParams['font.sans-serif'],
                    'axes.unicode_minus': False,
                }
            )
        except Exception:
            # 兜底：至少设置样式
            sns.set(style='whitegrid')
        
        # 1. 散点图：预测值 vs 真实值
        for task in self.tasks:
            plt.figure(figsize=(10, 6))
            
            # 绘制散点图
            sns.scatterplot(x=targets[task], y=predictions[task], alpha=0.6)
            
            # 绘制对角线
            min_val = min(min(targets[task]), min(predictions[task])) if len(predictions[task]) > 0 and len(targets[task]) > 0 else 0
            max_val = max(max(targets[task]), max(predictions[task])) if len(predictions[task]) > 0 and len(targets[task]) > 0 else 1
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 添加标题和标签（中文）
            plt.title(f"{self.task_names_cn.get(task, task)}：预测值 vs 真实值")
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            
            # 添加评估指标（中文）
            mse = mean_squared_error(targets[task], predictions[task]) if len(predictions[task]) > 0 else 0.0
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets[task], predictions[task]) if len(predictions[task]) > 0 else 0.0
            r2 = r2_score(targets[task], predictions[task]) if len(predictions[task]) > 0 else 0.0
            pearson = np.corrcoef(targets[task], predictions[task])[0, 1] if len(predictions[task]) > 1 else 0.0
            
            plt.figtext(0.15, 0.85, 
                       f'均方误差(MSE): {mse:.4f}\n'
                       f'均方根误差(RMSE): {rmse:.4f}\n'
                       f'平均绝对误差(MAE): {mae:.4f}\n'
                       f'决定系数(R²): {r2:.4f}\n'
                       f'皮尔逊相关: {pearson:.4f}')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task}_scatter.png'))
            plt.close()
        
        # 2. 直方图：预测误差分布
        for task in self.tasks:
            plt.figure(figsize=(10, 6))
            
            # 计算误差
            errors = predictions[task] - targets[task]
            
            # 绘制直方图
            sns.histplot(errors, kde=True)
            
            # 添加标题和标签（中文）
            plt.title(f"{self.task_names_cn.get(task, task)}：误差分布")
            plt.xlabel('预测误差')
            plt.ylabel('频数')
            
            # 添加统计信息（中文）
            mean_error = np.mean(errors) if len(errors) > 0 else 0.0
            std_error = np.std(errors) if len(errors) > 0 else 0.0
            
            plt.figtext(0.15, 0.85, 
                       f'平均误差: {mean_error:.4f}\n'
                       f'标准差: {std_error:.4f}')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{task}_error_hist.png'))
            plt.close()
        
        # 3. 热图：任务间相关性
        plt.figure(figsize=(12, 10))
        
        # 创建相关性矩阵（兼容不同任务样本数不同的情况）
        series_data = {}
        for task in self.tasks:
            # 仅在该任务存在数据时添加列
            if len(targets[task]) > 0 and len(predictions[task]) > 0:
                series_data[f"{self.task_names_cn.get(task, task)}-真实"] = pd.Series(targets[task])
                series_data[f"{self.task_names_cn.get(task, task)}-预测"] = pd.Series(predictions[task])
        
        if len(series_data) >= 2:
            df_corr = pd.DataFrame(series_data)  # 不同长度会按索引对齐并用NaN填充
            corr_matrix = df_corr.corr(method='pearson', min_periods=1)
            
            # 绘制热图（中文标题）
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('真实值与预测值的相关性矩阵')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        else:
            # 数据不足时跳过绘制
            plt.text(0.5, 0.5, '数据不足，无法计算相关性热图', ha='center', va='center')
            plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap_skipped.png'))
        plt.close()

        # 2.5 Bland-Altman 图（总体评分）
        try:
            task = 'overall'
            if len(predictions[task]) > 1 and len(targets[task]) > 1:
                means = 0.5 * (predictions[task] + targets[task])
                diffs = predictions[task] - targets[task]
                md = np.mean(diffs)
                sd = np.std(diffs)
                loA1, loA2 = md - 1.96*sd, md + 1.96*sd
                plt.figure(figsize=(10,6))
                sns.scatterplot(x=means, y=diffs, alpha=0.6)
                plt.axhline(md, color='r', linestyle='--', label=f'均值差={md:.3f}')
                plt.axhline(loA1, color='g', linestyle='--', label=f'-1.96σ={loA1:.3f}')
                plt.axhline(loA2, color='g', linestyle='--', label=f'+1.96σ={loA2:.3f}')
                plt.title('总体评分 Bland-Altman 图')
                plt.xlabel('均值(预测,真实)')
                plt.ylabel('差值(预测-真实)')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{task}_bland_altman.png'))
                plt.close()
        except Exception:
            pass
        
        # 4. 箱线图：各任务的预测和真实分布
        plt.figure(figsize=(14, 8))
        
        # 准备数据
        box_data = []
        for task in self.tasks:
            for i in range(len(targets[task])):
                box_data.append({
                    '任务': self.task_names_cn.get(task, task),
                    '类型': '真实值',
                    '分数': targets[task][i]
                })
                if i < len(predictions[task]):
                    box_data.append({
                        '任务': self.task_names_cn.get(task, task),
                        '类型': '预测值',
                        '分数': predictions[task][i]
                    })
        
        df_box = pd.DataFrame(box_data)
        
        # 绘制箱线图（中文）
        if len(df_box) > 0:
            sns.boxplot(x='任务', y='分数', hue='类型', data=df_box)
            plt.title('真实分数与预测分数的分布')
            plt.xlabel('任务')
            plt.ylabel('分数')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'score_distribution_boxplot.png'))
        else:
            plt.text(0.5, 0.5, '数据不足，无法绘制箱线图', ha='center', va='center')
            plt.savefig(os.path.join(self.output_dir, 'score_distribution_boxplot_skipped.png'))
        plt.close()
        
        # 5. 条形图：各评估指标比较（中文）
        metrics_localized = {
            'rmse': '均方根误差 (RMSE)',
            'mae': '平均绝对误差 (MAE)',
            'r2': '决定系数 (R²)',
            'pearson': '皮尔逊相关',
            'spearman': '斯皮尔曼相关',
            'accuracy': '准确率'
        }
        
        metrics = self._calculate_metrics(predictions, targets)
        
        for metric in ['rmse', 'mae', 'medae', 'r2', 'pearson', 'spearman', 'kendall', 'ccc', 'accuracy', 'qwk']:
            plt.figure(figsize=(12, 6))
            
            # 准备数据（中文任务名）
            tasks_display = [self.task_names_cn.get(task, task) for task in self.tasks]
            metric_values = [metrics[task][metric] for task in self.tasks]
            
            # 绘制条形图
            bars = plt.bar(tasks_display, metric_values)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', rotation=0)
            
            # 添加标题和标签（中文）
            plt.title(f"各任务的{metrics_localized.get(metric, metric)}")
            plt.xlabel('任务')
            plt.ylabel(metrics_localized.get(metric, metric))
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{metric}_comparison.png'))
            plt.close()

    def _bootstrap_metrics(self, predictions, targets, n_resamples=200, random_state=42):
        rng = np.random.default_rng(random_state)
        tasks = self.tasks
        results = {t: {} for t in tasks}
        # 哪些指标做 CI
        metric_fns = {
            'rmse': lambda y, x: float(np.sqrt(mean_squared_error(y, x))),
            'mae': lambda y, x: float(mean_absolute_error(y, x)),
            'r2': lambda y, x: float(r2_score(y, x)),
            'pearson': lambda y, x: float(np.corrcoef(y, x)[0,1]) if len(y) > 1 else 0.0,
            'spearman': lambda y, x: float(__import__('scipy').stats.spearmanr(y, x)[0]) if len(y) > 1 else 0.0,
            'ccc': lambda y, x: float((2*np.cov(y, x, ddof=0)[0,1]) / (np.var(y) + np.var(x) + (np.mean(y)-np.mean(x))**2 + 1e-12)) if len(y) > 1 else 0.0
        }
        for t in tasks:
            y = targets[t]
            x = predictions[t]
            if len(y) < 2:
                results[t] = {k: {'low':0.0,'high':0.0} for k in metric_fns}
                continue
            n = len(y)
            for name, fn in metric_fns.items():
                vals = []
                for _ in range(n_resamples):
                    idx = rng.integers(0, n, size=n)
                    vals.append(fn(y[idx], x[idx]))
                low, high = np.percentile(vals, [2.5, 97.5])
                results[t][name] = {'low': float(low), 'high': float(high)}
        return results

    def _save_bootstrap_cis(self, metrics_ci):
        try:
            path = os.path.join(self.output_dir, 'metrics_ci.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metrics_ci, f, indent=4, ensure_ascii=False)
            logger.info(f"Bootstrap confidence intervals saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save bootstrap CIs: {e}")
    
    def _save_results(self, metrics, predictions, targets, video_ids):
        """保存结果
        
        Args:
            metrics (dict): 评估指标字典
            predictions (dict): 预测值字典
            targets (dict): 真实值字典
            video_ids (list): 视频ID列表
        """
        # 1. 保存评估指标
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Evaluation metrics saved to {metrics_path}")
        
        # 2. 保存预测结果
        results = []
        
        for i, video_id in enumerate(video_ids):
            result = {'video_id': video_id}
            
            for task in self.tasks:
                if i < len(predictions[task]):
                    result[f'{task}_pred'] = float(predictions[task][i])
                    result[f'{task}_true'] = float(targets[task][i])
                    result[f'{task}_error'] = float(predictions[task][i] - targets[task][i])
            
            results.append(result)
        
        # 保存为CSV文件（使用带BOM的UTF-8以兼容Excel）
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, 'predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Prediction results saved to {csv_path}")
        
        # 3. 生成评估报告
        report = self._generate_report(metrics)
        report_path = os.path.join(self.output_dir, 'evaluation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def generate_report(self, metrics=None):
        """生成评估报告并返回报告路径（供外部调用）
        
        Args:
            metrics (dict, optional): 评估指标字典；若未提供，则尝试从输出目录读取 metrics.json
        
        Returns:
            str: 报告文件路径
        """
        if metrics is None:
            # 回退到读取已保存的 metrics.json
            import json, os
            metrics_path = os.path.join(self.output_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            else:
                # 如果不存在则先返回空的占位报告
                metrics = {'average': {m: 0.0 for m in ['mse','rmse','mae','r2','pearson','spearman','accuracy']}}
                for t in self.tasks:
                    metrics[t] = metrics['average'].copy()

        report = self._generate_report(metrics)
        report_path = os.path.join(self.output_dir, 'evaluation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return report_path
    
    def _generate_report(self, metrics):
        """生成评估报告
        
        Args:
            metrics (dict): 评估指标字典
        
        Returns:
            str: 评估报告
        """
        report = "# AI生成说话人脸视频评价模型评估报告\n\n"
        
        # 添加时间戳
        from datetime import datetime
        report += f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 添加模型信息（兼容当前 config.yaml 结构）
        report += "## 模型信息\n\n"
        mcfg = self.config.get('model', {})
        tcfg = mcfg.get('transformer', {})
        report += f"- 模型名称: {mcfg.get('name', 'N/A')}\n"
        report += f"- 编码器维度: {mcfg.get('encoder_dim', 'N/A')}\n"
        report += f"- 隐层维度: {mcfg.get('hidden_dim', 'N/A')}\n"
        report += f"- 视觉/音频/关键点/AU 维度: {mcfg.get('visual_dim', 'N/A')}/{mcfg.get('audio_dim', 'N/A')}/{mcfg.get('keypoint_dim', 'N/A')}/{mcfg.get('au_dim', 'N/A')}\n"
        report += f"- Transformer层数: {tcfg.get('num_layers', 'N/A')}\n"
        report += f"- 多头注意力头数: {tcfg.get('num_heads', 'N/A')}\n\n"
        
        # 添加评估指标总结
        report += "## 评估指标总结\n\n"
        report += "### 平均指标\n\n"
        report += "| 指标 | 值 |\n"
        report += "|------|------|\n"
        
        for metric, value in metrics['average'].items():
            report += f"| {metric.upper()} | {value:.4f} |\n"
        
        report += "\n### 各任务指标\n\n"
        report += "| 任务 | RMSE | MAE | R² | Pearson | Spearman | 准确率 |\n"
        report += "|------|------|------|------|------|------|------|\n"
        
        for task in self.tasks:
            task_metrics = metrics[task]
            # 使用中文任务名
            task_cn = self.task_names_cn.get(task, task)
            report += f"| {task_cn} | "
            report += f"{task_metrics.get('rmse', 0):.4f} | "
            report += f"{task_metrics.get('mae', 0):.4f} | "
            report += f"{task_metrics.get('r2', 0):.4f} | "
            report += f"{task_metrics.get('pearson', 0):.4f} | "
            report += f"{task_metrics.get('spearman', 0):.4f} | "
            report += f"{task_metrics.get('accuracy', 0):.4f} |\n"
        
        # 添加结果分析
        report += "\n## 结果分析\n\n"
        
        # 找出表现最好和最差的任务（基于 RMSE）
        import numpy as _np
        rmse_values = {task: metrics[task].get('rmse', _np.inf) for task in self.tasks}
        best_task = min(rmse_values, key=rmse_values.get)
        worst_task = max(rmse_values, key=rmse_values.get)
        
        report += f"- 表现最好的任务: **{self.task_names_cn.get(best_task, best_task)}** (RMSE: {rmse_values[best_task]:.4f})\n"
        report += f"- 表现最差的任务: **{self.task_names_cn.get(worst_task, worst_task)}** (RMSE: {rmse_values[worst_task]:.4f})\n\n"
        
        # 添加结论和建议
        report += "## 结论和建议\n\n"
        
        # 根据平均R²值给出总体评价
        avg_r2 = metrics['average'].get('r2', 0.0)
        if avg_r2 > 0.8:
            report += "- 模型整体表现**优秀**，能够准确预测各项评价指标。\n"
        elif avg_r2 > 0.6:
            report += "- 模型整体表现**良好**，对大多数评价指标有较好的预测能力。\n"
        elif avg_r2 > 0.4:
            report += "- 模型整体表现**一般**，对部分评价指标的预测存在一定误差。\n"
        else:
            report += "- 模型整体表现**较差**，预测结果与真实值存在较大差距。\n"
        
        # 针对表现最差的任务给出改进建议
        report += f"- 针对**{self.task_names_cn.get(worst_task, worst_task)}**任务的改进建议:\n"
        report += "  - 增加相关特征的提取和处理\n"
        report += "  - 调整模型结构，增强对该任务的学习能力\n"
        report += "  - 考虑使用更专业的预训练模型提取特征\n\n"
        
        # 添加图表说明
        report += "## 评估图表\n\n"
        report += "本评估生成了以下可视化图表：\n\n"
        report += "1. **散点图**: 展示每个任务的预测值与真实值的对比\n"
        report += "2. **误差分布直方图**: 展示每个任务预测误差的分布情况\n"
        report += "3. **相关性热图**: 展示各任务真实值和预测值之间的相关性\n"
        report += "4. **分数分布箱线图**: 比较各任务真实分数和预测分数的分布\n"
        report += "5. **评估指标对比图**: 展示各任务在不同评估指标上的表现\n"
        
        return report


def evaluate_model(model, config, test_loader, device=None):
    """评估模型
    
    Args:
        model (torch.nn.Module): 模型
        config (dict): 配置字典
        test_loader (torch.utils.data.DataLoader): 测试数据加载器
        device (torch.device): 计算设备
    
    Returns:
        dict: 评估结果
    """
    # 创建评估器
    evaluator = Evaluator(model, config, test_loader, device)
    
    # 评估模型
    metrics = evaluator.evaluate()
    
    return metrics


class VideoQualityEvaluator:
    """视频质量评估器
    
    用于评估单个视频的质量。
    
    Args:
        model (torch.nn.Module): 模型
        feature_extractor: 特征提取器
        config (dict): 配置字典
        device (torch.device): 计算设备
    """
    
    def __init__(self, model, feature_extractor, config, device=None):
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"VideoQualityEvaluator initialized with device: {self.device}")
    
    def evaluate_video(self, video_path, output_dir=None, verbose=False):
        """评估视频质量
        
        Args:
            video_path (str): 视频文件路径
            output_dir (str, optional): 输出目录路径
            verbose (bool, optional): 是否显示详细信息
        
        Returns:
            dict: 评估结果
        """
        logger.info(f"Evaluating video: {video_path}")
        
        # 提取特征
        features = self.feature_extractor.extract_all_features(video_path)
        
        # 转换为张量
        visual_features = torch.from_numpy(features['visual']).float().unsqueeze(0).to(self.device)
        audio_features = torch.from_numpy(features['audio']).float().unsqueeze(0).to(self.device)
        keypoint_features = torch.from_numpy(features['keypoint']).float().unsqueeze(0).to(self.device)
        au_features = torch.from_numpy(features['au']).float().unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                visual_features=visual_features,
                audio_features=audio_features,
                keypoint_features=keypoint_features,
                au_features=au_features
            )
        
        # 整理结果
        results = {
            'lip_sync_score': float(outputs['lip_sync'].cpu().numpy()[0]),
            'expression_score': float(outputs['expression'].cpu().numpy()[0]),
            'audio_quality_score': float(outputs['audio_quality'].cpu().numpy()[0]),
            'cross_modal_score': float(outputs['cross_modal'].cpu().numpy()[0]),
            'overall_score': float(outputs['overall'].cpu().numpy()[0])
        }
        
        # 生成评估报告
        report = self._generate_report(results, video_path)
        results['report'] = report
        
        # 如果指定了输出目录，保存报告
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            if verbose:
                logger.info(f"Report saved to {report_path}")
        
        return results
    
    def _generate_report(self, results, video_path):
        """生成评估报告
        
        Args:
            results (dict): 评估结果
            video_path (str): 视频文件路径
        
        Returns:
            str: 评估报告
        """
        video_name = os.path.basename(video_path)
        
        report = f"# 视频质量评估报告: {video_name}\n\n"
        
        # 添加时间戳
        from datetime import datetime
        report += f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 添加评分结果
        report += "## 评分结果\n\n"
        report += "| 评估指标 | 分数 | 等级 |\n"
        report += "|---------|------|------|\n"
        
        # 定义评分等级
        def get_grade(score):
            if score >= 4.5:
                return "优秀"
            elif score >= 3.5:
                return "良好"
            elif score >= 2.5:
                return "一般"
            elif score >= 1.5:
                return "较差"
            else:
                return "差"
        
        for metric, score in results.items():
            if metric != 'report':
                grade = get_grade(score)
                report += f"| {metric.replace('_score', '').replace('_', ' ').title()} | {score:.2f} | {grade} |\n"
        
        # 添加总体评价
        report += "\n## 总体评价\n\n"
        
        overall_score = results['overall_score']
        overall_grade = get_grade(overall_score)
        
        if overall_grade == "优秀":
            report += "该视频整体质量**优秀**，各方面表现均达到很高水平。\n\n"
        elif overall_grade == "良好":
            report += "该视频整体质量**良好**，大部分方面表现良好，但仍有提升空间。\n\n"
        elif overall_grade == "一般":
            report += "该视频整体质量**一般**，部分方面表现不足，需要改进。\n\n"
        elif overall_grade == "较差":
            report += "该视频整体质量**较差**，多个方面存在明显问题，需要大幅改进。\n\n"
        else:
            report += "该视频整体质量**差**，各方面表现均不理想，需要全面重做。\n\n"
        
        # 添加详细分析
        report += "## 详细分析\n\n"
        
        # 口型同步性分析
        lip_sync_score = results['lip_sync_score']
        report += "### 口型同步性\n\n"
        if lip_sync_score >= 4.0:
            report += "口型与语音**高度同步**，观众感知非常自然。\n\n"
        elif lip_sync_score >= 3.0:
            report += "口型与语音**基本同步**，偶有不匹配但不影响整体观感。\n\n"
        elif lip_sync_score >= 2.0:
            report += "口型与语音**同步性一般**，存在明显不匹配，影响观感。\n\n"
        else:
            report += "口型与语音**严重不同步**，给人强烈的违和感。\n\n"
        
        # 表情自然度分析
        expression_score = results['expression_score']
        report += "### 表情自然度\n\n"
        if expression_score >= 4.0:
            report += "面部表情**非常自然**，情感表达丰富且符合语境。\n\n"
        elif expression_score >= 3.0:
            report += "面部表情**较为自然**，基本符合语境但略显呆板。\n\n"
        elif expression_score >= 2.0:
            report += "面部表情**不够自然**，存在僵硬或过度夸张的问题。\n\n"
        else:
            report += "面部表情**极不自然**，给人明显的机械感或诡异感。\n\n"
        
        # 音频质量分析
        audio_quality_score = results['audio_quality_score']
        report += "### 音频质量\n\n"
        if audio_quality_score >= 4.0:
            report += "音频质量**优秀**，声音清晰自然，无明显噪音或失真。\n\n"
        elif audio_quality_score >= 3.0:
            report += "音频质量**良好**，声音基本清晰，有轻微噪音或不自然感。\n\n"
        elif audio_quality_score >= 2.0:
            report += "音频质量**一般**，存在明显噪音、失真或机械感。\n\n"
        else:
            report += "音频质量**较差**，噪音严重或声音高度失真。\n\n"
        
        # 跨模态一致性分析
        cross_modal_score = results['cross_modal_score']
        report += "### 跨模态一致性\n\n"
        if cross_modal_score >= 4.0:
            report += "视觉与音频内容**高度一致**，表情、口型与语音内容完美匹配。\n\n"
        elif cross_modal_score >= 3.0:
            report += "视觉与音频内容**基本一致**，偶有不匹配但整体协调。\n\n"
        elif cross_modal_score >= 2.0:
            report += "视觉与音频内容**一致性一般**，存在明显不协调之处。\n\n"
        else:
            report += "视觉与音频内容**严重不一致**，给人强烈的违和感。\n\n"
        
        # 添加改进建议
        report += "## 改进建议\n\n"
        
        # 找出得分最低的方面
        scores = {
            '口型同步性': lip_sync_score,
            '表情自然度': expression_score,
            '音频质量': audio_quality_score,
            '跨模态一致性': cross_modal_score
        }
        worst_aspect = min(scores, key=scores.get)
        
        report += f"重点改进方向: **{worst_aspect}**\n\n"
        
        if worst_aspect == '口型同步性':
            report += "- 优化口型生成算法，提高与音频的同步精度\n"
            report += "- 增加关键音素与口型关键帧的对应关系\n"
            report += "- 考虑引入更精细的唇部动作控制\n"
        elif worst_aspect == '表情自然度':
            report += "- 增强面部表情的多样性和自然过渡\n"
            report += "- 优化情感表达与语音内容的匹配\n"
            report += "- 减少面部动作的机械感和重复性\n"
        elif worst_aspect == '音频质量':
            report += "- 提高音频采样率和位深度\n"
            report += "- 优化音频合成算法，减少噪音和失真\n"
            report += "- 增强音色的自然度和情感表达\n"
        elif worst_aspect == '跨模态一致性':
            report += "- 加强视觉与音频内容的语义对齐\n"
            report += "- 优化表情与语音情感的匹配度\n"
            report += "- 提高整体协调性和一致性\n"
        
        return report


def evaluate_video(model, feature_extractor, config, video_path, device=None):
    """评估视频质量
    
    Args:
        model (torch.nn.Module): 模型
        feature_extractor: 特征提取器
        config (dict): 配置字典
        video_path (str): 视频文件路径
        device (torch.device): 计算设备
    
    Returns:
        dict: 评估结果
    """
    # 创建视频质量评估器
    evaluator = VideoQualityEvaluator(model, feature_extractor, config, device)
    
    # 评估视频
    results = evaluator.evaluate_video(video_path)
    
    return results