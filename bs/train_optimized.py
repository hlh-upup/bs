#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化的AI生成说话人脸视频评价模型训练脚本
基于专家分析和实际数据结构
"""
import logging
import os
import sys
import argparse
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import pickle
import time
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TalkingFaceDataset(Dataset):
    """优化的数据集类"""
    
    def __init__(self, data, split='train'):
        self.data = data[split]
        self.features = self.data['features']
        self.labels = self.data['labels']
        self.valid_masks = self.data['valid_masks']
        # 兼容新的 lip_sync_score_new
        if 'lip_sync_score_new' in self.labels and 'lip_sync_score_new' in self.valid_masks:
            # 将其映射成旧字段，后续代码无需改
            self.labels['lip_sync_score'] = self.labels['lip_sync_score_new']
            self.valid_masks['lip_sync_score'] = self.valid_masks['lip_sync_score_new']
        self.split = split
    
    def __len__(self):
        return len(self.labels[list(self.labels.keys())[0]])
    
    def __getitem__(self, idx):
        features = {}
        for key in ['visual', 'audio', 'keypoint', 'au']:
            arr = self.features[key][idx]
            if isinstance(arr, np.ndarray):
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            features[key] = torch.as_tensor(arr, dtype=torch.float32)
        labels = {}
        valid_masks = {}
        for task in ['lip_sync_score', 'expression_score', 'audio_quality_score', 'cross_modal_score', 'overall_score']:
            val = self.labels[task][idx]
            if np.isnan(val):
                # 遇到 NaN 标签，强制失效
                valid_masks[task.replace('_score', '')] = torch.tensor(False, dtype=torch.bool)
                val = 0.0
            labels[task.replace('_score', '')] = torch.tensor(val, dtype=torch.float32)
            valid_masks[task.replace('_score', '')] = torch.tensor(bool(self.valid_masks[task][idx]), dtype=torch.bool)
        return features, labels, valid_masks

class OptimizedMTLModel(nn.Module):
    """优化的多任务学习模型"""
    
    def __init__(self, input_dims, hidden_dim=512, num_layers=6, num_heads=16, dropout=0.3):
        super().__init__()
        
        # 特征嵌入维度
        embed_dim = hidden_dim
        
        # 特征嵌入层
        self.feature_embedders = nn.ModuleDict({
            'visual': nn.Sequential(
                nn.Linear(input_dims['visual'], 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, embed_dim)
            ),
            'audio': nn.Sequential(
                nn.Linear(input_dims['audio'], 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, embed_dim)
            ),
            'keypoint': nn.Sequential(
                nn.Linear(input_dims['keypoint'], 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, embed_dim)
            ),
            'au': nn.Sequential(
                nn.Linear(input_dims['au'], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, embed_dim)
            )
        })
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(150, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 任务特定头
        self.task_heads = nn.ModuleDict({
            'lip_sync': nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ),
            'expression': nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ),
            'audio_quality': nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ),
            'cross_modal': nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ),
            'overall': nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        })
        
        # 任务权重
        self.task_weights = {
            'lip_sync': 0.8,
            'expression': 1.2,
            'audio_quality': 1.0,
            'cross_modal': 1.5,  # 重点优化
            'overall': 1.3
        }
    
    def forward(self, features):
        # 特征嵌入
        embedded_features = []
        for key, embedder in self.feature_embedders.items():
            if key in features:
                x = embedder(features[key])  # (batch, seq, embed_dim)
                embedded_features.append(x)
        
        # 特征融合
        fused = torch.stack(embedded_features, dim=1).mean(dim=1)  # (batch, seq, embed_dim)
        
        # 添加位置编码
        seq_len = fused.size(3)
        fused = fused + self.pos_encoding[:seq_len]
        
        # Transformer处理
        output = self.transformer(fused)
        
        # 全局平均池化
        pooled = output.mean(dim=1)  # (batch, embed_dim)
        
        # 多任务预测
        predictions = {}
        for task, head in self.task_heads.items():
            predictions[task] = head(pooled).squeeze(-1)
        
        return predictions

class Trainer:
    """优化的训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, output_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.03,
            betas=(0.9, 0.992)
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.MSELoss(reduction='none')
        
        # 训练历史
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
        self.best_val_loss = float('inf')
    
    def compute_loss(self, predictions, targets, valid_masks):
        """计算加权多任务损失 (修复广播导致的维度问题)"""
        total_loss = 0
        losses = {}
        task_weights = self.model.task_weights
        for task, pred in predictions.items():
            if task in targets and task in valid_masks:
                mask = valid_masks[task]
                # DataLoader 聚合后如果是标量，会变成形状 [B]；若仍有额外维度，压缩
                if mask.dim() > 1:
                    mask = mask.squeeze()
                target = targets[task]
                if target.dim() > 1:
                    target = target.squeeze()
                if pred.dim() > 1:
                    pred = pred.squeeze()
                # 现在 pred/target/mask 都应为 [B]
                assert pred.shape == target.shape, f"pred {pred.shape} vs target {target.shape}"
                # 掩码同样对齐
                assert mask.shape[0] == pred.shape[0], f"mask {mask.shape} vs pred {pred.shape}"
                if torch.any(mask):
                    per_sample_loss = (pred - target) ** 2  # 手写 MSE (逐元素)
                    task_loss = per_sample_loss[mask].mean() * task_weights.get(task, 1.0)
                    losses[task] = task_loss.item()
                    total_loss += task_loss
        return total_loss, losses
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        task_losses = {task: 0 for task in self.model.task_weights.keys()}
        count = 0
        
        for features, labels, valid_masks in self.train_loader:
            # 移动到设备
            features = {k: v.to(self.device) for k, v in features.items()}
            labels = {k: v.to(self.device) for k, v in labels.items()}
            valid_masks = {k: v.to(self.device) for k, v in valid_masks.items()}
            
            # 前向传播
            predictions = self.model(features)
            loss, task_losses_batch = self.compute_loss(predictions, labels, valid_masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for task, l in task_losses_batch.items():
                task_losses[task] += l
            count += 1
        
        # 平均损失
        avg_loss = total_loss / count
        avg_task_losses = {task: loss / count for task, loss in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = {task: [] for task in self.model.task_weights.keys()}
        all_targets = {task: [] for task in self.model.task_weights.keys()}
        all_masks = {task: [] for task in self.model.task_weights.keys()}
        with torch.no_grad():
            for features, labels, valid_masks in self.val_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = {k: v.to(self.device) for k, v in labels.items()}
                valid_masks = {k: v.to(self.device) for k, v in valid_masks.items()}
                predictions = self.model(features)
                loss, _ = self.compute_loss(predictions, labels, valid_masks)
                for task in predictions.keys():
                    if task in labels and task in valid_masks:
                        mask = valid_masks[task]
                        if mask.dim() > 1:
                            mask = mask.squeeze()
                        pred = predictions[task]
                        target = labels[task]
                        if pred.dim() > 1:
                            pred = pred.squeeze()
                        if target.dim() > 1:
                            target = target.squeeze()
                        if torch.any(mask):
                            sel_pred = pred[mask]
                            sel_target = target[mask]
                            sel_pred = torch.nan_to_num(sel_pred, nan=0.0, posinf=0.0, neginf=0.0)
                            sel_target = torch.nan_to_num(sel_target, nan=0.0, posinf=0.0, neginf=0.0)
                            all_predictions[task].extend(sel_pred.cpu().numpy().tolist())
                            all_targets[task].extend(sel_target.cpu().numpy().tolist())
                            all_masks[task].extend(mask[mask].cpu().numpy().tolist())
                total_loss += loss.item()
        metrics = {}
        for task in all_predictions.keys():
            if len(all_predictions[task]) > 0:
                pred = np.array(all_predictions[task], dtype=np.float32)
                target = np.array(all_targets[task], dtype=np.float32)
                finite_mask = np.isfinite(pred) & np.isfinite(target)
                pred = pred[finite_mask]
                target = target[finite_mask]
                n = int(pred.size)
                unique_count = int(len(np.unique(target))) if n > 0 else 0
                var = float(np.var(target)) if n > 0 else 0.0
                if n > 1:
                    mse = float(np.mean((pred - target) ** 2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(pred - target)))
                    if var > 0:
                        try:
                            r2 = float(r2_score(target, pred))
                        except Exception:
                            r2 = float('nan')
                    else:
                        r2 = None  # 方差为0，仅输出误差，不输出R²
                    metrics[task] = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'n': n, 'var': var, 'unique': unique_count}
                else:
                    metrics[task] = {'mse': None, 'rmse': None, 'mae': None, 'r2': None, 'n': n, 'var': var, 'unique': unique_count}
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics
    
    def train(self, epochs=150):
        """训练模型"""
        logger = logging.getLogger(__name__)
        logger.info(f"开始训练，共{epochs}个epoch")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_task_losses = self.train_epoch()
            
            # 验证
            val_loss, val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, self.output_dir / 'best_model.pth')
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': self.history
                }, self.output_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
            # 打印进度
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                       f"Time: {elapsed:.2f}s")
            
            # 打印每个任务的指标
            if val_metrics:
                def fmt(v):
                    if v is None:
                        return 'NA'
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        return 'NA'
                    return f"{v:.3f}"
                metrics_str = ", ".join([
                    f"{k}: r2={fmt(m.get('r2'))}, rmse={fmt(m.get('rmse'))}, var={fmt(m.get('var'))}, unique={m.get('unique',0)}, n={m.get('n',0)}" for k, m in val_metrics.items()
                ])
                logger.info(f"Validation metrics -> {metrics_str}")
        
        logger.info("训练完成！")
        return self.history

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练优化的AI生成说话人脸视频评价模型")
    parser.add_argument("--data_path", type=str, default="datasets/ac_final_processed.pkl",
                        help="处理后的数据文件路径")
    parser.add_argument("--output_dir", type=str, default="experiments/optimized_run1",
                        help="输出目录")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=150,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("加载处理后的数据...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    # 创建数据集
    train_dataset = TalkingFaceDataset(data, 'train')
    val_dataset = TalkingFaceDataset(data, 'val')
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")
    
    # 创建模型
    input_dims = {
        'visual': 163,
        'audio': 768,
        'keypoint': 1404,
        'au': 17
    }
    
    model = OptimizedMTLModel(input_dims)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, args.device, str(output_dir))
    
    # 开始训练
    history = trainer.train(args.epochs)
    
    # 保存训练结果
    with open(output_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("训练完成！")
    print(f"最佳模型保存在: {output_dir}/best_model.pth")
    print(f"训练历史保存在: {output_dir}/training_history.pkl")

if __name__ == "__main__":
    main()