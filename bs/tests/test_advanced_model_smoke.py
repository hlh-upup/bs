#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最小冒烟测试：AdvancedMTLTalkingFaceEvaluator 前向+loss+反向梯度检查"""
import torch
import os
import sys

# 保证可以 import models
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.advanced_mtl_model import AdvancedMTLTalkingFaceEvaluator

def run_smoke():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'visual_dim': 100,
        'audio_dim': 200,
        'keypoint_dim': 50,
        'au_dim': 17,
        'encoder_dim': 128,
        'dropout': 0.1,
        'transformer': {
            'num_layers': 2,
            'num_heads': 4,
            'dim_feedforward': 256,
            'dropout': 0.1
        },
        'advanced': {
            'task_weighting': 'learned',
            'consistency': {'mode': 'std', 'weight': 0.05}
        },
        'task_heads': {
            'score_min': 1.0,
            'score_max': 5.0,
            'out_activation': 'sigmoid'
        }
    }

    model = AdvancedMTLTalkingFaceEvaluator(config).to(device)
    B, T = 4, 20
    vis = torch.randn(B, T, config['visual_dim'], device=device)
    aud = torch.randn(B, T, config['audio_dim'], device=device)
    kpt = torch.randn(B, T, config['keypoint_dim'], device=device)
    au = torch.randn(B, T, config['au_dim'], device=device)

    preds = model(vis, aud, kpt, au)
    assert all(t in preds for t in ['lip_sync','expression','audio_quality','cross_modal','overall']), '缺少任务输出'

    targets = {f'{t}_score': torch.rand(B, 1, device=device) * 4 + 1 for t in ['lip_sync','expression','audio_quality','cross_modal','overall']}
    losses = model.compute_loss(preds, targets, epoch=0)
    total = sum(losses.values())
    total.backward()

    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print('Forward OK, loss keys:', list(losses.keys()))
    print('Total loss:', float(total.item()))
    print('Any finite grad:', grad_ok)
    assert grad_ok, '未检测到有效梯度'

if __name__ == '__main__':
    run_smoke()
