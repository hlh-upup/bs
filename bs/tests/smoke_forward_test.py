#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最小冒烟测试：验证改进模型的前向传播在伪造输入下能正常运行并输出所有任务的标量向量。
运行：python -m tests.smoke_forward_test
"""

import torch
from models import create_model


def run_smoke():
    config = {
        'name': 'ImprovedMultiTaskTalkingFaceEvaluator',
        'visual_dim': 2048,
        'audio_dim': 768,
        'keypoint_dim': 1404,
        'au_dim': 17,
        'encoder_dim': 256,
        'dropout': 0.2,
        'transformer': {
            'num_layers': 2,
            'num_heads': 8,
            'dim_feedforward': 512,
            'dropout': 0.1,
        },
        'task_heads': {
            'lip_sync': {'hidden_dims': [128, 64], 'dropout': 0.2},
            'expression': {'hidden_dims': [128, 64], 'dropout': 0.2},
            'audio_quality': {'hidden_dims': [128, 64], 'dropout': 0.2},
            'cross_modal': {'hidden_dims': [128, 64], 'dropout': 0.2},
            'overall': {'hidden_dims': [128, 64], 'dropout': 0.2},
        }
    }
    model = create_model(config)
    B, T = 2, 10
    visual = torch.randn(B, T, config['visual_dim'])
    audio = torch.randn(B, T, config['audio_dim'])
    keypoint = torch.randn(B, T, config['keypoint_dim'])
    au = torch.randn(B, T, config['au_dim'])
    out = model(visual, audio, keypoint, au)
    assert set(out.keys()) == {'lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall'}
    for k, v in out.items():
        assert v.shape[0] == B, f"{k} batch mismatch {v.shape}"
        assert v.ndim in (1, 2), f"{k} ndim unexpected {v.ndim}"
    print('SMOKE OK:', {k: tuple(v.shape) for k, v in out.items()})


if __name__ == '__main__':
    run_smoke()
