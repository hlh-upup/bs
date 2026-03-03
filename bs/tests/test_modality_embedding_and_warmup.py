#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试模态嵌入（Modality Embedding）和GradNorm热身期（Warmup）功能"""
import torch
import os
import sys

# 保证可以 import train_optimized
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from train_optimized import OptimizedMTLModel, Trainer


def test_modality_embeddings_exist():
    """测试模型包含可学习的模态嵌入参数"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    
    # 验证模态嵌入存在
    assert hasattr(model, 'modality_embeddings'), '模型缺少 modality_embeddings'
    for key in ['visual', 'audio', 'keypoint', 'au']:
        assert key in model.modality_embeddings, f'缺少模态嵌入: {key}'
        param = model.modality_embeddings[key]
        assert param.shape == (1, 1, 64), f'模态嵌入形状错误: {param.shape}'
        assert param.requires_grad, f'模态嵌入 {key} 不可学习'
    print('PASS: test_modality_embeddings_exist')


def test_modality_embeddings_affect_output():
    """测试模态嵌入实际影响模型输出"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    model.eval()
    
    B, T = 2, 10
    features = {
        'visual': torch.randn(B, T, 163),
        'audio': torch.randn(B, T, 768),
        'keypoint': torch.randn(B, T, 1404),
        'au': torch.randn(B, T, 17)
    }
    
    # 获取正常输出
    with torch.no_grad():
        out1 = model(features)
    
    # 将模态嵌入全部清零后获取输出
    with torch.no_grad():
        for key in model.modality_embeddings:
            model.modality_embeddings[key].zero_()
        out2 = model(features)
    
    # 输出应该不同（清零模态嵌入应该改变结果）
    any_diff = False
    for task in out1:
        if not torch.allclose(out1[task], out2[task], atol=1e-6):
            any_diff = True
            break
    # 注意：由于randn初始化，清零后输出几乎一定不同
    assert any_diff, '清零模态嵌入后输出未发生变化，模态嵌入未生效'
    print('PASS: test_modality_embeddings_affect_output')


def test_forward_pass_with_modality_embeddings():
    """测试带模态嵌入的前向传播正常运行"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    model.eval()
    
    B, T = 2, 10
    features = {
        'visual': torch.randn(B, T, 163),
        'audio': torch.randn(B, T, 768),
        'keypoint': torch.randn(B, T, 1404),
        'au': torch.randn(B, T, 17)
    }
    
    with torch.no_grad():
        predictions = model(features)
    
    # 验证所有任务都有输出
    expected_tasks = {'lip_sync', 'expression', 'audio_quality', 'cross_modal', 'overall'}
    assert set(predictions.keys()) == expected_tasks, f'任务输出不完整: {predictions.keys()}'
    
    # 验证输出形状
    for task, pred in predictions.items():
        assert pred.shape == (B,), f'{task} 输出形状错误: {pred.shape}'
    
    print('PASS: test_forward_pass_with_modality_embeddings')


def test_gradnorm_warmup_uniform_weights():
    """测试GradNorm热身期使用均匀权重"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    
    # 创建伪数据加载器（不实际用于训练）
    trainer = Trainer(model, None, None, 'cpu', '/tmp/test_warmup', warmup_epochs=5)
    
    # 构造伪预测和标签
    B = 4
    predictions = {task: torch.rand(B) for task in model.task_weights}
    targets = {task: torch.rand(B) for task in model.task_weights}
    valid_masks = {task: torch.ones(B, dtype=torch.bool) for task in model.task_weights}
    
    # 热身期内（epoch < warmup_epochs），权重应为均匀的1.0
    trainer.current_epoch = 0
    loss_warmup, losses_warmup = trainer.compute_loss(predictions, targets, valid_masks)
    
    # 热身期结束后（epoch >= warmup_epochs），使用模型配置的权重
    trainer.current_epoch = 5
    loss_dynamic, losses_dynamic = trainer.compute_loss(predictions, targets, valid_masks)
    
    # 由于模型权重不均匀（0.8, 1.2, 1.0, 1.5, 1.3），
    # 热身期和非热身期的总损失应不同
    assert abs(loss_warmup.item() - loss_dynamic.item()) > 1e-8, \
        f'热身期和非热身期损失相同: warmup={loss_warmup.item()}, dynamic={loss_dynamic.item()}'
    
    print('PASS: test_gradnorm_warmup_uniform_weights')


def test_gradnorm_warmup_boundary():
    """测试GradNorm热身期边界条件"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    trainer = Trainer(model, None, None, 'cpu', '/tmp/test_warmup_boundary', warmup_epochs=3)
    
    B = 4
    predictions = {task: torch.rand(B) for task in model.task_weights}
    targets = {task: torch.rand(B) for task in model.task_weights}
    valid_masks = {task: torch.ones(B, dtype=torch.bool) for task in model.task_weights}
    
    # epoch 0, 1, 2 应使用均匀权重
    losses_warmup = []
    for ep in range(3):
        trainer.current_epoch = ep
        loss, _ = trainer.compute_loss(predictions, targets, valid_masks)
        losses_warmup.append(loss.item())
    
    # 热身期内所有epoch应产生相同损失（因为权重都是1.0）
    for i in range(len(losses_warmup) - 1):
        assert abs(losses_warmup[i] - losses_warmup[i + 1]) < 1e-6, \
            f'热身期内损失不一致: epoch {i}={losses_warmup[i]}, epoch {i+1}={losses_warmup[i+1]}'
    
    # epoch 3 应使用动态权重
    trainer.current_epoch = 3
    loss_after, _ = trainer.compute_loss(predictions, targets, valid_masks)
    
    # 动态权重应产生不同的损失
    assert abs(losses_warmup[0] - loss_after.item()) > 1e-8, \
        '热身期后损失未变化，动态权重未生效'
    
    print('PASS: test_gradnorm_warmup_boundary')


def test_backward_with_modality_embeddings():
    """测试模态嵌入参数在反向传播中获得梯度"""
    input_dims = {'visual': 163, 'audio': 768, 'keypoint': 1404, 'au': 17}
    model = OptimizedMTLModel(input_dims, hidden_dim=64, num_layers=1, num_heads=4)
    
    B, T = 2, 10
    features = {
        'visual': torch.randn(B, T, 163),
        'audio': torch.randn(B, T, 768),
        'keypoint': torch.randn(B, T, 1404),
        'au': torch.randn(B, T, 17)
    }
    
    predictions = model(features)
    loss = sum(pred.sum() for pred in predictions.values())
    loss.backward()
    
    # 验证模态嵌入获得了梯度
    for key in ['visual', 'audio', 'keypoint', 'au']:
        param = model.modality_embeddings[key]
        assert param.grad is not None, f'模态嵌入 {key} 未获得梯度'
        assert torch.isfinite(param.grad).all(), f'模态嵌入 {key} 梯度包含非有限值'
    
    print('PASS: test_backward_with_modality_embeddings')


if __name__ == '__main__':
    test_modality_embeddings_exist()
    test_forward_pass_with_modality_embeddings()
    test_modality_embeddings_affect_output()
    test_backward_with_modality_embeddings()
    test_gradnorm_warmup_uniform_weights()
    test_gradnorm_warmup_boundary()
    print('\n所有测试通过!')
