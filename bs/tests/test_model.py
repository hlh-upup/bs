#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 模型测试

此脚本用于测试模型的基本功能。
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TalkingFaceEvaluationModel
from config import get_default_config
from utils import get_device


def test_model_forward():
    """测试模型前向传播"""
    print("测试模型前向传播...")
    
    # 加载配置
    config = get_default_config()
    
    # 获取设备
    device = get_device(use_cuda=torch.cuda.is_available())
    print(f"使用设备: {device}")
    
    # 创建模型
    model = TalkingFaceEvaluationModel(config)
    model.to(device)
    model.eval()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建随机输入
    batch_size = 2
    seq_len = 10
    
    # 视觉特征
    visual_dim = config['model']['visual_encoder']['output_dim']
    visual_features = torch.randn(batch_size, seq_len, visual_dim).to(device)
    
    # 音频特征
    audio_dim = config['model']['audio_encoder']['output_dim']
    audio_features = torch.randn(batch_size, seq_len, audio_dim).to(device)
    
    # 关键点特征
    keypoint_dim = config['model']['keypoint_encoder']['output_dim']
    keypoint_features = torch.randn(batch_size, seq_len, keypoint_dim).to(device)
    
    # AU特征
    au_dim = config['model']['au_encoder']['output_dim']
    au_features = torch.randn(batch_size, seq_len, au_dim).to(device)
    
    # 创建标签
    lip_sync_scores = torch.rand(batch_size, 1).to(device)
    expression_scores = torch.rand(batch_size, 1).to(device)
    audio_scores = torch.rand(batch_size, 1).to(device)
    cross_modal_scores = torch.rand(batch_size, 1).to(device)
    
    # 前向传播
    print("执行前向传播...")
    outputs = model(
        visual_features=visual_features,
        audio_features=audio_features,
        keypoint_features=keypoint_features,
        au_features=au_features
    )
    
    # 检查输出
    print("检查输出...")
    assert 'lip_sync' in outputs, "输出中缺少口型同步预测"
    assert 'expression' in outputs, "输出中缺少表情预测"
    assert 'audio' in outputs, "输出中缺少音频预测"
    assert 'cross_modal' in outputs, "输出中缺少跨模态预测"
    
    # 检查输出形状
    assert outputs['lip_sync'].shape == (batch_size, 1), f"口型同步预测形状错误: {outputs['lip_sync'].shape}"
    assert outputs['expression'].shape == (batch_size, 1), f"表情预测形状错误: {outputs['expression'].shape}"
    assert outputs['audio'].shape == (batch_size, 1), f"音频预测形状错误: {outputs['audio'].shape}"
    assert outputs['cross_modal'].shape == (batch_size, 1), f"跨模态预测形状错误: {outputs['cross_modal'].shape}"
    
    # 计算损失
    print("计算损失...")
    loss = model.compute_loss(
        outputs=outputs,
        targets={
            'lip_sync': lip_sync_scores,
            'expression': expression_scores,
            'audio': audio_scores,
            'cross_modal': cross_modal_scores
        }
    )
    
    # 检查损失
    assert isinstance(loss, dict), "损失应该是字典类型"
    assert 'total' in loss, "损失中缺少总损失"
    assert 'lip_sync' in loss, "损失中缺少口型同步损失"
    assert 'expression' in loss, "损失中缺少表情损失"
    assert 'audio' in loss, "损失中缺少音频损失"
    assert 'cross_modal' in loss, "损失中缺少跨模态损失"
    
    print("模型前向传播测试通过!")
    print(f"总损失: {loss['total'].item():.4f}")
    print(f"口型同步损失: {loss['lip_sync'].item():.4f}")
    print(f"表情损失: {loss['expression'].item():.4f}")
    print(f"音频损失: {loss['audio'].item():.4f}")
    print(f"跨模态损失: {loss['cross_modal'].item():.4f}")
    
    return True


def test_model_save_load():
    """测试模型保存和加载"""
    print("\n测试模型保存和加载...")
    
    # 加载配置
    config = get_default_config()
    
    # 获取设备
    device = get_device(use_cuda=torch.cuda.is_available())
    
    # 创建模型
    model = TalkingFaceEvaluationModel(config)
    model.to(device)
    
    # 保存模型
    save_path = "test_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")
    
    # 创建新模型
    new_model = TalkingFaceEvaluationModel(config)
    new_model.to(device)
    
    # 加载模型
    new_model.load_state_dict(torch.load(save_path, map_location=device))
    print("模型已加载")
    
    # 验证参数是否相同
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2), "加载的模型参数与原始模型不同"
    
    # 删除保存的模型
    os.remove(save_path)
    print(f"已删除测试模型文件: {save_path}")
    
    print("模型保存和加载测试通过!")
    
    return True


def main():
    """主函数"""
    print("开始测试AI生成说话人脸视频评价模型...\n")
    
    # 测试模型前向传播
    test_model_forward()
    
    # 测试模型保存和加载
    test_model_save_load()
    
    print("\n所有测试通过!")


if __name__ == "__main__":
    main()