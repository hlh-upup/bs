#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型

基于多任务学习框架的AI生成说话人脸视频评价模型，实现口型同步、表情自然度、
声音质量及跨模态一致性的细粒度联合评估。
"""

__version__ = '0.1.0'
__author__ = 'AI生成说话人脸视频评价模型团队'

# 导入主要模块
from models import TalkingFaceEvaluationModel
from data import TalkingFaceVideoDataset, create_dataloaders, prepare_data
from features import FeatureExtractor
from training import Trainer
from evaluation import Evaluator, VideoQualityEvaluator
from utils import setup_logging, load_config, set_seed, get_device
from config import get_default_config

__all__ = [
    'TalkingFaceEvaluationModel',
    'TalkingFaceVideoDataset',
    'create_dataloaders',
    'prepare_data',
    'FeatureExtractor',
    'Trainer',
    'Evaluator',
    'VideoQualityEvaluator',
    'setup_logging',
    'load_config',
    'set_seed',
    'get_device',
    'get_default_config'
]