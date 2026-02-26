#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 配置包初始化
"""

import os
from utils import load_config

# 获取默认配置文件路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# 加载默认配置
def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return load_config(DEFAULT_CONFIG_PATH)

__all__ = ['get_default_config', 'DEFAULT_CONFIG_PATH']