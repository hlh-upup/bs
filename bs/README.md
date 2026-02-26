# AI生成说话人脸视频评价模型

## 项目概述

本项目实现了一个基于多任务学习（Multi-Task Learning, MTL）框架的AI生成说话人脸视频评价模型，能够对口型同步、表情自然度、声音质量及跨模态一致性等维度进行细粒度跨模态特征对齐与联合评估。

## 功能特点

- 多维度评价：同时评估口型同步、表情自然度、声音质量及跨模态一致性
- 跨模态对齐：通过Transformer架构实现音频与视频特征的细粒度对齐
- 可解释性：提供评价结果的可视化和解释
- 资源优化：采用特征预计算、低分辨率处理等策略，适应16GB显存限制

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (GPU加速)
- Windows操作系统

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── config/                   # 配置文件
│   └── config.yaml           # 主配置文件
├── data/                     # 数据处理相关
│   ├── dataset.py            # 数据集定义
│   ├── preprocess.py         # 数据预处理
│   └── feature_extraction.py # 特征提取
├── models/                   # 模型定义
│   ├── model.py              # 主模型架构
│   ├── fusion.py             # 多模态融合模块
│   └── heads.py              # 多任务预测头
├── utils/                    # 工具函数
│   ├── metrics.py            # 评价指标
│   ├── visualization.py      # 可视化工具
│   └── losses.py             # 损失函数
└── main.py                   # 主程序入口
```

## 使用方法

### 数据预处理

```bash
python data/preprocess.py --input_dir path/to/videos --output_dir path/to/processed_data
```

### 特征提取

```bash
python data/feature_extraction.py --input_dir path/to/processed_data --output_dir path/to/features
```

### 模型训练

```bash
python main.py --mode train --config config/config.yaml
```

### 模型评估

```bash
python main.py --mode eval --config config/config.yaml --checkpoint path/to/model.pth
```

### 视频质量评价

```bash
python main.py --mode predict --video path/to/video.mp4 --checkpoint path/to/model.pth
```