#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt文件
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# 获取版本号
about = {}
with open(os.path.join("__init__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

setup(
    name="talking_face_evaluator",
    version=about["__version__"],
    author=about["__author__"],
    author_email="example@example.com",
    description="基于多任务学习框架的AI生成说话人脸视频评价模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/talking_face_evaluator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "talking_face_train=main:train",
            "talking_face_evaluate=main:evaluate",
            "talking_face_predict=main:predict",
        ],
    },
)