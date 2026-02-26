#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU优化的特征提取器

主要优化策略：
1. 批量推理以提高GPU利用率
2. 内存池管理避免频繁分配
3. 异步数据加载
4. 模型预热和缓存
5. 混合精度计算
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import librosa
from typing import List, Dict, Tuple, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from contextlib import contextmanager

from .extractor import (
    VisualFeatureExtractor, AudioFeatureExtractor, 
    KeypointFeatureExtractor, AUFeatureExtractor,
    SyncNetFeatureExtractor
)

logger = logging.getLogger(__name__)

class VideoDataset(Dataset):
    """视频数据集，用于批量加载"""
    
    def __init__(self, video_paths: List[str], target_fps: int = 25, max_frames: int = 150):
        self.video_paths = video_paths
        self.target_fps = target_fps
        self.max_frames = max_frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        try:
            frames = self._load_video_frames(video_path)
            return {
                'frames': frames,
                'video_path': video_path,
                'success': True
            }
        except Exception as e:
            logger.error(f"加载视频失败 {video_path}: {e}")
            # 返回空帧
            dummy_frames = torch.zeros((self.max_frames, 3, 224, 224))
            return {
                'frames': dummy_frames,
                'video_path': video_path,
                'success': False
            }
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """加载视频帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.target_fps))
            
            frame_count = 0
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 预处理帧
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
                    frames.append(frame)
                
                frame_count += 1
            
            # 填充或截断到固定长度
            if len(frames) < self.max_frames:
                # 填充最后一帧
                last_frame = frames[-1] if frames else np.zeros((3, 224, 224))
                while len(frames) < self.max_frames:
                    frames.append(last_frame)
            else:
                frames = frames[:self.max_frames]
            
            return torch.tensor(np.stack(frames), dtype=torch.float32)
            
        finally:
            cap.release()

class GPUOptimizedExtractor:
    """GPU优化的特征提取器"""
    
    def __init__(self, config, device, batch_size=8, num_workers=4, use_amp=True):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # 初始化各个提取器
        self._init_extractors()
        
        # 性能监控
        self.processing_times = []
        self.gpu_memory_usage = []
        
        # 预热模型
        self._warmup_models()
        
        logger.info(f"GPU优化提取器初始化完成，批次大小: {batch_size}, 混合精度: {use_amp}")
    
    def _init_extractors(self):
        """初始化特征提取器"""
        feature_config = self.config.get('features', {})
        self.extractors = {}
        
        if 'visual' in feature_config:
            self.extractors['visual'] = VisualFeatureExtractor(self.config, self.device)
        if 'audio' in feature_config:
            self.extractors['audio'] = AudioFeatureExtractor(self.config, self.device)
        if 'keypoint' in feature_config:
            self.extractors['keypoint'] = KeypointFeatureExtractor(self.config, self.device)
        if 'au' in feature_config:
            self.extractors['au'] = AUFeatureExtractor(self.config, self.device)
        if 'syncnet' in feature_config:
            self.extractors['syncnet'] = SyncNetFeatureExtractor(self.config, self.device)
    
    def _warmup_models(self):
        """预热模型以优化性能"""
        logger.info("开始模型预热...")
        
        # 创建虚拟输入进行预热
        dummy_frames = torch.randn(1, 150, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            for name, extractor in self.extractors.items():
                try:
                    if hasattr(extractor, 'model') and extractor.model is not None:
                        if name == 'visual':
                            # 预热视觉模型
                            dummy_input = dummy_frames[0, 0:1]  # 单帧
                            _ = extractor.model(dummy_input)
                        elif name == 'audio':
                            # 预热音频模型
                            dummy_audio = torch.randn(1, 16000 * 3).to(self.device)  # 3秒音频
                            if hasattr(extractor.model, 'feature_extractor'):
                                inputs = extractor.model.feature_extractor(
                                    dummy_audio.cpu().numpy()[0], 
                                    sampling_rate=16000, 
                                    return_tensors="pt"
                                )
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                                _ = extractor.model(**inputs)
                    logger.info(f"预热完成: {name}")
                except Exception as e:
                    logger.warning(f"预热失败 {name}: {e}")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        logger.info("模型预热完成")
    
    @contextmanager
    def _gpu_memory_monitor(self):
        """GPU内存监控上下文管理器"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            self.processing_times.append(end_time - start_time)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / 1024**2  # MB
                self.gpu_memory_usage.append(memory_used)
    
    def extract_visual_features_batch(self, video_frames_batch: torch.Tensor) -> torch.Tensor:
        """批量提取视觉特征"""
        if 'visual' not in self.extractors:
            return torch.zeros((video_frames_batch.size(0), 150, 163))
        
        extractor = self.extractors['visual']
        batch_features = []
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    for video_frames in video_frames_batch:
                        # video_frames: [T, C, H, W]
                        video_frames = video_frames.to(self.device)
                        
                        # 批量处理帧
                        frame_features = []
                        for i in range(0, video_frames.size(0), 8):  # 每次处理8帧
                            batch_frames = video_frames[i:i+8]
                            features = extractor.model(batch_frames)
                            frame_features.append(features.cpu())
                        
                        # 合并特征
                        all_features = torch.cat(frame_features, dim=0)
                        batch_features.append(all_features)
            else:
                for video_frames in video_frames_batch:
                    video_frames = video_frames.to(self.device)
                    
                    frame_features = []
                    for i in range(0, video_frames.size(0), 8):
                        batch_frames = video_frames[i:i+8]
                        features = extractor.model(batch_frames)
                        frame_features.append(features.cpu())
                    
                    all_features = torch.cat(frame_features, dim=0)
                    batch_features.append(all_features)
        
        return torch.stack(batch_features)
    
    def extract_audio_features_batch(self, video_paths: List[str]) -> List[torch.Tensor]:
        """批量提取音频特征"""
        if 'audio' not in self.extractors:
            return [torch.zeros((150, 768)) for _ in video_paths]
        
        extractor = self.extractors['audio']
        batch_features = []
        
        # 并行加载音频
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            audio_futures = [executor.submit(self._load_audio, path) for path in video_paths]
            audio_data = [future.result() for future in audio_futures]
        
        # 批量处理音频
        with torch.no_grad():
            for audio in audio_data:
                if audio is not None:
                    try:
                        if self.use_amp:
                            with torch.cuda.amp.autocast():
                                features = self._extract_single_audio_features(audio, extractor)
                        else:
                            features = self._extract_single_audio_features(audio, extractor)
                        batch_features.append(features)
                    except Exception as e:
                        logger.error(f"音频特征提取失败: {e}")
                        batch_features.append(torch.zeros((150, 768)))
                else:
                    batch_features.append(torch.zeros((150, 768)))
        
        return batch_features
    
    def _load_audio(self, video_path: str) -> Optional[np.ndarray]:
        """加载音频数据"""
        try:
            # 使用librosa直接从视频提取音频
            audio, sr = librosa.load(video_path, sr=16000, duration=6.0)  # 最多6秒
            return audio
        except Exception as e:
            logger.error(f"音频加载失败 {video_path}: {e}")
            return None
    
    def _extract_single_audio_features(self, audio: np.ndarray, extractor) -> torch.Tensor:
        """提取单个音频的特征"""
        # 使用HuBERT模型提取特征
        inputs = extractor.model.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = extractor.model(**inputs)
            features = outputs.last_hidden_state.squeeze(0)  # [T, D]
        
        # 调整到固定长度
        target_length = 150
        if features.size(0) > target_length:
            # 均匀采样
            indices = torch.linspace(0, features.size(0) - 1, target_length, dtype=torch.long)
            features = features[indices]
        elif features.size(0) < target_length:
            # 填充
            padding = torch.zeros((target_length - features.size(0), features.size(1)))
            features = torch.cat([features.cpu(), padding], dim=0)
        
        return features.cpu()
    
    def extract_features_batch(self, video_paths: List[str]) -> List[Dict]:
        """批量提取所有特征"""
        with self._gpu_memory_monitor():
            # 创建数据集和数据加载器
            dataset = VideoDataset(video_paths)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=min(self.num_workers, 2),  # 限制数据加载器的工作进程
                pin_memory=True
            )
            
            all_results = []
            
            for batch in dataloader:
                batch_video_paths = batch['video_path']
                batch_frames = batch['frames']  # [B, T, C, H, W]
                batch_success = batch['success']
                
                # 提取视觉特征
                visual_features = None
                if 'visual' in self.extractors:
                    try:
                        visual_features = self.extract_visual_features_batch(batch_frames)
                    except Exception as e:
                        logger.error(f"批量视觉特征提取失败: {e}")
                
                # 提取音频特征
                audio_features = None
                if 'audio' in self.extractors:
                    try:
                        audio_features = self.extract_audio_features_batch(batch_video_paths)
                    except Exception as e:
                        logger.error(f"批量音频特征提取失败: {e}")
                
                # 组装结果
                for i, video_path in enumerate(batch_video_paths):
                    if not batch_success[i]:
                        all_results.append({})
                        continue
                    
                    features = {}
                    
                    if visual_features is not None:
                        features['visual'] = visual_features[i].detach().cpu().numpy()
                    
                    if audio_features is not None:
                        features['audio'] = audio_features[i].detach().cpu().numpy()
                    
                    # 其他特征使用原有方法单独提取
                    try:
                        if 'keypoint' in self.extractors:
                            features['keypoint'] = self.extractors['keypoint'].extract_features(video_path)
                        
                        if 'au' in self.extractors:
                            features['au'] = self.extractors['au'].extract_features(video_path)
                        
                        if 'syncnet' in self.extractors:
                            # SyncNet需要音频文件
                            audio_path = self._extract_temp_audio(video_path)
                            if audio_path:
                                syncnet_features = self.extractors['syncnet'].extract_features(video_path, audio_path)
                                if isinstance(syncnet_features, dict):
                                    features['syncnet'] = np.array([
                                        syncnet_features.get('sync_score', 0.0),
                                        syncnet_features.get('offset', 0.0)
                                    ])
                                else:
                                    features['syncnet'] = syncnet_features
                                os.unlink(audio_path)
                    
                    except Exception as e:
                        logger.error(f"其他特征提取失败 {video_path}: {e}")
                    
                    all_results.append(features)
                
                # 清理GPU内存
                torch.cuda.empty_cache()
            
            return all_results
    
    def _extract_temp_audio(self, video_path: str) -> Optional[str]:
        """提取临时音频文件"""
        import tempfile
        import subprocess
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            command = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return audio_path
        except Exception as e:
            logger.error(f"临时音频提取失败: {e}")
            return None
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if not self.processing_times:
            return {}
        
        stats = {
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'min_processing_time': min(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'batches_processed': len(self.processing_times)
        }
        
        if self.gpu_memory_usage:
            stats.update({
                'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage),
                'max_gpu_memory_mb': max(self.gpu_memory_usage),
                'total_gpu_memory_mb': sum(self.gpu_memory_usage)
            })
        
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 重置统计信息
        self.processing_times.clear()
        self.gpu_memory_usage.clear()
