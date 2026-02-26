#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI生成说话人脸视频评价模型 - 特征提取模块

实现视觉、音频、关键点和AU特征的提取功能。
使用py-feat库进行面部特征提取。
"""

# 导入基础库
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import librosa
import logging

# 尝试导入py-feat
try:
    import feat
    from feat import Detector
    PY_FEAT_AVAILABLE = True
except ImportError:
    PY_FEAT_AVAILABLE = False
    Detector = None
    print("Warning: py-feat not available. Face feature extraction may be limited.")

# 尝试导入MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    print("Warning: MediaPipe not available. Face mesh detection will be disabled.")
from tqdm import tqdm
import logging
import tempfile
import subprocess
import pickle

# 设置日志记录器
logger = logging.getLogger(__name__)

# 尝试导入transformers库，如果失败则设置为None
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers库导入失败，音频深度特征提取将不可用")
    Wav2Vec2Processor = None
    Wav2Vec2Model = None
    Wav2Vec2FeatureExtractor = None
    TRANSFORMERS_AVAILABLE = False
    
# 使用py-feat库进行面部特征提取
PYFEAT_AVAILABLE = PY_FEAT_AVAILABLE

from utils.syncnet.syncnet_instance import SyncNetInstance


class SyncNetFeatureExtractor:
    """SyncNet特征提取器

    使用SyncNet模型评估音视频同步性。

    Args:
        config (dict): 配置字典
        device (torch.device): 计算设备
    """

    def __init__(self, config, device):
        self.config = config['features']['syncnet']
        self.device = device
        self.model_path = self.config['model_path']
        self.batch_size = self.config['batch_size']
        self.v_shift = self.config['v_shift']

        # 加载SyncNet模型
        self.s = SyncNetInstance()
        self.s.load_parameters(self.model_path)
        self.s.to(self.device)
        self.s.eval()

    def extract_features(self, video_path, audio_path=None):
        """提取视频的SyncNet特征

        Args:
            video_path (str): 视频文件路径
            audio_path (str, optional): 音频文件路径，如果为None则从视频中提取

        Returns:
            dict: 包含置信度和偏移量的字典
            
        Raises:
            Exception: 当处理视频时发生错误
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for SyncNet extraction: {video_path}")
            
        confidence, offset = self.s.evaluate(video_path, audio_path)
        # 使用更合理的映射函数将置信度转换为0-5分制的同步评分
        # 使用sigmoid函数进行非线性映射，使得中等置信度区间更敏感
        import math
        # 将0-1的confidence映射到0-5，使用sigmoid增强中间区域的敏感度
        normalized_conf = max(0.0, min(1.0, float(confidence)))  # 确保在0-1范围内
        # 使用调整后的sigmoid函数：f(x) = 5 * (1 / (1 + exp(-10*(x-0.5))))
        # 这样0.5附近的变化会被放大，0和1附近相对平缓
        sync_score = 5.0 * (1.0 / (1.0 + math.exp(-10.0 * (normalized_conf - 0.5))))
        sync_score = max(0.0, min(5.0, sync_score))  # 确保在0-5范围内
        return {'sync_score': sync_score, 'offset': int(offset)}


# 导入其他库
import parselmouth

class VisualFeatureExtractor:
    """视觉特征提取器
    
    使用py-feat库进行面部特征提取，包括AU、情感、关键点等。
    
    Args:
        config (dict): 配置字典
        device (torch.device): 计算设备
    """
    
    def __init__(self, config, device=None):
        self.config = config
        self.device = device
        self.feature_dim = config['features']['visual']['feature_dim']
        self.target_fps = config['features']['visual']['target_fps']
        self.sequence_length = config['features']['visual']['sequence_length']
        
        # 检查py-feat是否可用
        if not PY_FEAT_AVAILABLE:
            raise ImportError("py-feat library is not available. Please install py-feat to use visual feature extraction.")
        
        # 初始化py-feat检测器
        try:
            # 配置py-feat检测器
            self.detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet", 
                au_model="svm",
                emotion_model="resmasknet",
                facepose_model="img2pose",
                device=device
            )
            logger.info("py-feat detector initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize py-feat detector: {e}")
    

    
    def extract_features(self, video_path):
        """提取视频的视觉特征
        
        使用py-feat库进行面部特征提取，包括AU、情感、关键点等。
        
        Args:
            video_path (str): 视频文件路径
        
        Returns:
            np.ndarray: 视觉特征，形状为 [sequence_length, feature_dim]
            
        Raises:
            FileNotFoundError: 当视频文件不存在时
            RuntimeError: 当特征提取失败时
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            # 使用py-feat检测视频
            results = self.detector.detect_video(video_path)
            
            if results is None or len(results) == 0:
                raise RuntimeError(f"No faces detected in video: {video_path}")
            
            # 提取各种特征
            features_list = []
            
            # AU特征
            au_columns = [col for col in results.columns if col.startswith('AU')]
            if au_columns:
                au_data = results[au_columns]
                # 确保数据是numpy数组，处理可能的PyTorch张量
                if hasattr(au_data.iloc[0, 0], 'detach'):
                    au_data = au_data.apply(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else x)
                au_features = au_data.values
                features_list.append(au_features)
            
            # 情感特征
            emotion_columns = [col for col in results.columns if col in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]
            if emotion_columns:
                emotion_data = results[emotion_columns]
                # 确保数据是numpy数组，处理可能的PyTorch张量
                if hasattr(emotion_data.iloc[0, 0], 'detach'):
                    emotion_data = emotion_data.apply(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else x)
                emotion_features = emotion_data.values
                features_list.append(emotion_features)
            
            # 关键点特征
            landmark_columns = [col for col in results.columns if col.startswith('x_') or col.startswith('y_')]
            if landmark_columns:
                landmark_data = results[landmark_columns]
                # 确保数据是numpy数组，处理可能的PyTorch张量
                if hasattr(landmark_data.iloc[0, 0], 'detach'):
                    landmark_data = landmark_data.apply(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else x)
                landmark_features = landmark_data.values
                features_list.append(landmark_features)
            
            # 头部姿态特征
            pose_columns = [col for col in results.columns if col in ['pose_Rx', 'pose_Ry', 'pose_Rz']]
            if pose_columns:
                pose_data = results[pose_columns]
                # 确保数据是numpy数组，处理可能的PyTorch张量
                if hasattr(pose_data.iloc[0, 0], 'detach'):
                    pose_data = pose_data.apply(lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else x)
                pose_features = pose_data.values
                features_list.append(pose_features)
            
            # 合并所有特征
            if not features_list:
                raise RuntimeError(f"No valid features extracted from video: {video_path}")
                
            visual_features = np.concatenate(features_list, axis=1)
            
            # 调整序列长度
            visual_features = self._adjust_sequence_length(visual_features)
            
            return visual_features
            
        except Exception as e:
            raise RuntimeError(f"Error extracting visual features from {video_path}: {e}")
    

    
    def _adjust_sequence_length(self, features):
        """调整特征序列长度"""
        if features is None or len(features) == 0:
            return np.zeros((self.sequence_length, self.feature_dim))
            
        if len(features) > self.sequence_length:
            # 如果特征太多，进行均匀采样
            indices = np.linspace(0, len(features) - 1, self.sequence_length, dtype=int)
            features = features[indices]
        elif len(features) < self.sequence_length:
            # 如果特征太少，进行填充
            if len(features) > 0:
                padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
                features = np.vstack([features, padding])
            else:
                features = np.zeros((self.sequence_length, self.feature_dim))
        
        return features


class AudioFeatureExtractor:
    """音频特征提取器
    
    使用 Hubert 模型提取音频特征，支持torch 2.5.1版本。
    
    Args:
        config (dict): 配置字典
        device (torch.device): 计算设备
    """
    
    def __init__(self, config, device=None):
        self.config = config['features']['audio']
        self.device = device
        self.sample_rate = self.config['sample_rate']
        self.sequence_length = self.config['sequence_length']

        # 检查 transformers 库是否可用
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 库不可用，无法使用 Hubert 模型进行音频特征提取")

        # 加载 Hubert 模型
        model_name = self.config['model']
        if model_name != 'hubert':
            raise ValueError(f"只支持 Hubert 模型，当前配置的模型: {model_name}")
        
        try:
            # 优先使用包含完整文件的本地模型
            if os.path.exists("models/hubert-base") and os.path.exists("models/hubert-base/preprocessor_config.json"):
                model_path = "models/hubert-base"
                logger.info("使用 hubert-base 模型（包含完整配置文件）")
            elif os.path.exists("models/hubert-base-safe") and os.path.exists("models/hubert-base-safe/preprocessor_config.json"):
                model_path = "models/hubert-base-safe"
                logger.info("使用 hubert-base-safe 模型")
            else:
                model_path = "facebook/hubert-base-ls960"
                logger.info("使用在线 Hubert 模型（本地模型文件不完整）")
            
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            self.model = Wav2Vec2Model.from_pretrained(model_path)
            
            # 确保与torch 2.5.1兼容
            self.model = self.model.to(device)
            self.model.eval()
            
            # 设置模型为推理模式
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.deep_feature_dim = self.model.config.hidden_size
            logger.info(f"成功加载Hubert模型用于音频特征提取，模型路径: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载 Hubert 模型失败: {e}")
    


    def _extract_features(self, audio_waveform):
        """使用 Hubert 模型提取音频特征"""
        try:
            # 确保输入是正确的格式
            if isinstance(audio_waveform, np.ndarray):
                audio_waveform = audio_waveform.astype(np.float32)
            
            with torch.no_grad():
                # 使用torch.inference_mode()以获得更好的性能（torch 2.5.1特性）
                with torch.inference_mode():
                    inputs = self.processor(audio_waveform, sampling_rate=self.sample_rate, return_tensors="pt")
                    # 移动到设备，确保与torch 2.5.1兼容
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    # 确保返回的是CPU上的numpy数组
                    features = outputs.last_hidden_state.squeeze().detach().cpu().numpy()
                    return features
        except Exception as e:
            raise RuntimeError(f"使用 Hubert 模型提取音频特征失败: {e}")

    def extract_features(self, video_path, audio_path=None):
        """使用 Hubert 模型提取视频的音频特征
        
        Args:
            video_path (str): 视频文件路径
            audio_path (str, optional): 音频文件路径，如果为None则从视频中提取
            
        Returns:
            np.ndarray: Hubert 特征，形状为 [sequence_length, feature_dim]
            
        Raises:
            FileNotFoundError: 当视频或音频文件不存在时
            RuntimeError: 当提取特征过程中发生错误
        """
        temp_audio_path = None
        try:
            # 加载音频
            if audio_path and os.path.exists(audio_path):
                audio_waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            else:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found for audio extraction: {video_path}")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                command = [
                    'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                    '-ar', str(self.sample_rate), '-ac', '1', '-y', temp_audio_path
                ]
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                audio_waveform, _ = librosa.load(temp_audio_path, sr=self.sample_rate, mono=True)

            # 使用 Hubert 模型提取特征
            features = self._extract_features(audio_waveform)
            
            # 统一序列长度
            if len(features) > self.sequence_length:
                indices = np.linspace(0, len(features) - 1, self.sequence_length, dtype=int)
                features = features[indices]
            elif len(features) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(features), features.shape[1]))
                features = np.vstack([features, padding])

            return features
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary audio file {temp_audio_path}: {e}")


class KeypointFeatureExtractor:
    """关键点特征提取器
    
    使用 MediaPipe 人脸网格提取面部关键点特征。
    
    Args:
        config (dict): 配置字典
        device (torch.device): 计算设备
    """
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.feature_dim = config['features']['keypoint']['feature_dim']
        self.target_fps = config['features']['keypoint']['target_fps']
        self.sequence_length = config['features']['keypoint']['sequence_length']
        
        # 初始化MediaPipe人脸网格
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("成功初始化 MediaPipe 人脸网格")
        except Exception as e:
            raise RuntimeError(f"初始化 MediaPipe 人脸网格失败: {e}")
    
    def extract_features(self, video_path):
        """提取视频的关键点特征
        
        Args:
            video_path (str): 视频文件路径
        
        Returns:
            np.ndarray: 关键点特征，形状为 [sequence_length, feature_dim]
            
        Raises:
            FileNotFoundError: 当视频文件不存在时
            IOError: 当无法打开视频文件时
            Exception: 当处理视频时发生错误
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for keypoint extraction: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for keypoint extraction: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        if self.target_fps > 0:
            sample_interval = max(1, round(fps / self.target_fps))
        else:
            # 根据序列长度确定采样间隔
            sample_interval = max(1, round(frame_count / self.sequence_length))
        
        # 提取特征
        features = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样帧
            if frame_idx % sample_interval == 0:
                # 转换为RGB（MediaPipe需要RGB格式）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # 使用MediaPipe检测关键点
                    results = self.face_mesh.process(frame_rgb)
                    
                    if results.multi_face_landmarks:
                        # 获取第一个检测到的人脸的关键点
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # 提取关键点坐标
                        keypoints = []
                        for landmark in face_landmarks.landmark:
                            keypoints.extend([landmark.x, landmark.y, landmark.z])
                        
                        # 如果关键点数量与特征维度不匹配，进行调整
                        if len(keypoints) > self.feature_dim:
                            keypoints = keypoints[:self.feature_dim]
                        elif len(keypoints) < self.feature_dim:
                            keypoints.extend([0] * (self.feature_dim - len(keypoints)))
                        
                        features.append(keypoints)
                    else:
                        # 如果MediaPipe未检测到人脸，抛出异常
                        if len(features) == 0 and frame_idx > sample_interval * 5:  # 允许前几帧没有人脸
                            raise ValueError(f"MediaPipe 未能在视频 {video_path} 的多个帧中检测到人脸")
                except Exception as e:
                    # 如果连续多帧处理失败，抛出异常
                    if len(features) == 0 and frame_idx > sample_interval * 5:  # 允许前几帧处理失败
                        raise RuntimeError(f"使用 MediaPipe 处理关键点提取失败: {e}")

            
            frame_idx += 1
        
        cap.release()
        
        # 调整特征序列长度
        features = np.array(features)
        if len(features) > self.sequence_length:
            # 如果特征太多，进行均匀采样
            indices = np.linspace(0, len(features) - 1, self.sequence_length, dtype=int)
            features = features[indices]
        elif len(features) < self.sequence_length:
            # 如果特征太少，进行填充
            padding = np.zeros((self.sequence_length - len(features), self.feature_dim))
            features = np.vstack([features, padding])
        
        return features


class AUFeatureExtractor:
    """AU特征提取器
    
    使用MediaPipe面部关键点提取面部动作单元(AU)特征。
    基于面部关键点的几何关系计算AU特征，而不依赖于py-feat库。
    
    Args:
        config (dict): 配置字典
        device (torch.device): 计算设备
    """
    
    def __init__(self, config, device):
        self.config = config['features']['au']
        self.device = device
        self.sequence_length = self.config['sequence_length']
        self.feature_dim = self.config.get('feature_dim', 17)  # 从config获取或设为默认值
        
        # 初始化MediaPipe人脸网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 定义关键AU对应的关键点索引
        # 这些索引基于MediaPipe Face Mesh的468个关键点
        # 眉毛相关的AU (AU1, AU2, AU4)
        self.inner_brow_indices = [65, 295]  # 左右内眉毛
        self.mid_brow_indices = [66, 296]    # 左右中眉毛
        self.outer_brow_indices = [105, 334] # 左右外眉毛
        self.brow_reference = [8]            # 鼻尖作为参考点
        
        # 眼睛相关的AU (AU5, AU6, AU7)
        self.upper_eyelid_indices = [159, 386]  # 左右上眼睑
        self.lower_eyelid_indices = [145, 374]  # 左右下眼睑
        self.eye_corner_indices = [133, 362]    # 左右眼角
        
        # 鼻子相关的AU (AU9, AU10)
        self.nose_indices = [5, 6]  # 鼻子关键点
        
        # 嘴巴相关的AU (AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU28)
        self.mouth_corner_indices = [61, 291]    # 左右嘴角
        self.upper_lip_indices = [13, 14]        # 上唇中心
        self.lower_lip_indices = [17, 18]        # 下唇中心
        self.lip_reference = [152, 377]          # 参考点
        
        logger.info("初始化AU特征提取器（基于MediaPipe面部关键点）")
    
    def _calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _calculate_au_features(self, landmarks):
        """根据面部关键点计算AU特征
        
        Args:
            landmarks (np.ndarray): 面部关键点坐标，形状为 [468, 3]
            
        Returns:
            np.ndarray: AU特征向量
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(self.feature_dim)
        
        features = []
        
        # 获取参考点
        nose_tip = landmarks[self.brow_reference[0]]
        
        # 计算AU1 (内眉上扬)
        left_inner_brow = landmarks[self.inner_brow_indices[0]]
        right_inner_brow = landmarks[self.inner_brow_indices[1]]
        au1 = (self._calculate_distance(left_inner_brow, nose_tip) + 
               self._calculate_distance(right_inner_brow, nose_tip)) / 2
        features.append(au1)
        
        # 计算AU2 (外眉上扬)
        left_outer_brow = landmarks[self.outer_brow_indices[0]]
        right_outer_brow = landmarks[self.outer_brow_indices[1]]
        au2 = (self._calculate_distance(left_outer_brow, nose_tip) + 
               self._calculate_distance(right_outer_brow, nose_tip)) / 2
        features.append(au2)
        
        # 计算AU4 (眉头皱起)
        left_mid_brow = landmarks[self.mid_brow_indices[0]]
        right_mid_brow = landmarks[self.mid_brow_indices[1]]
        au4 = self._calculate_distance(left_mid_brow, right_mid_brow)
        features.append(au4)
        
        # 计算AU5 (上眼睑抬起)
        left_upper_eyelid = landmarks[self.upper_eyelid_indices[0]]
        right_upper_eyelid = landmarks[self.upper_eyelid_indices[1]]
        left_lower_eyelid = landmarks[self.lower_eyelid_indices[0]]
        right_lower_eyelid = landmarks[self.lower_eyelid_indices[1]]
        au5 = (self._calculate_distance(left_upper_eyelid, left_lower_eyelid) + 
               self._calculate_distance(right_upper_eyelid, right_lower_eyelid)) / 2
        features.append(au5)
        
        # 计算AU6 (脸颊抬起)
        left_eye_corner = landmarks[self.eye_corner_indices[0]]
        right_eye_corner = landmarks[self.eye_corner_indices[1]]
        left_mouth_corner = landmarks[self.mouth_corner_indices[0]]
        right_mouth_corner = landmarks[self.mouth_corner_indices[1]]
        au6 = (self._calculate_distance(left_eye_corner, left_mouth_corner) + 
               self._calculate_distance(right_eye_corner, right_mouth_corner)) / 2
        features.append(au6)
        
        # 计算AU7 (眼睑紧闭)
        au7 = 1.0 / (au5 + 1e-6)  # 眼睛开度的倒数
        features.append(au7)
        
        # 计算AU9 (鼻子皱起)
        nose_point1 = landmarks[self.nose_indices[0]]
        nose_point2 = landmarks[self.nose_indices[1]]
        au9 = self._calculate_distance(nose_point1, nose_point2)
        features.append(au9)
        
        # 计算AU10 (上唇上扬)
        upper_lip = landmarks[self.upper_lip_indices[0]]
        au10 = self._calculate_distance(upper_lip, nose_tip)
        features.append(au10)
        
        # 计算AU12 (嘴角上扬/微笑)
        left_lip_ref = landmarks[self.lip_reference[0]]
        right_lip_ref = landmarks[self.lip_reference[1]]
        au12 = (self._calculate_distance(left_mouth_corner, left_lip_ref) + 
                self._calculate_distance(right_mouth_corner, right_lip_ref)) / 2
        features.append(au12)
        
        # 计算AU14 (嘴角紧绷)
        au14 = 1.0 / (au12 + 1e-6)  # 嘴角上扬的倒数
        features.append(au14)
        
        # 计算AU15 (嘴角下垂)
        au15 = au14 * 0.8  # 简化计算，与AU14相关
        features.append(au15)
        
        # 计算AU17 (下巴抬起)
        lower_lip = landmarks[self.lower_lip_indices[0]]
        au17 = self._calculate_distance(lower_lip, nose_tip)
        features.append(au17)
        
        # 计算AU20 (嘴唇水平拉伸)
        au20 = self._calculate_distance(left_mouth_corner, right_mouth_corner)
        features.append(au20)
        
        # 计算AU23 (嘴唇紧闭)
        au23 = 1.0 / (self._calculate_distance(upper_lip, lower_lip) + 1e-6)
        features.append(au23)
        
        # 计算AU25 (嘴唇分开)
        au25 = self._calculate_distance(upper_lip, lower_lip)
        features.append(au25)
        
        # 计算AU26 (下巴下垂)
        au26 = au25 * 1.2  # 简化计算，与AU25相关
        features.append(au26)
        
        # 计算AU28 (嘴唇内吸)
        au28 = 1.0 / (au20 + 1e-6)  # 嘴唇水平拉伸的倒数
        features.append(au28)
        
        # 归一化特征
        features = np.array(features)
        if np.max(features) > 0:
            features = features / np.max(features)
        
        return features
    
    def extract_features(self, video_path):
        """提取视频的AU特征
        
        使用MediaPipe面部关键点计算AU特征。
        
        Args:
            video_path (str): 视频文件路径
            
        Returns:
            np.ndarray: 形状为(sequence_length, feature_dim)的AU特征矩阵
            
        Raises:
            FileNotFoundError: 当视频文件不存在时
            IOError: 当无法打开视频文件时
            Exception: 当处理视频时发生错误
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for AU extraction: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file for AU extraction: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        target_fps = self.config.get('target_fps', 25)
        if target_fps > 0:
            sample_interval = max(1, round(fps / target_fps))
        else:
            # 根据序列长度确定采样间隔
            sample_interval = max(1, round(frame_count / self.sequence_length))
        
        # 提取特征
        features = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样帧
            if frame_idx % sample_interval == 0:
                # 转换为RGB（MediaPipe需要RGB格式）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用MediaPipe检测关键点
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    # 获取第一个检测到的人脸的关键点
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 提取关键点坐标
                    landmarks = np.array([
                        [landmark.x, landmark.y, landmark.z] 
                        for landmark in face_landmarks.landmark
                    ])
                    
                    # 计算AU特征
                    au_features = self._calculate_au_features(landmarks)
                    features.append(au_features)
                else:
                    # 如果未检测到人脸，记录警告并跳过此帧
                    logger.warning(f"No face detected in frame {frame_idx}")
                    # 如果连续多帧未检测到人脸，可能需要抛出异常
                    if len(features) == 0 and frame_idx > sample_interval * 5:  # 允许前几帧没有人脸
                        raise ValueError(f"No face detected in multiple frames of {video_path}")
            
            frame_idx += 1
        
        cap.release()
        
        # 调整特征序列长度
        features = np.array(features)
        if len(features) > self.sequence_length:
            # 如果特征太多，进行均匀采样
            indices = np.linspace(0, len(features) - 1, self.sequence_length, dtype=int)
            features = features[indices]
        elif len(features) < self.sequence_length:
            # 如果特征太少，进行填充
            padding = np.zeros((self.sequence_length - len(features), self.feature_dim))
            features = np.vstack([features, padding])
        
        logger.info(f"成功提取AU特征: {video_path}, 形状: {features.shape}")
        return features


class CrossModalConsistencyExtractor:
    """跨模态一致性特征提取器"""
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def extract_features(self, audio_features, visual_features):
        """计算音视频特征之间的相关性作为一致性分数"""
        try:
            # 确保特征是numpy数组
            if isinstance(audio_features, dict):
                audio_feat = audio_features.get('deep', audio_features.get('acoustic'))
            else:
                audio_feat = audio_features
            
            if audio_feat is None or visual_features is None:
                return {'correlation': 0.0}

            # 取平均值得到一维向量
            audio_vec = np.mean(audio_feat, axis=0)
            visual_vec = np.mean(visual_features, axis=0)

            # 确保长度一致
            min_len = min(len(audio_vec), len(visual_vec))
            audio_vec = audio_vec[:min_len]
            visual_vec = visual_vec[:min_len]

            # 计算皮尔逊相关系数
            correlation = np.corrcoef(audio_vec, visual_vec)[0, 1]
            return {'correlation': float(correlation) if not np.isnan(correlation) else 0.0}
        except Exception as e:
            logger.error(f"Error calculating cross-modal consistency: {e}")
            return {'correlation': 0.0}


class FeatureExtractor:
    """特征提取器

    整合所有特征提取器，提供统一的接口。
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.extractors = {}
        feature_config = config.get('features', {})
        if 'visual' in feature_config:
            self.extractors['visual'] = VisualFeatureExtractor(config, device)
        if 'audio' in feature_config:
            self.extractors['audio'] = AudioFeatureExtractor(config, device)
        if 'keypoint' in feature_config: # 修正了keypoints到keypoint
            self.extractors['keypoint'] = KeypointFeatureExtractor(config, device)
        if 'au' in feature_config:
            self.extractors['au'] = AUFeatureExtractor(config, device)
        if 'syncnet' in feature_config:
            self.extractors['syncnet'] = SyncNetFeatureExtractor(config, device)
        if 'consistency' in feature_config:
            self.extractors['consistency'] = CrossModalConsistencyExtractor(config, device)

    def extract_all_features(self, video_path):
        """提取指定视频的所有特征"""
        all_features = {}
        audio_path = None
        audio_extraction_success = False

        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            # 返回空特征字典
            for name in self.extractors.keys():
                all_features[name] = {}
            return all_features

        # 预提取音频，避免重复提取
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio_path = temp_audio.name
            command = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            audio_extraction_success = True
        except Exception as e:
            logger.error(f"Failed to pre-extract audio from {video_path}: {e}")
            audio_path = None  # 确保后续流程知道音频提取失败
            audio_extraction_success = False

        for name, extractor in self.extractors.items():
            logger.info(f"Extracting {name} features for {video_path}...")
            try:
                if name == 'syncnet':
                    if audio_extraction_success and audio_path and os.path.exists(audio_path):
                        features = extractor.extract_features(video_path, audio_path)
                        # 对于 SyncNet，将字典转换为数值数组
                        if isinstance(features, dict):
                            features = np.array([features.get('sync_score', 0.0), features.get('offset', 0.0)])
                    else:
                        raise ValueError(f"Cannot extract SyncNet features due to audio extraction failure")
                elif name == 'audio':
                    if audio_extraction_success and audio_path and os.path.exists(audio_path):
                        features = extractor.extract_features(video_path, audio_path)
                    else:
                        # 音频提取失败时，尝试直接从视频中提取
                        logger.warning(f"Attempting to extract audio directly from video due to pre-extraction failure")
                        features = extractor.extract_features(video_path, None)
                elif name == 'consistency':
                    # 这个提取器依赖于其他特征
                    if 'audio' not in all_features or 'visual' not in all_features:
                        raise ValueError(f"Cannot extract consistency features: required features missing")
                    audio_feats = all_features.get('audio')
                    visual_feats = all_features.get('visual')
                    features = extractor.extract_features(audio_feats, visual_feats)
                    # 将字典转换为数值数组
                    if isinstance(features, dict):
                        features = np.array([features.get('correlation', 0.0)])
                else:
                    features = extractor.extract_features(video_path)
                
                all_features[name] = features
            except Exception as e:
                # 记录错误并向上抛出异常
                logger.error(f"Error extracting '{name}' features for {video_path}: {e}")
                raise Exception(f"Failed to extract {name} features: {e}")

        # 清理临时音频文件
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary audio file {audio_path}: {e}")

        return all_features

    def process_video_batch(self, video_paths, output_dir):
        """批量处理视频
        
        Args:
            video_paths (list): 视频文件路径列表
            output_dir (str): 输出目录
            
        Raises:
            Exception: 当处理视频时发生错误
        """
        # 创建输出目录
        output_feature_types = [ft for ft in ['visual', 'audio', 'keypoint', 'au', 'syncnet'] if ft in self.extractors]
        for feature_type in output_feature_types:
            os.makedirs(os.path.join(output_dir, feature_type), exist_ok=True)
        
        # 跟踪处理结果
        processed_videos = 0
        failed_videos = []
        
        # 处理每个视频
        for video_path in tqdm(video_paths, desc="Extracting features"):
            try:
                # 获取视频ID
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                
                # 检查是否已处理
                if all(os.path.exists(os.path.join(output_dir, feature_type, f"{video_id}.npy")) 
                       for feature_type in output_feature_types):
                    logger.info(f"Video {video_id} already processed, skipping")
                    processed_videos += 1
                    continue
                
                # 提取特征
                features = self.extract_all_features(video_path)
                # 保存特征
                output_path = os.path.join(output_dir, f"{video_id}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(features, f)
                logger.info(f"Saved features for {video_path} to {output_path}")
                processed_videos += 1
            except Exception as e:
                error_msg = f"Error processing video {video_path}: {e}"
                logger.error(error_msg)
                failed_videos.append((video_path, str(e)))
        
        # 处理完成后报告结果
        if failed_videos:
            failure_report = "\n".join([f"{path}: {error}" for path, error in failed_videos])
            raise Exception(f"Failed to process {len(failed_videos)} out of {len(video_paths)} videos. " 
                           f"Successfully processed: {processed_videos}. \nFailures:\n{failure_report}")


# 文件结束