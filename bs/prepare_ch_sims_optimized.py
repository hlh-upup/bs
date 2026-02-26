#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版本的CH-SIMS数据集特征提取脚本

主要优化点：
1. 批量处理视频以提高GPU利用率
2. 多进程并行处理
3. 内存优化和缓存管理
4. 更高效的I/O操作
"""

import os
import pandas as pd
from tqdm import tqdm
import logging
import yaml
import torch
import scipy.integrate
import numpy as np
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import partial
import gc

# 猴子补丁：将 simps 替换为 simpson
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson

from features.extractor import FeatureExtractor
from features.enhanced_extractor import EnhancedFeatureExtractor
from utils.utils import setup_logging, get_device

class OptimizedFeatureExtractor:
    """优化的特征提取器，支持批量处理和并行化"""
    
    def __init__(self, config, device, log_dir="logs", batch_size=4, num_workers=2):
        self.config = config
        self.device = device
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 创建基础提取器
        self.base_extractor = EnhancedFeatureExtractor(config, device, log_dir)
        
        # 性能统计
        self.processing_times = []
        self.gpu_utilization = []
        
    def extract_features_batch(self, video_paths):
        """批量提取特征"""
        batch_features = []
        batch_start_time = time.time()
        
        # 预分配GPU内存
        torch.cuda.empty_cache()
        
        for video_path in video_paths:
            try:
                features = self.base_extractor.extract_all_features(video_path)
                batch_features.append((video_path, features))
            except Exception as e:
                logging.error(f"批量处理中提取特征失败 {video_path}: {e}")
                batch_features.append((video_path, {}))
        
        batch_time = time.time() - batch_start_time
        self.processing_times.append(batch_time)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()
        
        return batch_features
    
    def process_video_batch_parallel(self, video_batch):
        """并行处理视频批次"""
        results = []
        
        # 使用线程池处理I/O密集型操作
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 将批次分成更小的子批次
            sub_batches = [video_batch[i:i+self.batch_size] 
                          for i in range(0, len(video_batch), self.batch_size)]
            
            # 并行处理子批次
            futures = [executor.submit(self.extract_features_batch, sub_batch) 
                      for sub_batch in sub_batches]
            
            for future in futures:
                try:
                    batch_results = future.result(timeout=300)  # 5分钟超时
                    results.extend(batch_results)
                except Exception as e:
                    logging.error(f"批次处理超时或失败: {e}")
        
        return results
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_batch_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'min_batch_time': min(self.processing_times),
            'max_batch_time': max(self.processing_times),
            'batches_processed': len(self.processing_times)
        }

def prepare_ch_sims_optimized(base_dir, config, device, output_file, 
                             batch_size=8, num_workers=4, use_cache=True):
    """
    优化版本的CH-SIMS数据集准备函数
    
    Args:
        base_dir (str): CH-SIMS数据集根目录
        config (dict): 配置字典
        device (torch.device): 计算设备
        output_file (str): 输出文件路径
        batch_size (int): 批处理大小
        num_workers (int): 并行工作进程数
        use_cache (bool): 是否使用缓存
    """
    try:
        meta_path = os.path.join(base_dir, 'meta.csv')
        raw_video_dir = os.path.join(base_dir, 'Raw')
        
        if not os.path.exists(meta_path):
            logging.error(f"错误: 在 '{meta_path}' 未找到 meta.csv 文件。")
            return False
        
        df = pd.read_csv(meta_path)
        logging.info(f"总共找到 {len(df)} 个视频片段。开始优化特征提取...")
        
        # 初始化优化的特征提取器
        log_dir = os.path.join(os.path.dirname(output_file), 'feature_extraction_logs')
        feature_extractor = OptimizedFeatureExtractor(
            config, device, log_dir, batch_size, num_workers
        )
        
        # 检查缓存
        cache_file = output_file.replace('.pkl', '_cache.pkl')
        processed_videos = set()
        all_data = []
        
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    all_data = cached_data.get('data', [])
                    processed_videos = set(cached_data.get('processed_videos', []))
                logging.info(f"从缓存加载了 {len(all_data)} 条记录")
            except Exception as e:
                logging.warning(f"缓存加载失败: {e}")
        
        # 准备待处理的视频列表
        video_tasks = []
        skipped_count = 0
        
        for _, row in df.iterrows():
            try:
                video_id_str = str(row['video_id']).strip()
                clip_id_num = int(row['clip_id'])
                unique_clip_id = f"{video_id_str}_{clip_id_num:05d}"
                
                # 跳过已处理的视频
                if unique_clip_id in processed_videos:
                    continue
                
                # 查找视频文件
                video_file_path = None
                possible_clip_ids = [f"{clip_id_num:05d}", f"{clip_id_num:04d}", 
                                   f"{clip_id_num:03d}", str(clip_id_num)]
                possible_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
                
                for clip_str in possible_clip_ids:
                    path_to_check_base = os.path.join(raw_video_dir, video_id_str, clip_str)
                    for ext in possible_extensions:
                        path_to_check = path_to_check_base + ext
                        if os.path.exists(path_to_check):
                            video_file_path = path_to_check
                            break
                    if video_file_path:
                        break
                
                if video_file_path is None:
                    logging.warning(f"视频文件未找到，跳过: {unique_clip_id}")
                    skipped_count += 1
                    continue
                
                video_tasks.append({
                    'video_path': video_file_path,
                    'unique_clip_id': unique_clip_id,
                    'row': row
                })
                
            except Exception as e:
                logging.warning(f"准备任务时出错: {e}")
                skipped_count += 1
                continue
        
        logging.info(f"准备处理 {len(video_tasks)} 个新视频")
        
        # 批量处理视频
        processed_count = 0
        total_batches = (len(video_tasks) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(video_tasks), batch_size), 
                     desc="批量处理视频", total=total_batches):
            batch_tasks = video_tasks[i:i+batch_size]
            batch_video_paths = [task['video_path'] for task in batch_tasks]
            
            # 批量提取特征
            batch_results = feature_extractor.process_video_batch_parallel(batch_video_paths)
            
            # 处理批次结果
            for j, (video_path, features) in enumerate(batch_results):
                if j >= len(batch_tasks):
                    break
                    
                task = batch_tasks[j]
                unique_clip_id = task['unique_clip_id']
                row = task['row']
                
                if not features:
                    logging.warning(f"视频 {unique_clip_id} 特征提取失败，跳过")
                    skipped_count += 1
                    continue
                
                # 提取标签
                labels = {
                    'lip_sync_score': features.get('syncnet', {}).get('confidence', 0.0),
                    'expression_score': float(row.get('label_V', 0.0)),
                    'audio_quality_score': float(row.get('label_A', 0.0)),
                    'cross_modal_score': float(row.get('label', 0.0)),
                    'overall_score': float(row.get('label', 0.0))
                }
                
                # 将数据打包
                data_point = {
                    'video_id': unique_clip_id,
                    'split': row.get('mode', 'train'),
                    'features': features,
                    'labels': labels
                }
                all_data.append(data_point)
                processed_videos.add(unique_clip_id)
                processed_count += 1
            
            # 定期保存缓存
            if (i // batch_size + 1) % 10 == 0:  # 每10个批次保存一次
                if use_cache:
                    cache_data = {
                        'data': all_data,
                        'processed_videos': list(processed_videos)
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)
                    logging.info(f"已保存缓存，当前处理了 {len(all_data)} 条记录")
        
        # 获取性能统计
        perf_stats = feature_extractor.get_performance_stats()
        logging.info(f"性能统计: {perf_stats}")
        
        logging.info(f"最终数据集包含 {len(all_data)} 条有效记录，跳过 {skipped_count} 条无效记录。")
        logging.info(f"新处理了 {processed_count} 个视频")
        
        # 保存最终结果
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        logging.info(f"所有数据已成功保存到: {output_file}")
        
        # 清理缓存文件
        if use_cache and os.path.exists(cache_file):
            os.remove(cache_file)
            logging.info("已清理缓存文件")
        
        return True
        
    except Exception as e:
        import traceback
        logging.error(f"优化处理过程中发生错误: {e}")
        logging.error(f"错误堆栈: \n{traceback.format_exc()}")
        return False

def monitor_gpu_usage():
    """监控GPU使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_used = torch.cuda.memory_allocated(0)
        gpu_utilization = (gpu_memory_used / gpu_memory) * 100
        
        logging.info(f"GPU内存使用: {gpu_memory_used / 1024**3:.2f}GB / {gpu_memory / 1024**3:.2f}GB ({gpu_utilization:.1f}%)")
        return gpu_utilization
    return 0

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("prepare_ch_sims_optimized.log"),
            logging.StreamHandler()
        ]
    )
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取设备
    device = get_device()
    logging.info(f"使用设备: {device}")
    
    # 监控GPU状态
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        monitor_gpu_usage()
    
    # 配置路径
    ch_sims_base_dir = r'f:\bs\datasets\ch-simsv2s'
    output_pickle_file = r'f:\bs\datasets\ch_sims_processed_data_optimized.pkl'
    
    # 优化参数
    batch_size = 8  # 根据GPU内存调整
    num_workers = 4  # 根据CPU核心数调整
    
    logging.info(f"开始优化处理，批次大小: {batch_size}, 工作进程数: {num_workers}")
    
    start_time = time.time()
    success = prepare_ch_sims_optimized(
        ch_sims_base_dir, config, device, output_pickle_file,
        batch_size=batch_size, num_workers=num_workers, use_cache=True
    )
    
    total_time = time.time() - start_time
    
    if success:
        logging.info(f"✅ 优化处理完成！总耗时: {total_time:.2f}秒")
        logging.info(f"数据已保存到: {output_pickle_file}")
        monitor_gpu_usage()
    else:
        logging.error(f"❌ 优化处理失败！耗时: {total_time:.2f}秒")