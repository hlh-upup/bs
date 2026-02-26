import os
import pandas as pd
from tqdm import tqdm
import logging
import yaml
import torch
import scipy.integrate
import sys
import os
import pandas as pd
from tqdm import tqdm

# 猴子补丁：将 simps 替换为 simpson
if not hasattr(scipy.integrate, 'simps'):
    scipy.integrate.simps = scipy.integrate.simpson
import numpy as np
import pickle

# --- 新增的导入 ---
from features.extractor import FeatureExtractor
from features.enhanced_extractor import EnhancedFeatureExtractor
from utils.utils import setup_logging, get_device
# ------------------

# (setup_logging 函数的定义需要在这里，或者从 utils.utils 正确导入)
# 为了脚本独立性，我们可以在这里重新定义一个简化的版本
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("prepare_ch_sims_with_features.log"),
            logging.StreamHandler()
        ]
    )
def _safe_get_syncnet_confidence(all_features):
    """安全地从特征中提取syncnet置信度"""
    try:
        syncnet_data = all_features.get('syncnet', None)
        
        # 如果是字典格式
        if isinstance(syncnet_data, dict):
            return syncnet_data.get('confidence', 0.0)
        
        # 如果是numpy数组格式
        elif isinstance(syncnet_data, np.ndarray):
            # 假设第一个值是置信度
            if len(syncnet_data) > 0:
                return float(syncnet_data[0])
            else:
                return 0.0
        
        # 如果是其他格式，尝试转换为float
        elif syncnet_data is not None:
            try:
                return float(syncnet_data)
            except (ValueError, TypeError):
                return 0.0
        
        # 如果为None或其他情况
        else:
            return 0.0
            
    except Exception as e:
        logging.warning(f"提取syncnet置信度时出错: {e}")
        return 0.0

def prepare_ch_sims_with_features(base_dir, config, device, output_file):
    """
    准备 CH-SIMS 数据集，提取特征并保存为单个文件。

    Args:
        base_dir (str): CH-SIMS 数据集的根目录。
        config (dict): 项目的配置字典。
        device (torch.device): 计算设备。
        output_file (str): 输出的 pickle 文件路径。
    """
    try:
        meta_path = os.path.join(base_dir, 'meta.csv')
        raw_video_dir = os.path.join(base_dir, 'Raw')

        if not os.path.exists(meta_path):
            logging.error(f"错误: 在 '{meta_path}' 未找到 meta.csv 文件。")
            return False

        df = pd.read_csv(meta_path)
        logging.info(f"总共找到 {len(df)} 个视频片段。开始提取特征...")

        # 初始化增强特征提取器（带日志记录功能）
        log_dir = os.path.join(os.path.dirname(output_file), 'feature_extraction_logs')
        feature_extractor = EnhancedFeatureExtractor(config, device, log_dir)
        logging.info(f"初始化增强特征提取器，日志目录: {log_dir}")

        all_data = []
        skipped_count = 0
        # 添加缓存计数器
        CACHE_INTERVAL = 5    # 每处理10个视频保存一次

        for idx, row in enumerate(tqdm(df.iterrows(), total=df.shape[0], desc="处理视频并提取特征"), 1):
            _, row = row  # 解包enumerate的结果
            try:
                video_id_str = str(row['video_id']).strip()
                clip_id_num = int(row['clip_id'])
                unique_clip_id = f"{video_id_str}_{clip_id_num:05d}"

                # ... (这里保留之前版本中强大的文件查找逻辑) ...
                video_file_path = None
                possible_clip_ids = [f"{clip_id_num:05d}", f"{clip_id_num:04d}", f"{clip_id_num:03d}", str(clip_id_num)]
                possible_extensions = ['.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV']
                for clip_str in possible_clip_ids:
                    path_to_check_base = os.path.join(raw_video_dir, video_id_str, clip_str)
                    for ext in possible_extensions:
                        path_to_check = path_to_check_base + ext
                        if os.path.exists(path_to_check):
                            video_file_path = path_to_check
                            break
                    if video_file_path: break
                
                if video_file_path is None:
                    logging.warning(f"视频文件未找到，跳过: {unique_clip_id}")
                    skipped_count += 1
                    continue

                # --- 核心逻辑：提取所有特征（使用增强提取器，自动记录日志）---
                all_features = feature_extractor.extract_all_features(video_file_path)
                
                # 检查特征提取是否成功
                if not all_features:
                    logging.warning(f"视频 {unique_clip_id} 特征提取失败，跳过")
                    skipped_count += 1
                    continue
                # --------------------------------

                # 提取标签
                labels = {
                    'lip_sync_score': _safe_get_syncnet_confidence(all_features),
                    'expression_score': float(row.get('label_V', 0.0)),
                    'audio_quality_score': float(row.get('label_A', 0.0)),
                    'cross_modal_score': float(row.get('label', 0.0)),
                    'overall_score': float(row.get('label', 0.0))
                }

                # 将所有数据打包
                data_point = {
                    'video_id': unique_clip_id,
                    'split': row.get('mode', 'train'),
                    'features': all_features, # 保存所有提取的特征
                    'labels': labels
                }
                all_data.append(data_point)

                # 每处理10个视频保存一次缓存
                if len(all_data) % CACHE_INTERVAL == 0:
                    cache_file = output_file.replace('.pkl', f'_cache_{len(all_data)}.pkl')
                    with open(cache_file, 'wb') as f:
                        pickle.dump(all_data, f)
                    logging.info(f"已处理 {len(all_data)} 个视频，缓存保存到: {cache_file}")

            except Exception as e:
                logging.warning(f"处理记录 {row.get('video_id', 'N/A')}_{row.get('clip_id', 'N/A')} 时出错: {e}")
                skipped_count += 1
                continue

        # 打印和保存处理摘要
        feature_extractor.print_summary()
        feature_extractor.save_summary()
        
        # 获取详细统计信息
        stats = feature_extractor.get_statistics()
        logging.info(f"特征提取统计: 总计 {stats['total_processed']} 个视频")
        logging.info(f"成功提取 {stats['success_count']} 个，失败 {stats['failure_count']} 个")
        logging.info(f"成功率: {stats['success_rate']:.2f}%")
        logging.info(f"最终数据集包含 {len(all_data)} 条有效记录，跳过 {skipped_count} 条无效记录。")

        # 保存到单个文件
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        logging.info(f"所有数据已成功保存到: {output_file}")

        return True

    except Exception as e:
        import traceback
        logging.error(f"处理过程中发生未预期的错误: {e}")
        logging.error(f"错误类型: {type(e).__name__}")
        logging.error(f"详细错误信息: {str(e)}")
        logging.error(f"错误堆栈: \n{traceback.format_exc()}")
        # 检查是否是特定的键错误
        if isinstance(e, KeyError) and str(e) == "'features'":
            logging.error("可能是数据结构问题：'features' 键不存在于数据点中")
            if all_data and len(all_data) > 0:
                logging.error(f"最后一个成功处理的数据点结构: {list(all_data[-1].keys())}")
        return False

# 在主函数中修改这些参数
if __name__ == '__main__':
    # A4000优化设置
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)  # 使用85%显存
    
    # 增大缓存间隔，减少I/O
    CACHE_INTERVAL = 5  # 每5个视频保存一次，而不是10个
    
    # 设置环境变量优化
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # 加载配置
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取设备
    device = get_device()

    # --- 请根据您的路径进行配置 ---
    ch_sims_base_dir = r'f:\bs\datasets\ch-simsv2s'
    # 将所有处理好的数据（特征+标签）保存为这一个文件
    output_pickle_file = r'f:\bs\datasets\ch_sims_processed_data.pkl'
    # --------------------------------
    
    # 测试单个视频的特征提取
    # print("测试单个视频的特征提取...")
    # test_log_dir = r'f:\bs\test_logs'
    # feature_extractor = EnhancedFeatureExtractor(config, device, test_log_dir)
    # test_video_path = r'f:\bs\datasets\ch-simsv2s\Raw\video_0001\0001.mp4'
    
    # try:
    #     print(f"开始提取视频特征: {test_video_path}")
    #     features = feature_extractor.extract_all_features(test_video_path)
    #     # 打印详细特征信息
    #     print("特征提取成功!")
    #     print(f"提取的特征类型: {list(features.keys())}")
    #     for feature_type, feature_data in features.items():
    #         if hasattr(feature_data, 'shape'):
    #             print(f"  - {feature_type}: 形状 {feature_data.shape}")
    #             print(f"    数据类型: {feature_data.dtype}")
    #             print(f"    数值范围: [{np.min(feature_data):.4f}, {np.max(feature_data):.4f}]")
    #             print(f"    均值: {np.mean(feature_data):.4f}, 标准差: {np.std(feature_data):.4f}")
    #             if feature_data.ndim > 1:
    #                 print(f"    前5个时间步的特征维度统计:")
    #                 for i in range(min(5, feature_data.shape[0])):
    #                     frame_stats = f"      时间步{i}: 均值={np.mean(feature_data[i]):.4f}, 标准差={np.std(feature_data[i]):.4f}"
    #                     print(frame_stats)
    #         elif isinstance(feature_data, dict):
    #             print(f"  - {feature_type}: {list(feature_data.keys())}")
    #         else:
    #             print(f"  - {feature_type}: 形状 {feature_data.shape if hasattr(feature_data, 'shape') else '非数组'}")
        
    #     # 验证特征是否符合模型输入要求
    #     print("\n=== 模型兼容性检查 ===")
    #     expected_dims = {
    #         'visual': (150, 163),  # 实际提取的维度
    #         'audio': (150, 768),
    #         'keypoint': (150, 1404),
    #         'au': (150, 17),
    #         'syncnet': (2,),
    #         'consistency': (1,)
    #     }
        
    #     all_compatible = True
    #     for feature_type, expected_shape in expected_dims.items():
    #         if feature_type in features:
    #             feature_data = features[feature_type]
                
    #             # 处理字典类型的特征（如syncnet）
    #             if isinstance(feature_data, dict):
    #                 if feature_type == 'syncnet':
    #                     # 检查syncnet字典是否包含期望的键
    #                     expected_keys = ['sync_score', 'offset']
    #                     has_all_keys = all(key in feature_data for key in expected_keys)
    #                     status = "✓" if has_all_keys else "✗"
    #                     print(f"  {status} {feature_type}: 期望键 {expected_keys}, 实际键 {list(feature_data.keys())}")
    #                     if not has_all_keys:
    #                         all_compatible = False
    #                 else:
    #                     print(f"  ✗ {feature_type}: 意外的字典格式")
    #                     all_compatible = False
    #             # 处理数组类型的特征
    #             elif hasattr(feature_data, 'shape'):
    #                 actual_shape = feature_data.shape
    #                 is_compatible = actual_shape == expected_shape
    #                 status = "✓" if is_compatible else "✗"
    #                 print(f"  {status} {feature_type}: 期望 {expected_shape}, 实际 {actual_shape}")
    #                 if not is_compatible:
    #                     all_compatible = False
    #             else:
    #                 print(f"  ✗ {feature_type}: 未知数据类型 {type(feature_data)}")
    #                 all_compatible = False
    #         else:
    #             print(f"  ✗ {feature_type}: 缺失")
    #             all_compatible = False
        
    #     print(f"\n模型兼容性: {'通过' if all_compatible else '不通过'}")
    #     print("特征提取测试完成!")
        
    #     # 显示测试统计信息
    #     feature_extractor.print_summary()
    # except Exception as e:
    #     import traceback
    #     print(f"特征提取失败: {e}")
    #     print(f"错误类型: {type(e).__name__}")
    #     print(f"详细错误信息: {str(e)}")
    #     print(f"错误堆栈: \n{traceback.format_exc()}")
    
    # 运行完整数据集处理
    print("\n开始处理完整数据集...")
    success = prepare_ch_sims_with_features(ch_sims_base_dir, config, device, output_pickle_file)
    if success:
        print(f"数据准备完成，已保存到: {output_pickle_file}")
        logging.info("✅ 数据集准备和特征提取完成！")
    else:
        print("数据准备失败，请检查日志获取详细信息。")
        logging.error("❌ 处理失败！请检查日志文件了解详情。")