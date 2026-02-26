import os
import pickle
import glob
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_current_data():
    """分析当前已处理的数据"""
    
    # 路径配置
    cache_dir = r'f:\bs\datasets'
    meta_path = r'f:\bs\datasets\ch-simsv2s\meta.csv'
    
    print("=== 数据处理状况分析 ===\n")
    
    # 1. 检查原始数据集大小
    try:
        df = pd.read_csv(meta_path)
        total_videos = len(df)
        print(f"📊 原始数据集总视频数: {total_videos}")
    except Exception as e:
        print(f"❌ 无法读取原始数据集: {e}")
        return
    
    # 2. 统计缓存文件
    cache_pattern = os.path.join(cache_dir, "ch_sims_processed_data_cache_*.pkl")
    cache_files = glob.glob(cache_pattern)
    
    if not cache_files:
        print("❌ 未找到任何缓存文件")
        return
    
    print(f"📁 找到缓存文件数量: {len(cache_files)}")
    
    # 3. 找到最大的缓存编号
    def extract_number(filename):
        try:
            basename = os.path.basename(filename)
            num_str = basename.replace("ch_sims_processed_data_cache_", "").replace(".pkl", "")
            return int(num_str)
        except:
            return 0
    
    max_cache_num = max(extract_number(f) for f in cache_files)
    print(f"🔢 最大缓存编号: {max_cache_num}")
    
    # 4. 统计已处理的唯一视频
    processed_ids = set()
    total_records = 0
    
    print("\n📈 正在统计已处理的视频...")
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                total_records += len(data)
                for item in data:
                    processed_ids.add(item['video_id'])
        except Exception as e:
            print(f"⚠️  读取 {os.path.basename(cache_file)} 时出错: {e}")
    
    unique_processed = len(processed_ids)
    progress_percentage = (unique_processed / total_videos) * 100
    
    print(f"\n=== 数据统计结果 ===")
    print(f"✅ 已处理唯一视频数: {unique_processed}")
    print(f"📦 总缓存记录数: {total_records}")
    print(f"📊 完成进度: {progress_percentage:.1f}%")
    print(f"⏳ 剩余视频数: {total_videos - unique_processed}")
    
    # 5. 检查数据质量（抽样检查几个缓存文件）
    print(f"\n=== 数据质量检查 ===")
    sample_files = sorted(cache_files, key=extract_number)[-3:]  # 检查最后3个文件
    
    feature_types = set()
    label_types = set()
    
    for cache_file in sample_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if data:
                    sample_item = data[0]
                    if 'features' in sample_item:
                        feature_types.update(sample_item['features'].keys())
                    if 'labels' in sample_item:
                        label_types.update(sample_item['labels'].keys())
        except Exception as e:
            print(f"⚠️  检查 {os.path.basename(cache_file)} 时出错: {e}")
    
    print(f"🎯 特征类型: {sorted(feature_types)}")
    print(f"🏷️  标签类型: {sorted(label_types)}")
    
    # 6. 估算磁盘使用情况
    total_size = 0
    for cache_file in cache_files:
        try:
            total_size += os.path.getsize(cache_file)
        except:
            pass
    
    size_gb = total_size / (1024**3)
    print(f"\n💾 缓存文件总大小: {size_gb:.2f} GB")
    
    # 7. 训练建议
    print(f"\n=== 训练建议 ===")
    if unique_processed >= 1500:
        print("✅ 数据量充足！可以开始训练")
        print(f"   - 已有 {unique_processed} 个视频样本")
        print(f"   - 建议按 8:1:1 比例划分训练/验证/测试集")
        print(f"   - 训练集约: {int(unique_processed * 0.8)} 个样本")
        print(f"   - 验证集约: {int(unique_processed * 0.1)} 个样本")
        print(f"   - 测试集约: {int(unique_processed * 0.1)} 个样本")
    elif unique_processed >= 1000:
        print("⚠️  数据量中等，可以尝试训练")
        print(f"   - 建议使用数据增强技术")
        print(f"   - 考虑使用预训练模型")
    else:
        print("❌ 数据量较少，建议继续处理更多视频")
    
    # 8. 下一步操作建议
    print(f"\n=== 下一步操作建议 ===")
    if progress_percentage >= 80:
        print("🎉 处理进度很高，建议：")
        print("   1. 合并现有缓存文件生成最终数据集")
        print("   2. 开始模型训练")
        print("   3. 如果磁盘空间不足，可以删除部分缓存文件")
    else:
        print("🔄 如果想要更多数据：")
        print("   1. 清理磁盘空间")
        print("   2. 继续处理剩余视频")
        print("   3. 或者直接使用现有数据开始训练")

if __name__ == '__main__':
    analyze_current_data()