import os
import pickle
import numpy as np
from tqdm import tqdm
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_features_to_float32(data):
    """递归地将数据结构中的float64 numpy数组转换为float32。"""
    if isinstance(data, np.ndarray) and data.dtype == np.float64:
        return data.astype(np.float32)
    if isinstance(data, dict):
        return {k: convert_features_to_float32(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_features_to_float32(v) for v in data]
    return data

def analyze_labels_and_create_dataset(cache_file, output_file, validation_split=0.2, random_state=42):
    """
    分析标签分布，然后创建最终的、压缩的数据集。
    """
    if not os.path.exists(cache_file):
        print(f"错误：缓存文件 '{cache_file}' 不存在。")
        return

    print(f"正在从 '{cache_file}' 加载数据...")
    with open(cache_file, 'rb') as f:
        all_data = pickle.load(f)

    print(f"成功加载 {len(all_data)} 个样本。")

    # --- 新增：分析标签分布 ---
    print("\n--- 标签分布分析 ---")
    labels_df = pd.DataFrame([sample['labels'] for sample in all_data])
    total_samples = len(labels_df)

    for column in labels_df.columns:
        invalid_count = (labels_df[column] == -1.0).sum()
        valid_count = total_samples - invalid_count
        invalid_percentage = (invalid_count / total_samples) * 100
        print(f"标签 '{column}':")
        print(f"  - 有效样本数: {valid_count}")
        print(f"  - 无效样本数 (-1.0): {invalid_count} ({invalid_percentage:.2f}%)")
    print("------------------------")
    # --------------------------

    final_data = {
        'train': [],
        'val': [],
        'test': []
    }

    print("\n正在压缩特征并将数据分割到 train/val/test 集合中...")
    for sample in tqdm(all_data, desc="处理样本"):
        sample['features'] = convert_features_to_float32(sample['features'])
        split_type = sample.get('split', 'train')
        if split_type in final_data:
            final_data[split_type].append(sample)
        else:
            print(f"警告：未知的split类型 '{split_type}'，样本将被忽略。")

    print("\n数据集统计:")
    print(f"  - 训练集样本数: {len(final_data['train'])}")
    # --- 新增：如果验证集为空，则从训练集中拆分 ---
    if not final_data['val'] and validation_split > 0 and final_data['train']:
        print(f"\n检测到验证集为空，将从训练集中拆分 {validation_split*100:.0f}% 的数据作为验证集...")
        train_data = final_data['train']
        
        # 使用 train_test_split 进行拆分
        new_train_data, val_data = train_test_split(
            train_data, 
            test_size=validation_split, 
            random_state=random_state
        )
        
        final_data['train'] = new_train_data
        final_data['val'] = val_data
        print("拆分完成。")
    # ------------------------------------------

    print("\n最终数据集统计:")
    print(f"  - 训练集样本数: {len(final_data['train'])}")
    print(f"  - 验证集样本数: {len(final_data['val'])}")
    print(f"  - 测试集样本数: {len(final_data['test'])}")

    print(f"\n正在将最终数据集保存到 '{output_file}'...")
    with open(output_file, 'wb') as f:
        pickle.dump(final_data, f)

    original_size = os.path.getsize(cache_file) / (1024 * 1024)
    new_size = os.path.getsize(output_file) / (1024 * 1024)

    print("\n处理完成！")
    print(f"  - 原始文件大小: {original_size:.2f} MB")
    print(f"  - 新文件大小: {new_size:.2f} MB")
    print(f"  - 空间节省: {original_size - new_size:.2f} MB ({((original_size - new_size) / original_size) * 100:.2f}%)")

def cleanup_cache_files(directory):
    """删除目录中所有匹配 'ch_sims_processed_data_cache_*.pkl' 模式的文件。"""
    cache_files = glob.glob(os.path.join(directory, 'ch_sims_processed_data_cache_*.pkl'))
    if not cache_files:
        print("未找到需要清理的缓存文件。")
        return

    confirm = input(f"找到 {len(cache_files)} 个缓存文件。确定要删除它们吗？ (y/n): ")
    if confirm.lower() == 'y':
        total_reclaimed_space = 0
        for f in tqdm(cache_files, desc="删除缓存文件"):
            try:
                file_size = os.path.getsize(f)
                os.remove(f)
                total_reclaimed_space += file_size
            except Exception as e:
                print(f"删除文件 '{f}' 时出错: {e}")
        print(f"清理完成。总共释放了 {total_reclaimed_space / (1024*1024*1024):.2f} GB 的磁盘空间。")
    else:
        print("操作已取消。")

if __name__ == '__main__':
    base_dir = 'f:\\bs\\datasets'
    latest_cache_file = os.path.join(base_dir, 'ch_sims_processed_data_cache_1985.pkl')
    final_output_file = os.path.join(base_dir, 'ch_sims_final_dataset.pkl')

    # 现在调用函数时可以指定验证集比例
    analyze_labels_and_create_dataset(latest_cache_file, final_output_file, validation_split=0.2)

    # print("\n---")
    # cleanup_cache_files(base_dir)