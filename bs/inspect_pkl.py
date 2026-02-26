import pickle
import sys

# 增加pickle加载的数据大小限制
# 注意：这需要足够的内存，如果内存不足可能会失败
# pickle.DEFAULT_PROTOCOL = 4
# sys.setrecursionlimit(10000)

def inspect_ch_sims_pkl(pkl_path):
    """加载并探查CH-SIMS的预处理pkl文件"""
    print(f"Attempting to load {pkl_path}...")
    
    try:
        with open(pkl_path, 'rb') as f:
            # 使用pickle.load()来加载数据
            data = pickle.load(f)
        
        print("File loaded successfully!")
        
        # 探查数据结构
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of entries (videos): {len(data)}")
            
            # 查看第一个视频的数据结构
            first_key = list(data.keys())[0]
            print(f"\n--- Inspecting first entry: '{first_key}' ---")
            video_data = data[first_key]
            
            if isinstance(video_data, dict):
                for feature_name, feature_value in video_data.items():
                    print(f"  - Feature '{feature_name}':")
                    print(f"    Type: {type(feature_value)}")
                    # 通常特征是numpy数组
                    if hasattr(feature_value, 'shape'):
                        print(f"    Shape: {feature_value.shape}")
                    else:
                        print(f"    Value snippet: {str(feature_value)[:100]}")
            else:
                print(f"Content type of first entry: {type(video_data)}")

        else:
            print("Data is not a dictionary. Further inspection needed.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("This could be due to memory limitations or a different data format.")
        print("Consider running this on a machine with more RAM if it's a memory issue.")

if __name__ == '__main__':
    # ！！！请确保路径正确
    file_path = 'f:\\bs\\datasets\\unaligned-001.pkl'
    inspect_ch_sims_pkl(file_path)