import torch
import sys

# 将模型文件的路径指向你的文件
model_path = 'f:/bs/models/pre-trained/stable_syncnet.pt'

print(f"Inspecting model file: {model_path}")

try:
    # 加载模型文件
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 检查checkpoint的类型并打印键
    if isinstance(checkpoint, dict):
        # 常见的情况是模型保存在一个字典中，键可能是 'state_dict' 或 'model'
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\nKeys found in 'state_dict':")
            for key in state_dict.keys():
                print(key)
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("\nKeys found in 'model':")
            for key in state_dict.keys():
                print(key)
        else:
            # 如果没有常见的键，直接打印字典的顶层键
            print("\nKeys found in the checkpoint dictionary:")
            for key in checkpoint.keys():
                print(key)
    else:
        # 有些模型文件可能直接就是 state_dict
        print("\nCheckpoint is not a dictionary. It might be the state_dict itself.")
        # 尝试把它当作一个 state_dict 来处理
        try:
            state_dict = checkpoint
            print("Keys found in the object:")
            for key in state_dict.keys():
                print(key)
        except Exception as e:
            print(f"Could not treat the object as a state_dict: {e}")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")