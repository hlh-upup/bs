#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复后的Get_Inference接口实现
基于C#原版的前后端配合逻辑
"""

# 新的Get_Inference函数 - 与C#前端和Vue前端都兼容
new_get_inference_correct = '''@app.route('/Get_Inference', methods=['POST'])
def Get_Inference():
    """主推理接口：VITS + SadTalker/Wav2Lip 异步推理
    兼容C#和Vue前端，返回Audio_Video_Inference任务状态
    """
    POST_JSON = request.get_json() or {}
    user = POST_JSON.get('User')

    if not user:
        return jsonify(result="Failed")

    try:
        # 创建用户目录结构
        result_vits_user_path, result_sadtalker_user_path, result_wav2lip_user_path, save_user_path = Create_File(user)

        # 检查音频文件存在情况，决定推理方式
        ref_wav_path = os.path.join(save_user_path, "Ref_Wav.wav")
        user_wav_path = os.path.join(save_user_path, "User_Wav.wav")

        # 设置任务状态为进行中
        Task_State(save_user_path, "Audio_Video_Inference", False)

        # 根据音频文件选择推理方式
        if os.path.exists(user_wav_path):
            # 用户训练的音频 + Wav2Lip推理
            print(f"Starting User Audio + Wav2Lip inference for user: {user}")
            executor.submit(User_Wav_Wav2Lip_Inference, result_wav2lip_user_path, save_user_path)
        elif os.path.exists(ref_wav_path):
            # 参考音频 + VITS + SadTalker推理
            print(f"Starting VITS + SadTalker inference for user: {user}")
            executor.submit(VITS_Sadtalker_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)
        else:
            # 默认VITS + SadTalker推理
            print(f"Starting default VITS + SadTalker inference for user: {user}")
            executor.submit(VITS_Sadtalker_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)

        # 立即返回任务ID，前端开始轮询
        return jsonify(result="Audio_Video_Inference")

    except Exception as e:
        print(f"Get_Inference error for user {user}: {e}")
        return jsonify(result="Failed")'''

# 确保Get_State接口正确工作
get_state_check = '''
# Get_State接口应该已经存在，验证其正确性
# POST /Get_State
# 输入: {"User": "username", "Task": "Audio_Video_Inference"}
# 输出: {"result": "True"} | {"result": "False"} | {"result": "Failed"}
'''

if __name__ == "__main__":
    print("修复后的Get_Inference接口逻辑:")
    print(new_get_inference_correct)
    print("\n配合逻辑说明:")
    print("1. 前端调用 /Get_Inference")
    print("2. 后端立即返回 'Audio_Video_Inference' 任务ID")
    print("3. 前端开始轮询 /Get_State")
    print("4. 后端异步执行推理，完成后设置状态为True")
    print("5. 前端检测到True后继续下一步流程")