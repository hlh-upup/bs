#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复Get_Inference接口的脚本
"""

import os

# 文件路径
server_file = "/f/human-repo/Digital_Human_API-main1/Digital_Human_API-main/server1.4.0.py"

# 新的Get_Inference函数实现
new_get_inference = '''@app.route('/Get_Inference', methods=['POST'])
def Get_Inference():
    """主推理接口：VITS + SadTalker/Wav2Lip 异步推理
    与Vue前端预期一致，返回Audio_Video_Inference任务状态
    """
    POST_JSON = request.get_json() or {}
    user = POST_JSON.get('User')
    try:
        result_vits_user_path, result_sadtalker_user_path, result_wav2lip_user_path, save_user_path = Create_File(user)

        # 检查是否有用户训练的音频文件，决定使用哪种推理方式
        ref_wav_path = os.path.join(save_user_path, "Ref_Wav.wav")
        user_wav_path = os.path.join(save_user_path, "User_Wav.wav")

        if os.path.exists(user_wav_path):
            # 用户音频 + Wav2Lip推理
            Task_State(save_user_path, "Audio_Video_Inference", False)
            executor.submit(User_Wav_Wav2Lip_Inference, result_wav2lip_user_path, save_user_path)
        elif os.path.exists(ref_wav_path):
            # 参考音频 + VITS + SadTalker推理
            Task_State(save_user_path, "Audio_Video_Inference", False)
            executor.submit(VITS_Sadtalker_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)
        else:
            # 默认VITS + SadTalker推理
            Task_State(save_user_path, "Audio_Video_Inference", False)
            executor.submit(VITS_Sadtalker_Inference, result_vits_user_path, result_sadtalker_user_path, save_user_path)

        return jsonify(result="Audio_Video_Inference")
    except Exception as e:
        print(f"Get_Inference error: {e}")
        return jsonify(result="Failed")'''

def fix_get_inference():
    """修复Get_Inference函数"""
    try:
        # 读取原文件
        with open(server_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 找到需要替换的行数
        start_line = None
        end_line = None

        for i, line in enumerate(lines):
            if "@app.route('/Get_Inference', methods=['POST'])" in line:
                start_line = i
            elif start_line is not None and line.strip().startswith("@app.route") and i > start_line:
                end_line = i
                break
            elif start_line is not None and "return Get_Test_Inference()" in line:
                end_line = i + 1
                break

        if start_line is not None and end_line is not None:
            # 替换函数
            new_lines = lines[:start_line] + [new_get_inference + '\n\n'] + lines[end_line:]

            # 写回文件
            with open(server_file, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            print(f"✅ 成功修复Get_Inference函数 (第{start_line+1}-{end_line}行)")
            return True
        else:
            print("❌ 未找到Get_Inference函数")
            return False

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

if __name__ == "__main__":
    fix_get_inference()