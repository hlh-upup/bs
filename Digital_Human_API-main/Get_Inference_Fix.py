@app.route('/Get_Inference', methods=['POST'])
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
        return jsonify(result="Failed")