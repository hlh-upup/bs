# 修复后的VITS_Sadtalker_Inference函数
def VITS_Sadtalker_Inference(result_vits_user_path, result_sadtalker_user_path, save_user_path):
    try:
        print("开始VITS+Sadtalker推理流程...")

        DH = VITS_Sadtalker_Join(result_vits_user_path, result_sadtalker_user_path, save_user_path)
        sad_parames_yaml_path, vits_parames_yaml_path, _ = Get_Parmes(save_user_path)

        print("设置模型参数...")
        DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)

        ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
        with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
            data = json.load(f)
        ref_text = data["Text"]
        print(f"参考音频路径: {ref_wav_path}")
        print(f"参考文本: {ref_text}")

        # VITS推理（即使有问题也要继续）
        try:
            print("开始VITS推理...")
            DH.Inference_VITS(ref_wav_path,ref_text)
            print("VITS推理完成！")
        except Exception as e:
            print(f"VITS推理出现异常: {e}")
            print("但继续执行SadTalker推理...")

        # 强制执行SadTalker推理
        try:
            imag_path = os.path.join(save_user_path,"Image.png")
            print(f"开始SadTalker推理，图像路径: {imag_path}")

            # 确保图像文件存在
            if not os.path.exists(imag_path):
                print(f"错误：图像文件不存在 {imag_path}")
                raise FileNotFoundError(f"Image file not found: {imag_path}")

            # 确保音频文件JSON存在
            audio_json_path = os.path.join(save_user_path, "Audio_save_path.json")
            if not os.path.exists(audio_json_path):
                print(f"错误：音频文件JSON不存在 {audio_json_path}")
                raise FileNotFoundError(f"Audio JSON file not found: {audio_json_path}")

            print("执行SadTalker推理...")
            DH.Inference_SadTalker(imag_path)
            print("SadTalker推理完成！")

        except Exception as e:
            print(f"SadTalker推理出现异常: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 标记任务完成
        print("设置任务状态为完成...")
        Task_State(save_user_path, "Audio_Video_Inference", True)
        print("VITS+Sadtalker推理流程全部完成！")

    except Exception as e:
        print(f"VITS_Sadtalker_Inference 整体失败: {e}")
        import traceback
        traceback.print_exc()
        # 即使失败也要设置状态，避免前端无限等待
        Task_State(save_user_path, "Audio_Video_Inference", False)
        raise