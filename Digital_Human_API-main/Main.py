import sys
sys.path.append("SadTalker")
from SadTalker.Inference import SadTalker_Model

sys.path.append("VITS")
sys.path.append("VITS/GPT_SoVITS")
from VITS.Inference import GPT_SoVITS_Model
from VITS.train import GPT_SoVITS_Tarin

sys.path.append("Easy_Wav2Lip")
from Easy_Wav2Lip.Motion_Inference import Wav2Lip_Model

from util.PPT2Video import Ppt_2_Video
from util.Function import Clear_File, Change_image_Size, Sort_Key, Write_Json
from util.WavJoin import Add_Wav_Processor

import json
import os
import shutil
import yaml
import wave
import logging

logger = logging.getLogger("pipeline")

#####################################################################################
#                              需要线程池完成的任务功能                             #
#####################################################################################

#推理VITS跟Sadtalker
def VITS_Sadtalker_Inference(result_vits_user_path, result_sadtalker_user_path, save_user_path):
    try:
        print('开始VITS+Sadtalker推理流程...')

        DH = VITS_Sadtalker_Join(result_vits_user_path, result_sadtalker_user_path, save_user_path)
        sad_parames_yaml_path, vits_parames_yaml_path, _ = Get_Parmes(save_user_path)

        print('设置模型参数...')
        DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)

        ref_wav_path = os.path.join(save_user_path,'Ref_Wav.wav')
        with open(os.path.join(save_user_path,'Ref_text.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        ref_text = data['Text']
        print(f'参考音频路径: {ref_wav_path}')
        print(f'参考文本: {ref_text}')

        # VITS推理（即使有问题也要继续）
        try:
            print('开始VITS推理...')
            DH.Inference_VITS(ref_wav_path,ref_text)
            print('VITS推理完成！')
        except Exception as e:
            print(f'VITS推理出现异常: {e}')
            print('但继续执行SadTalker推理...')

        # 强制执行SadTalker推理
        try:
            imag_path = os.path.join(save_user_path,'Image.png')
            print(f'开始SadTalker推理，图像路径: {imag_path}')

            # 确保图像文件存在
            if not os.path.exists(imag_path):
                print(f'错误：图像文件不存在 {imag_path}')
                raise FileNotFoundError(f'Image file not found: {imag_path}')

            # 确保音频文件JSON存在
            audio_json_path = os.path.join(save_user_path, 'Audio_save_path.json')
            if not os.path.exists(audio_json_path):
                print(f'错误：音频文件JSON不存在 {audio_json_path}')
                raise FileNotFoundError(f'Audio JSON file not found: {audio_json_path}')

            print('执行SadTalker推理...')
            DH.Inference_SadTalker(imag_path)
            print('SadTalker推理完成！')

        except Exception as e:
            print(f'SadTalker推理出现异常: {e}')
            import traceback
            traceback.print_exc()
            raise

        # 标记任务完成
        print('设置任务状态为完成...')
        try:
            result = Task_State(save_user_path, 'Audio_Video_Inference', True)
            print(f'Task_State 返回结果: {result}')
            print('VITS+Sadtalker推理流程全部完成！')
        except Exception as e:
            print(f'Task_State 设置失败: {e}')
            import traceback
            traceback.print_exc()
            raise

    except Exception as e:
        print(f'VITS_Sadtalker_Inference 整体失败: {e}')
        import traceback
        traceback.print_exc()
        # 即使失败也要设置状态，避免前端无限等待
        Task_State(save_user_path, 'Audio_Video_Inference', False)
        raise


# 推理用户音频跟Sadtalker
def User_Wav_Sadtalker_Inference(result_sadtalker_user_path, save_user_path):
    ppt_audio_dir = os.path.join(save_user_path, "PPT_Audio")
    audio_json_save_path = os.path.join(save_user_path, "Audio_save_path.json")
    video_json_save_path =os.path.join(save_user_path,"Video_save_path.json")
    image = os.path.join(save_user_path, "Image.png")
    
    Sad = SadTalker_Model()
    sad_parames_yaml_path, _, _ = Get_Parmes(save_user_path)
    Sad.Initialize_Parames(save_user_path, sad_parames_yaml_path)
    Sad.Initialize_Models()
    Write_Json(ppt_audio_dir, audio_json_save_path)
    
    #读取Audio_save_path.json里面的键值对
    with open(audio_json_save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    #把键跟值分别保存list
    ip_keys = list(data.keys())
    ip_values = list(data.values())

    # 规范化音频路径，确保使用正确的路径分隔符
    ip_values = [os.path.normpath(path) for path in ip_values]

    dict = {}

    for i in range(len(ip_keys)):
        save_path = os.path.join(result_sadtalker_user_path,ip_keys[i])

        Sad.Perform_Inference(image, ip_values[i],save_path)
        dict[ip_keys[i]] = os.path.normpath(save_path + ".mp4")
            
    # 使用with语句打开文件，确保在写入完成后自动关闭文件
    with open(video_json_save_path, 'w',encoding='utf-8') as json_file:
        # 使用json.dump()函数将数据写入JSON文件
        json.dump(dict, json_file)
    
    Task_State(save_user_path, "Audio_Video_Inference", True)


#推理VITS跟Wav2Lip
def VITS_Wav2Lip_Inference(result_vits_user_path, result_wav2lip_user_path, save_user_path):
    VWJ = VITS_Wav2Lip_Join(result_vits_user_path, result_wav2lip_user_path, save_user_path)
    _, vits_parames_yaml_path, w2p_parames_yaml_path = Get_Parmes(save_user_path)
    
    VWJ.Set_Params_and_Model(w2p_parames_yaml_path, vits_parames_yaml_path)
    
    ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
    with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    ref_text = data["Text"]
    print(ref_text)
    
    VWJ.Inference_VITS(ref_wav_path,ref_text)
    
    video_path = os.path.join(save_user_path,"Video.mp4")
    VWJ.Inference_Wav2Lip(video_path)
    Task_State(save_user_path, "Audio_Video_Inference", True)
    
# 推理用户音频跟Wav2Lip
def User_Wav_Wav2Lip_Inference(user_result_wav2lip_path, save_user_path):
    ppt_audio_dir = os.path.join(save_user_path, "PPT_Audio")
    audio_json_save_path = os.path.join(save_user_path, "Audio_save_path.json")
    json_file_path =os.path.join(save_user_path,"Video_save_path.json")
    video_path = os.path.join(save_user_path, "Video.mp4")
    
    Write_Json(ppt_audio_dir, audio_json_save_path)
    WM = Wav2Lip_Model()
    
    WM.Initialize_Parames(save_user_path)
    WM.Initialize_Models()
    
    #修改FPS
    fps_video = os.path.join(WM.wav2lip_temp, "fps_video.mp4")
    print("Change Video Fps")
    WM.Change_Video_Fps(video_path, fps_video, 25.0)     
    
    path = os.path.join(save_user_path,"Audio_save_path.json")
    #读取Audio_save_path.json里面的键值对
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    #把键跟值分别保存list
    ip_keys = list(data.keys())
    audio_path_values = list(data.values())

    # 规范化音频路径，确保使用正确的路径分隔符
    audio_path_values = [os.path.normpath(path) for path in audio_path_values]

    dict = {}
    for i in range(len(ip_keys)):
        save_path = os.path.join(user_result_wav2lip_path,ip_keys[i] + ".mp4")

        shear_video = WM.Shear_Video(fps_video, audio_path_values[i])
        WM.Perform_Inference(shear_video, audio_path_values[i], save_path)

        dict[ip_keys[i]] = save_path

    # 使用with语句打开文件，确保在写入完成后自动关闭文件
    with open(json_file_path, 'w',encoding='utf-8') as json_file:
        # 使用json.dump()函数将数据写入JSON文件
        json.dump(dict, json_file)
    Task_State(save_user_path, "Audio_Video_Inference", True)
    
#推理VITS
def VITS_Multiple_Inference(result_vits_user_path, result_sadtalker_user_path, save_user_path):
    DH = VITS_Sadtalker_Join(result_vits_user_path, result_sadtalker_user_path, save_user_path)
    sad_parames_yaml_path, vits_parames_yaml_path, _ = Get_Parmes(save_user_path)
    
    DH.Set_Params_and_Model(sad_parames_yaml_path, vits_parames_yaml_path)
    
    ref_wav_path = os.path.join(save_user_path,"Ref_Wav.wav")
    with open(os.path.join(save_user_path,"Ref_text.json"), "r", encoding='utf-8') as f:
        data = json.load(f)
    ref_text = data["Text"]
    print(ref_wav_path)
    print(ref_text)
        
    DH.Inference_VITS(ref_wav_path,ref_text)
    Task_State(save_user_path, "VITS_Inference", True)
    
#生成ppt融合视频（全插入）
def Video_Merge(save_user_path):
    Remove_Video_Background(save_user_path)
    video_path = Video_Joint(save_user_path)
    print("[Video_Merge] 视频合成完成")
    Last_Video_Join_Audio(save_user_path, video_path)
    print("[Video_Merge] 音频合成完成")
    Task_State(save_user_path, "Video_Merge", True)
    print(f"[Video_Merge] 状态已切换为 True")


#生成ppt融合视频（可选择插入）
def Video_Merge_Select_Into(save_user_path):
    Remove_Video_Background(save_user_path)
    video_path = Video_Joint_Select(save_user_path)
    Last_Video_Join_Audio(save_user_path, video_path)
    
    Task_State(save_user_path, "Video_Merge", True)

#训练VITS
def Train_VITS(save_user_path, user, json_data):
    print(f"[Train_VITS] 用户: {user}, 路径: {save_user_path}")
    print(f"[Train_VITS] 标注数据: {json_data}")
    VT = VITS_Train(user)
    list_file_path, audio_data_path = VT.Create_Audio_Label(save_user_path,json_data)
    print(f"[Train_VITS] Train.list 路径: {list_file_path}")
    print(f"[Train_VITS] Audio_Data 路径: {audio_data_path}")
    Clear_File(os.path.join(save_user_path,"Weight"))
    Clear_File(os.path.join("VITS/logs",user))
    print("[Train_VITS] 开始格式化数据...")
    VT.Format_Data(list_file_path, audio_data_path)
    print("[Train_VITS] 数据格式化完成，开始训练SoVITS...")
    VT.Train_SoVITS(save_user_path,10,10)
    print("[Train_VITS] SoVITS训练完成，开始训练GPT...")
    VT.Train_GPT(save_user_path,18,18)
    print("[Train_VITS] GPT训练完成，设置任务状态为True...")
    Task_State(save_user_path, "VITS_Train", True)
    print("[Train_VITS] 训练流程全部完成！")

#####################################################################################
#                                一些功能                                            #
#####################################################################################


# 创建user_path文件夹
def Create_File(user_name):

    # 确保使用正确的路径分隔符
    user_result_vits_path = os.path.normpath(os.path.join('Result', 'VITS', user_name))
    user_result_sadtalker_path = os.path.normpath(os.path.join('Result', 'SadTalker', user_name))
    user_result_wav2lip_path = os.path.normpath(os.path.join('Result', 'Wav2Lip', user_name))
    user_data_save_path = os.path.normpath(os.path.join('Data', user_name))

    # 新增目录：用于保存PPT视频和最终融合视频
    user_result_ppt_path = os.path.normpath(os.path.join('Result', 'ppt', user_name))
    user_result_generation_path = os.path.normpath(os.path.join('Result', 'generation', user_name))

    print(f'创建VITS路径: {user_result_vits_path}')
    print(f'创建SadTalker路径: {user_result_sadtalker_path}')
    print(f'创建数据保存路径: {user_data_save_path}')
    print(f'创建PPT视频路径: {user_result_ppt_path}')
    print(f'创建最终融合视频路径: {user_result_generation_path}')

    if not os.path.exists(user_result_vits_path):
        os.makedirs(user_result_vits_path)
        print(f'VITS目录已创建: {user_result_vits_path}')

    if not os.path.exists(user_result_sadtalker_path):
        os.makedirs(user_result_sadtalker_path)
        print(f'SadTalker目录已创建: {user_result_sadtalker_path}')

    if not os.path.exists(user_result_wav2lip_path):
        os.makedirs(user_result_wav2lip_path)
        print(f'Wav2Lip目录已创建: {user_result_wav2lip_path}')

    if not os.path.exists(user_data_save_path):
        os.makedirs(user_data_save_path)
        print(f'数据目录已创建: {user_data_save_path}')

    if not os.path.exists(user_result_ppt_path):
        os.makedirs(user_result_ppt_path)
        print(f'PPT视频目录已创建: {user_result_ppt_path}')

    if not os.path.exists(user_result_generation_path):
        os.makedirs(user_result_generation_path)
        print(f'最终融合视频目录已创建: {user_result_generation_path}')

    # 确保State.json文件存在
    state_file_path = os.path.join(user_data_save_path, "State.json")
    print(f'Create_File: 检查状态文件路径: {state_file_path}')
    print(f'Create_File: 状态文件是否存在: {os.path.exists(state_file_path)}')

    if not os.path.exists(state_file_path):
        print(f'Create_File: 开始创建状态文件...')
        # 创建初始状态文件
        initial_state = {
            "Audio_Video_Inference": "False",
            "VITS_Inference": "False",
            "Video_Merge": "False",
            "VITS_Train": "False"
        }

        try:
            with open(state_file_path, "w", encoding='utf-8') as f:
                json.dump(initial_state, f, ensure_ascii=False)
            print(f'Create_File: 状态文件创建成功: {state_file_path}')

            # 验证文件是否真的创建了
            if os.path.exists(state_file_path):
                print(f'Create_File: 验证成功 - 状态文件存在')
                with open(state_file_path, "r", encoding='utf-8') as f:
                    content = f.read()
                print(f'Create_File: 状态文件内容: {content}')
            else:
                print(f'Create_File: 验证失败 - 状态文件仍然不存在!')

        except Exception as e:
            print(f'Create_File: 创建状态文件失败: {e}')
            raise
    else:
        print(f'Create_File: 状态文件已存在')

        # 如果状态文件已存在，检查并重置失败的任务状态
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            # 检查是否有失败的任务需要重置
            reset_needed = False
            if state_data.get("Audio_Video_Inference") == "Failed":
                state_data["Audio_Video_Inference"] = "False"
                reset_needed = True
                print("Create_File: 重置 Audio_Video_Inference 状态为 False")

            if state_data.get("VITS_Train") == "Failed":
                state_data["VITS_Train"] = "False"
                reset_needed = True
                print("Create_File: 重置 VITS_Train 状态为 False")

            if state_data.get("VITS_Inference") == "Failed":
                state_data["VITS_Inference"] = "False"
                reset_needed = True
                print("Create_File: 重置 VITS_Inference 状态为 False")

            if state_data.get("Video_Merge") == "Failed":
                state_data["Video_Merge"] = "False"
                reset_needed = True
                print("Create_File: 重置 Video_Merge 状态为 False")

            # 如果有重置，写回文件
            if reset_needed:
                with open(state_file_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, ensure_ascii=False)
                print("Create_File: 已重置失败的任务状态")

        except (json.JSONDecodeError, Exception) as e:
            print(f"Create_File: 读取状态文件失败: {e}")
            # 如果读取失败，重新创建默认状态文件
            try:
                initial_state = {
                    "Audio_Video_Inference": "False",
                    "VITS_Inference": "False",
                    "Video_Merge": "False",
                    "VITS_Train": "False"
                }
                with open(state_file_path, "w", encoding='utf-8') as f:
                    json.dump(initial_state, f, ensure_ascii=False)
                print("Create_File: 重新创建默认状态文件")
            except Exception as create_e:
                print(f"Create_File: 重新创建状态文件失败: {create_e}")

    return user_result_vits_path, user_result_sadtalker_path, user_result_wav2lip_path, user_data_save_path, user_result_ppt_path, user_result_generation_path


# 初始化文件夹
def Init_File(user_result_vits_path, user_result_sadtalker_path, user_result_wav2lip_path, user_data_save_path):
    Clear_File(os.path.join(user_data_save_path, "Audio_Data"))
    Clear_File(os.path.join(user_data_save_path, "PPT_Audio"))
    Clear_File(os.path.join(user_data_save_path, "output_frames"))
    Clear_File(os.path.join(user_data_save_path, "input_frames"))
    Clear_File(os.path.join(user_data_save_path, "wav2lip_temp"))
    Clear_File(os.path.join(user_data_save_path, "Mov_Video"))
    Clear_File(os.path.join(user_data_save_path, "PPT_Video"))
    
    Clear_File(user_result_vits_path)
    Clear_File(user_result_sadtalker_path)
    Clear_File(user_result_wav2lip_path)
    
    
# 设置跟拉取任务状态
def Task_State(user_data_save_path, task, methods=None):

    json_file_path = os.path.join(user_data_save_path, "State.json")
    print(f"Task_State: 操作状态文件 {json_file_path}")

    # 检查状态文件是否存在
    if not os.path.exists(json_file_path):
        print(f"Task_State: 错误 - 状态文件不存在: {json_file_path}")
        print(f"Task_State: 用户数据目录内容: {os.listdir(user_data_save_path) if os.path.exists(user_data_save_path) else '目录不存在'}")
        raise FileNotFoundError(f"状态文件不存在: {json_file_path}")

    # 读取目标文件（统一用 json.load，避免格式不一致）
    import logging
    logger = logging.getLogger("pipeline")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
        print(f"Task_State: 成功读取状态文件，当前任务 {task} 的状态: {target_data.get(str(task), '未找到')}")
        logger.info(f"[Task_State] 成功读取状态文件，当前任务 {task} 的状态: {target_data.get(str(task), '未找到')}")
    except Exception as e:
        print(f"Task_State: 读取状态文件失败: {e}")
        logger.error(f"[Task_State] 读取状态文件失败: {e}")
        raise
        logger = logging.getLogger("pipeline")
        logger.info(f"[Task_State] 操作状态文件: {json_file_path}")

        # 检查状态文件是否存在
        if not os.path.exists(json_file_path):
            logger.error(f"[Task_State] 错误 - 状态文件不存在: {json_file_path}")
            logger.error(f"[Task_State] 用户数据目录内容: {os.listdir(user_data_save_path) if os.path.exists(user_data_save_path) else '目录不存在'}")
            raise FileNotFoundError(f"状态文件不存在: {json_file_path}")

        # 读取目标文件
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                target_data = yaml.safe_load(f)
            logger.info(f"[Task_State] 成功读取状态文件，当前任务 {task} 的状态: {target_data.get(str(task), '未找到')}")
        except Exception as e:
            logger.error(f"[Task_State] 读取状态文件失败: {e}")
            raise
    if methods != None:
        # 确保状态值始终是字符串类型，兼容前端逻辑
        if methods is True:
            target_data[str(task)] = "True"
            print(f"Task_State: 设置任务 {task} 状态为 True")
        elif methods is False:
            target_data[str(task)] = "Failed"
            print(f"Task_State: 设置任务 {task} 状态为 Failed")
        else:
            target_data[str(task)] = str(methods)
            print(f"Task_State: 设置任务 {task} 状态为 {methods}")

        print(f"Task_State: 写入状态文件前的数据: {target_data}")
        logger.info(f"[Task_State] 写入状态文件前的数据: {target_data}")
        try:
            with open(json_file_path, "w", encoding='utf-8') as json_file:
                json.dump(target_data, json_file, ensure_ascii=False)
            print(f"Task_State: 状态文件写入成功")
            logger.info(f"[Task_State] 状态文件写入成功，已写入: {target_data}")

            # 验证写入结果
            with open(json_file_path, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
            print(f"Task_State: 验证写入结果，任务 {task} 状态: {verify_data.get(str(task), '未找到')}")
            logger.info(f"[Task_State] 验证写入结果，任务 {task} 状态: {verify_data.get(str(task), '未找到')}，完整内容: {verify_data}")

        except Exception as e:
            print(f"Task_State: 写入状态文件失败: {e}")
            logger.error(f"[Task_State] 状态文件写入失败: {e}")
            raise

    return target_data[str(task)]
    
# 配置SadTalker参数
def Config_SadTalker_Parmes(user_data_save_path, json_dict):
    target_file_path = os.path.join(user_data_save_path, "SadTalker_config.yaml")
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
            shutil.copy("SadTalker/SadTalker_config.yaml", user_data_save_path)

    # 读取目标文件
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)

    # 替换值（带规范化）
    def _norm_enhancer(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return 'gfpgan' if v else None
        s = str(v).strip()
        if s == '':
            return None
        sl = s.lower()
        if sl in ('true','1','yes','y','gfpgan'):
            return 'gfpgan'
        if sl in ('false','0','no','n','none','null'):
            return None
        if sl == 'restoreformer':
            return 'RestoreFormer'
        if sl == 'codeformer':
            return 'codeformer'
        return s
    def _norm_bg(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return 'realesrgan' if v else None
        s = str(v).strip().lower()
        if s in ('true','1','yes','y','realesrgan'):
            return 'realesrgan'
        if s in ('false','0','no','n','none','null',''):
            return None
        return None

    for key, value in json_dict.items():
        if key in target_data:
            if key == 'enhancer':
                target_data[key] = _norm_enhancer(value)
            elif key == 'background_enhancer':
                target_data[key] = _norm_bg(value)
            else:
                target_data[key] = value

    # 写入目标文件
    with open(target_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_data, f)

    return target_file_path

# 配置VITS参数
def Config_VITS_Parmes(user_data_save_path, json_dict):
    target_file_path = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    
    # 读取目标文件
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)

    # 替换值
    for key, value in json_dict.items():
        if key in target_data:
            target_data[key] = value

    # 写入目标文件
    with open(target_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_data, f)
    
    return target_file_path

# 配置wav2lip参数
def Config_Wav2Lip_Parmes(user_data_save_path, json_dict):
    target_file_path = os.path.join(user_data_save_path, "Wav2Lip_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
        shutil.copy("Easy_Wav2Lip/Wav2Lip_config.yaml", user_data_save_path)
        
    # 读取目标文件
    with open(target_file_path, 'r', encoding='utf-8') as f:
        target_data = yaml.safe_load(f)
    # 替换值
    for key, value in json_dict.items():
        if key in target_data:
            target_data[key] = value

    # 写入目标文件
    with open(target_file_path, 'w', encoding='utf-8') as f:
        yaml.dump(target_data, f)
        
    
    return target_file_path

# 获取参数
def Get_Parmes(user_data_save_path):
    vits = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    sadtdlker = os.path.join(user_data_save_path, "SadTalker_config.yaml")
    wav2lip = os.path.join(user_data_save_path,"Wav2Lip_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(vits):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    if not os.path.exists(sadtdlker):
        shutil.copy("SadTalker/SadTalker_config.yaml", user_data_save_path)
    if not os.path.exists(wav2lip):
        shutil.copy("Easy_Wav2Lip/Wav2Lip_config.yaml", user_data_save_path)

    
    return sadtdlker, vits, wav2lip
    
# 保存PPT备注
def Save_PPT_Remake(user_data_save_path, ppt_remakes):
    ppt_remake_filename = os.path.join(user_data_save_path, "PPT_Remake.json")
    
    with open(ppt_remake_filename, "w", encoding='utf-8') as json_file:
        json.dump(ppt_remakes, json_file, ensure_ascii=False)
         
    return ppt_remake_filename
 
# 保存数字人插入页数的json
def Save_People_Location(user_data_save_path, people_location):
    people_location_filename = os.path.join(user_data_save_path, "People_Location.json")
    
    with open(people_location_filename, "w", encoding='utf-8') as json_file:
        json.dump(people_location, json_file, ensure_ascii=False)
         
    return people_location_filename

# 保存真人照片
def Save_Image(user_data_save_path, image_path):
    img_name = os.path.join(user_data_save_path, "Image.png")
    if os.path.exists(img_name):
        os.remove(img_name)

    #移动
    shutil.move(image_path, user_data_save_path)
    #修改保存在save_user_path照片的名字
    # 使用os.path.basename替代硬编码的路径分隔符
    original_filename = os.path.basename(image_path)
    os.rename(os.path.join(user_data_save_path, original_filename), img_name)
    Change_image_Size(img_name)

# 保存接收的视频
def Save_Video(user_data_save_path, video_at_path):
    # 从user_data_save_path中提取用户名
    user_name = os.path.basename(user_data_save_path)
    user_result_ppt_path = os.path.normpath(os.path.join('Result', 'ppt', user_name))

    video_name = os.path.join(user_result_ppt_path, "PPT_Video.mp4")

    if os.path.exists(video_name):
        os.remove(video_name)
    #移动
    shutil.move(video_at_path, user_result_ppt_path)
    # 使用os.path.basename替代硬编码的路径分隔符
    original_filename = os.path.basename(video_at_path)
    os.rename(os.path.join(user_result_ppt_path, original_filename), video_name)

# 保存音频时长
def Save_Tiem(user_data_save_path, audio_dir):
    wav_tiem = os.path.join(user_data_save_path,"Time.json")
    time_dict = {}

    # 统一处理：audio_dir应该是绝对路径，直接使用
    audio_dir_all_file = audio_dir

    print(f"Save_Tiem: 使用音频目录: {audio_dir_all_file}")
    print(f"Save_Tiem: 音频目录是否存在: {os.path.exists(audio_dir_all_file)}")

    # 检查目录是否存在
    if not os.path.exists(audio_dir_all_file):
        print(f"Save_Tiem: 警告 - 音频目录不存在: {audio_dir_all_file}")
        return None

    file_dir = os.listdir(audio_dir_all_file)
    file_list = sorted(file_dir, key=Sort_Key)

    if (len(file_list) > 0):
        #获取音频文件时长
        total_duration = 0.0
        for _, file in enumerate(file_list):
            # 只处理.wav文件
            if not file.lower().endswith('.wav'):
                continue

            #获取文件名（去掉扩展名）
            file_name = file.split(".")[0]
            file_path = os.path.join(audio_dir_all_file, file)

            try:
                with wave.open(file_path, 'rb') as wav_file:
                    # 获取帧数
                    frames = wav_file.getnframes()
                    # 获取帧速率（每秒的帧数）
                    frame_rate = wav_file.getframerate()
                    # 计算时长（以秒为单位）
                    duration = frames / float(frame_rate)
                time_dict[str(file_name)] = duration
                total_duration += duration
                print(f"Save_Tiem: {file_name} 时长 {duration:.2f}秒")
            except Exception as e:
                print(f"Save_Tiem: 读取音频文件 {file_path} 失败: {e}")
                continue

        with open(wav_tiem, 'w', encoding='utf-8') as f:
            json.dump(time_dict, f, ensure_ascii=False, indent=2)

        print(f"Save_Tiem: 已保存 {len(time_dict)} 个音频文件时长，总时长: {total_duration:.2f}秒")
        return time_dict
    else:
        print(f"Save_Tiem: 音频目录为空: {audio_dir_all_file}")
        return None

# 保存用于训练的音频
def Save_Train_Audio(user_data_save_path, name, audio_data):
    audio_data_path = os.path.join(user_data_save_path, "Audio_Data")
    
    audio = os.path.join(audio_data_path, name)
    if os.path.exists(audio):
        os.remove(audio)
        
    #保存到文件
    with open(audio, "wb") as img_file:
        img_file.write(audio_data)
        
# 保存插入ppt的音频
def Save_Insert_Audio(user_data_save_path, name, audio_data):
    insert_audio_path = os.path.join(user_data_save_path, "PPT_Audio")
    
    audio = os.path.join(insert_audio_path, name)
    if os.path.exists(audio):
        os.remove(audio)
        
    #保存到文件
    with open(audio, "wb") as img_file:
        img_file.write(audio_data)
        
#####################################################################################
#                                VITS功能                                           #
#####################################################################################

#训练VITS模型
class VITS_Train():
    def __init__(self, user):
        self.GST = GPT_SoVITS_Tarin(user)
        
    #对音频标注保存
    def Create_Audio_Label(self, user_data_save_path, data_json):
        audio_data_path = os.path.join(user_data_save_path, "Audio_Data")
        
        list_file_path = os.path.join(user_data_save_path, "Train.list")
        with open(list_file_path, "w", encoding='utf-8') as text_file:
            for key, value in data_json.items():
                name = os.path.join(audio_data_path,key)
                text_file.write(f"{name}|split|ZH|{value}\n")
        return list_file_path, audio_data_path

    #数据格式化
    def Format_Data(self, train_list, train_audio_path):
        if(self.GST.Format_Data(train_list,train_audio_path)):
            return True
        return False


    #训练VITS模型
    def Train_SoVITS(self, user_data_save_path, total_epoch, save_every_epoch):
        model_path = self.GST.Train_SoVITS(total_epoch, save_every_epoch)
        
        if(model_path != None):
            path = os.path.join(user_data_save_path, "Weight")
            shutil.move(model_path, path)
            return True
        return False

    #训练GPT模型
    def Train_GPT(self, user_data_save_path, total_epoch, save_every_epoch):

        model_path = self.GST.Train_GPT(total_epoch, save_every_epoch)
        
        if(model_path != None):
            path = os.path.join(user_data_save_path, "Weight")
            shutil.move(model_path, path)
            return True
        return False


#保存VITS的参照音频跟文字
def Save_VITS_Ref_Wav_And_Text(user_data_save_path, wav_path, ref_text, methods="move"):
    wav_name = os.path.join(user_data_save_path, "Ref_Wav.wav")
    if os.path.exists(wav_name):
        os.remove(wav_name)
    #移动或者copy
    if(methods == "copy"):
        shutil.copy(wav_path, user_data_save_path)
    elif(methods == "move"):
        shutil.move(wav_path, user_data_save_path)

    # 使用os.path.basename替代硬编码的路径分隔符
    original_filename = os.path.basename(wav_path)
    os.rename(os.path.join(user_data_save_path, original_filename), wav_name)
    
    ref_wav_text = os.path.join(user_data_save_path,"Ref_text.json")
    
    with open(ref_wav_text, "w", encoding='utf-8') as json_file:
        json.dump(ref_text, json_file, ensure_ascii=False)
        
    return ref_wav_text


#//////////////////////////////// VITS模型二选一(改模型路径) /////////////////////////////////////////////////////////////////////////////////////////////
 
# 选择自定义的VITS模型
def Select_Train_VITS_Model(user_data_save_path, user):
    pth = os.path.join(user_data_save_path, "Weight", f"{user}.pth")
    ckpt = os.path.join(user_data_save_path, "Weight", f"{user}.ckpt")

    model_dict = {
        "GPT_model_path" : ckpt,
        "SoVITS_model_path" : pth
    }
    
    Config_VITS_Parmes(user_data_save_path, model_dict)
 
    
    
# 选择预训练的VITS模型 （把ref_wav和ref_test复制到保存路径，然后再改掉GPT-SoVITS_config.yaml模型位置，改掉参考文字）
def Select_VITS_Model(user_data_save_path, index):
    target_file_path = os.path.join(user_data_save_path, "GPT-SoVITS_config.yaml")
    
    # 检查文件是否存在
    if not os.path.exists(target_file_path):
        shutil.copy("VITS/GPT-SoVITS_config.yaml", user_data_save_path)
    
    with open("VITS/Model.json",'r', encoding='utf-8') as f:
        model_json = json.load(f)
    
    model = model_json[str(index)]
    Config_VITS_Parmes(user_data_save_path, model)
 
        
    Save_VITS_Ref_Wav_And_Text(user_data_save_path, model["Ref_Wav"], {"Text" : model["Ref_Text"]}, "copy")
        
    return target_file_path
    
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#####################################################################################
#                                 视频拼接                                          #
#####################################################################################
     
#把sadtalker的每一个视频去背景
def Remove_Video_Background(user_data_save_path):
    output_frames = os.path.join(user_data_save_path, "output_frames")
    input_frames = os.path.join(user_data_save_path, "input_frames")
    mov_video = os.path.join(user_data_save_path, "Mov_Video")

    print(f"Remove_Video_Background: 开始处理，用户路径: {user_data_save_path}")
    print(f"Remove_Video_Background: 输出帧目录: {output_frames}")
    print(f"Remove_Video_Background: 输入帧目录: {input_frames}")
    print(f"Remove_Video_Background: Mov视频目录: {mov_video}")

    Clear_File(output_frames)
    Clear_File(input_frames)
    Clear_File(mov_video)

    PV = Ppt_2_Video(output_frames, input_frames, mov_video)

    video_json_path = os.path.join(user_data_save_path, "Video_save_path.json")
    print(f"Remove_Video_Background: 读取视频路径JSON: {video_json_path}")
    print(f"Remove_Video_Background: JSON文件是否存在: {os.path.exists(video_json_path)}")

    with open(video_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Remove_Video_Background: 读取到 {len(data)} 个SadTalker视频路径")
    print(f"Remove_Video_Background: 视频路径数据: {data}")

    for key, video_file_path in enumerate(data.values()):
        print(f"Remove_Video_Background: 处理第 {key} 个视频，路径: {video_file_path}")
        print(f"Remove_Video_Background: 视频文件是否存在: {os.path.exists(video_file_path)}")

        if not os.path.exists(video_file_path):
            print(f"Remove_Video_Background: 警告 - 视频文件不存在: {video_file_path}")
            continue

        try:
            print(f"Remove_Video_Background: 开始提取视频帧: {video_file_path}")
            PV.Video_To_Frames(video_file_path)
            print(f"Remove_Video_Background: 帧提取完成")

            print(f"Remove_Video_Background: 开始去除背景")
            PV.Remove_Background()
            print(f"Remove_Video_Background: 背景去除完成")

            print(f"Remove_Video_Background: 开始创建Mov视频，key: {key}")
            PV.Create_Video(key)

            mov_file_path = os.path.join(mov_video, f"{key}_Mov.mov")
            print(f"Remove_Video_Background: Mov视频创建完成: {mov_file_path}")
            print(f"Remove_Video_Background: Mov文件是否存在: {os.path.exists(mov_file_path)}")

        except Exception as e:
            print(f"Remove_Video_Background: 处理视频 {video_file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Remove_Video_Background: 所有SadTalker视频处理完成")
    
# 工具函数：获取视频时长（秒），优先 ffprobe，失败则回退 OpenCV
def _get_video_duration_sec(p: str) -> float:
    try:
        import subprocess
        proc = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", p
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        out = (proc.stdout or b"").decode().strip()
        return float(out) if out else 0.0
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return float(frames) / float(fps) if fps else 0.0
    except Exception:
        return 0.0

#Mov视频跟PPT合成最终视频(全插入)
def Video_Joint(user_data_save_path):

    output_frames = os.path.join(user_data_save_path, "output_frames")
    input_frames = os.path.join(user_data_save_path, "input_frames")
    Mov_Video = os.path.join(user_data_save_path, "Mov_Video")

    # 修复：使用正确的PPT视频路径
    user_name = os.path.basename(user_data_save_path)
    user_result_ppt_path = os.path.normpath(os.path.join('Result', 'ppt', user_name))
    # 首选 Result/ppt/<user>/PPT_Video.mp4
    ppt_video_path = os.path.join(user_result_ppt_path, "PPT_Video.mp4")
    # 备选 Data/<user>/PPT_Base_Video.mp4 或 Data/<user>/PPT_Video.mp4
    alt_base = os.path.join(user_data_save_path, "PPT_Base_Video.mp4")
    alt_ppt = os.path.join(user_data_save_path, "PPT_Video.mp4")
    output_ppt_video = os.path.join(user_data_save_path, "PPT_Video")

    print(f'Video_Joint: 使用PPT视频路径: {ppt_video_path}')
    exists_primary = os.path.exists(ppt_video_path)
    print(f'Video_Joint: PPT视频是否存在: {exists_primary}')
    if not exists_primary:
        if os.path.exists(alt_base):
            print(f"Video_Joint: 回退使用基础PPT视频: {alt_base}")
            ppt_video_path = alt_base
        elif os.path.exists(alt_ppt):
            print(f"Video_Joint: 回退使用用户目录下PPT视频: {alt_ppt}")
            ppt_video_path = alt_ppt
        else:
            # 最后回退：创建一段空白视频，时长为Time.json的总和
            try:
                from moviepy.editor import ColorClip
                with open(os.path.join(user_data_save_path, "Time.json"), 'r', encoding='utf-8') as f:
                    td = json.load(f)
                total_duration = sum(float(v) for v in td.values()) if isinstance(td, dict) else 5.0
                tmp_blank = os.path.join(user_data_save_path, "__blank_ppt_video.mp4")
                ColorClip(size=(1920,1080), color=(255,255,255), duration=max(total_duration, 1.0)).write_videofile(tmp_blank, fps=24, codec='libx264', audio=False)
                ppt_video_path = tmp_blank
                print(f"Video_Joint: 未找到PPT视频，已创建空白视频: {ppt_video_path}")
            except Exception as e:
                print(f"Video_Joint: 创建空白视频失败: {e}")

    
    Clear_File(output_ppt_video)
    
    print(f"Video_Joint: 初始化插入器 -> output_frames={output_frames}, input_frames={input_frames}, mov_dir={Mov_Video}")
    PV = Ppt_2_Video(output_frames, input_frames, Mov_Video)
    transition_time = 0
    
    with open(os.path.join(user_data_save_path, "Time.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # keys = list(data.keys())
    time_list = list(data.values())
    print(f"Video_Joint: Time列表长度={len(time_list)}, 值={time_list}")
        
    last_path = None
    for i in range(len(time_list)):
        # mov_path = os.path.join(Mov_Video, mov_list[i])
        mov_path = os.path.join(Mov_Video,  f"{i}" + "_Mov.mov")
        print(f"Video_Joint: 第{i}段 -> mov={mov_path}, exists={os.path.exists(mov_path)}")
        
        if i == 0:
            one_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            # transition_time += 3
            base_dur = _get_video_duration_sec(ppt_video_path)
            adj_ins = min(transition_time, max(0.0, base_dur - float(time_list[i])))
            print(f"Video_Joint: 插入第{i}段 -> base={ppt_video_path}, out={one_ppt_video_path}, ins={adj_ins} (原ins={transition_time}, base_dur={base_dur}), dur={time_list[i]}")
            PV.Insert_Video(ppt_video_path, mov_path, one_ppt_video_path, adj_ins, time_list[i])
            print(f"Video_Joint: 第{i}段完成 -> out_exists={os.path.exists(one_ppt_video_path)} size={(os.path.getsize(one_ppt_video_path) if os.path.exists(one_ppt_video_path) else -1)}")
            last_path = one_ppt_video_path
        else:
            one_ppt_video_path = last_path if last_path else ppt_video_path
            two_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            # transition_time += 4
            transition_time += time_list[i - 1]
            base_dur = _get_video_duration_sec(one_ppt_video_path)
            adj_ins = min(transition_time, max(0.0, base_dur - float(time_list[i])))
            print(f"Video_Joint: 插入第{i}段 -> base={one_ppt_video_path}, out={two_ppt_video_path}, ins={adj_ins} (原ins={transition_time}, base_dur={base_dur}), dur={time_list[i]}")
            PV.Insert_Video(one_ppt_video_path, mov_path, two_ppt_video_path, adj_ins, time_list[i])
            print(f"Video_Joint: 第{i}段完成 -> out_exists={os.path.exists(two_ppt_video_path)} size={(os.path.getsize(two_ppt_video_path) if os.path.exists(two_ppt_video_path) else -1)}")
            last_path = two_ppt_video_path
            
    return last_path if last_path else ppt_video_path
    
#Mov视频跟PPT合成最终视频（可选择插入）
def Video_Joint_Select(user_data_save_path):
    output_frames = os.path.join(user_data_save_path, "output_frames")
    input_frames = os.path.join(user_data_save_path, "input_frames")
    Mov_Video = os.path.join(user_data_save_path, "Mov_Video")

    # 修复：使用正确的PPT视频路径
    user_name = os.path.basename(user_data_save_path)
    user_result_ppt_path = os.path.normpath(os.path.join('Result', 'ppt', user_name))
    ppt_video_path = os.path.join(user_result_ppt_path, "PPT_Video.mp4")
    alt_base = os.path.join(user_data_save_path, "PPT_Base_Video.mp4")
    alt_ppt = os.path.join(user_data_save_path, "PPT_Video.mp4")
    output_ppt_video = os.path.join(user_data_save_path, "PPT_Video")

    print(f'Video_Joint_Select: 使用PPT视频路径: {ppt_video_path}')
    exists_primary = os.path.exists(ppt_video_path)
    print(f'Video_Joint_Select: PPT视频是否存在: {exists_primary}')
    if not exists_primary:
        if os.path.exists(alt_base):
            print(f"Video_Joint_Select: 回退使用基础PPT视频: {alt_base}")
            ppt_video_path = alt_base
        elif os.path.exists(alt_ppt):
            print(f"Video_Joint_Select: 回退使用用户目录下PPT视频: {alt_ppt}")
            ppt_video_path = alt_ppt
        else:
            try:
                from moviepy.editor import ColorClip
                with open(os.path.join(user_data_save_path, "Time.json"), 'r', encoding='utf-8') as f:
                    td = json.load(f)
                total_duration = sum(float(v) for v in td.values()) if isinstance(td, dict) else 5.0
                tmp_blank = os.path.join(user_data_save_path, "__blank_ppt_video.mp4")
                ColorClip(size=(1920,1080), color=(255,255,255), duration=max(total_duration, 1.0)).write_videofile(tmp_blank, fps=24, codec='libx264', audio=False)
                ppt_video_path = tmp_blank
                print(f"Video_Joint_Select: 未找到PPT视频，已创建空白视频: {ppt_video_path}")
            except Exception as e:
                print(f"Video_Joint_Select: 创建空白视频失败: {e}")
    
    last_video_path = ""
    
    Clear_File(output_ppt_video)
    
    print(f"Video_Joint_Select: 初始化插入器 -> output_frames={output_frames}, input_frames={input_frames}, mov_dir={Mov_Video}")
    PV = Ppt_2_Video(output_frames, input_frames, Mov_Video)
    transition_time = 0
    OnePPTVideoIndex = -1
    with open(os.path.join(user_data_save_path, "Time.json"), 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(os.path.join(user_data_save_path, "People_Location.json"), 'r', encoding='utf-8') as f:
        select = json.load(f)
    # keys = list(data.keys())
    time_list = list(data.values())
    print(f"Video_Joint_Select: Time列表长度={len(time_list)}, 值={time_list}")
        
    for i in range(len(time_list)):
        # mov_path = os.path.join(Mov_Video, mov_list[i])
        mov_path = os.path.join(Mov_Video,  f"{i}" + "_Mov.mov")
        print(f"Video_Joint_Select: 第{i}段 -> mov={mov_path}, exists={os.path.exists(mov_path)}")

        if i == 0:
            one_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            #transition_time += 3
            if(select[str(i)] == "True"):
                OnePPTVideoIndex = i
                base_dur = _get_video_duration_sec(ppt_video_path)
                adj_ins = min(transition_time, max(0.0, base_dur - float(time_list[i])))
                print(f"Video_Joint_Select: 插入第{i}段(选中) -> base={ppt_video_path}, out={one_ppt_video_path}, ins={adj_ins} (原ins={transition_time}, base_dur={base_dur}), dur={time_list[i]}")
                PV.Insert_Video(ppt_video_path, mov_path, one_ppt_video_path, adj_ins, time_list[i])
                print(f"Video_Joint_Select: 第{i}段完成 -> out_exists={os.path.exists(one_ppt_video_path)} size={(os.path.getsize(one_ppt_video_path) if os.path.exists(one_ppt_video_path) else -1)}")
                last_video_path = one_ppt_video_path
        else:
            if(OnePPTVideoIndex == -1):
                one_ppt_video_path = ppt_video_path
            else:
                one_ppt_video_path = os.path.join(output_ppt_video, f"{OnePPTVideoIndex}_ppt_video.mp4")
            two_ppt_video_path = os.path.join(output_ppt_video, f"{i}_ppt_video.mp4")
            # transition_time += 4
            transition_time += time_list[i - 1]
            if(select[str(i)] == "True"):
                OnePPTVideoIndex = i
                base_dur = _get_video_duration_sec(one_ppt_video_path)
                adj_ins = min(transition_time, max(0.0, base_dur - float(time_list[i])))
                print(f"Video_Joint_Select: 插入第{i}段(选中) -> base={one_ppt_video_path}, out={two_ppt_video_path}, ins={adj_ins} (原ins={transition_time}, base_dur={base_dur}), dur={time_list[i]}")
                PV.Insert_Video(one_ppt_video_path, mov_path, two_ppt_video_path, adj_ins, time_list[i])
                print(f"Video_Joint_Select: 第{i}段完成 -> out_exists={os.path.exists(two_ppt_video_path)} size={(os.path.getsize(two_ppt_video_path) if os.path.exists(two_ppt_video_path) else -1)}")
                last_video_path = two_ppt_video_path
            
    return last_video_path

#最终的PPT视频拼接音频
def Last_Video_Join_Audio(user_data_save_path, input_video):
    AWP = Add_Wav_Processor()
    print(f"Last_Video_Join_Audio: PPT视频拼接音频: {input_video}")
    # 从user_data_save_path中提取用户名
    user_name = os.path.basename(user_data_save_path)
    user_result_generation_path = os.path.normpath(os.path.join('Result', 'generation', user_name))

    audio_save_path = os.path.join(user_data_save_path, "Audio_save_path.json")
    last_video = os.path.join(user_result_generation_path, "last_video.mp4")
    join_audio = os.path.join(user_data_save_path, "Join_Audio.wav")

    with open(audio_save_path, 'r', encoding='utf-8') as f:
        path_dict = json.load(f)

    for i in path_dict.keys():
        if i == "Slide 1":
            
            print(f"Last_Video_Join_Audio: 处理第1段音频 {path_dict[str(i)]}")
            result_audio = AWP.Add_Silence_At_Beginning(path_dict[str(i)])
            print("Last_Video_Join_Audio: Add_Silence_At_Beginning 完成")
            result_audio.export(join_audio, format="wav")
            print(f"Last_Video_Join_Audio: 第1段音频导出完成 {join_audio}")
            print(f"Last_Video_Join_Audio: join_audio exists={os.path.exists(join_audio)}, size={(os.path.getsize(join_audio) if os.path.exists(join_audio) else -1)}")

        else:
            print(f"Last_Video_Join_Audio: 拼接音频 {path_dict[str(i)]}, {join_audio}")

            result_audio = AWP.Add_Silence_Between_Tracks(join_audio,path_dict[str(i)])
            print("Last_Video_Join_Audio: Add_Silence_Between_Tracks 完成")
            result_audio.export(join_audio, format="wav")
            print(f"Last_Video_Join_Audio: 第{i}段音频导出完成 {join_audio}")
            print(f"Last_Video_Join_Audio: join_audio exists={os.path.exists(join_audio)}, size={(os.path.getsize(join_audio) if os.path.exists(join_audio) else -1)}")
        

    print(f"Last_Video_Join_Audio: 合成音频文件: {join_audio}")
    try:
       
        AWP.Add_Audio_To_Video(input_video, join_audio, last_video)
        print(f"Last_Video_Join_Audio: 合成完成，输出: {last_video}")
        print(f"Last_Video_Join_Audio: last_video exists={os.path.exists(last_video)}, size={(os.path.getsize(last_video) if os.path.exists(last_video) else -1)}")
    except Exception as e:
        print(f"Last_Video_Join_Audio: 合成音频到视频失败: {e}")
        import traceback
        traceback.print_exc()
    return last_video
  
#PPT视频拼接音频
def Video_Join_Audio(user_data_save_path):
    AWP = Add_Wav_Processor()
    input_video = os.path.join(user_data_save_path, "PPT_Video.mp4")
    audio_save_path = os.path.join(user_data_save_path, "Audio_save_path.json")
    # 从user_data_save_path中提取用户名，使用generation路径保存最终视频
    user_name = os.path.basename(user_data_save_path)
    user_result_generation_path = os.path.normpath(os.path.join('Result', 'generation', user_name))
    last_video = os.path.join(user_result_generation_path, "last_video.mp4")
    join_audio = os.path.join(user_data_save_path, "Join_Audio.wav")
    
    with open(audio_save_path, 'r', encoding='utf-8') as f:
        path_dict = json.load(f)
        
    for i in path_dict.keys():
        print(f"[JoinAudio] key={i}, audio_path={path_dict[str(i)]}, join_audio={join_audio}")
        print(f"[JoinAudio] 导出前 exists={os.path.exists(join_audio)}, size={(os.path.getsize(join_audio) if os.path.exists(join_audio) else -1)}")
        try:
            if i == "0":
                result_audio = AWP.Add_Silence_At_Beginning(path_dict[str(i)])
                result_audio.export(join_audio, format="wav")
            else:
                result_audio = AWP.Add_Silence_Between_Tracks(join_audio,path_dict[str(i)])
                result_audio.export(join_audio, format="wav")
            print(f"[JoinAudio] 导出后 exists={os.path.exists(join_audio)}, size={(os.path.getsize(join_audio) if os.path.exists(join_audio) else -1)}")
        except Exception as e:
            print(f"[JoinAudio] 导出异常: {e}")
            import traceback; traceback.print_exc()
        print(f"[JoinAudio] 完成 key={i}")

    #  当不是第一个视频的时候就执行add_silence_between_tracks和上一个wav结合空出几秒自己选择
    #  生成完成add_audio_to_video贴入MP4
    AWP.Add_Audio_To_Video(input_video, join_audio, last_video)
    
    return last_video
  
#####################################################################################
#                                模型推理                                            #
#####################################################################################
  
#VITS跟Sadtalker结合
class VITS_Sadtalker_Join():
    def __init__(self,vits,sadtalker,save):
        
        #声明模型
        self.VITS = None
        self.Sad = None
        
        
        #实例化
        self.VITS = GPT_SoVITS_Model()
        self.Sad = SadTalker_Model()
        
        #全局变量
        self.user_result_vits_path = vits
        self.user_result_sadtalker_path = sadtalker
        self.user_data_save_path = save
        
        
        #清空文件
        Clear_File(self.user_result_vits_path)
        Clear_File(self.user_result_sadtalker_path)
           
    #预加载参数和模型
    def Set_Params_and_Model(self,sad_parames_yaml_path=None,vits_parames_yaml_path=None):
        self.VITS.Initialize_Parames(vits_parames_yaml_path)
        
        self.Sad.Initialize_Parames(self.user_data_save_path, sad_parames_yaml_path)
        self.Sad.Initialize_Models()
        

    #根据备注推理音频
    def Inference_VITS(self, ref_wav_path, ref_text):
        path = os.path.join(self.user_data_save_path,"PPT_Remake.json")
        #读取Json_Data/PPT_Remake.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 日志：PPT_Remake 类型与概览
        try:
            if isinstance(data, dict):
                logger.info(f"PPT_Remake.json 加载成功，类型=dict，键数={len(data)}，路径={path}")
            else:
                logger.warning(f"PPT_Remake.json 类型异常：type={type(data).__name__}，路径={path}")
        except Exception:
            pass

        # 将键按其中出现的数字进行自然排序（如 Slide 1, Slide 2, ...）以匹配 PPT 实际页序
        def _key_to_index(k):
            try:
                import re
                m = re.search(r"(\d+)", str(k))
                return int(m.group(1)) if m else float('inf')
            except Exception:
                return float('inf')
        ordered_keys = sorted(list(data.keys()), key=_key_to_index)
        ip_keys = ordered_keys
        PPT_Remake_values = [data[k] for k in ordered_keys]

        dict = {}

        print(f"VITS推理: 输出路径 {self.user_result_vits_path}")
        print(f"VITS推理: 待处理音频数量 {len(ip_keys)}")
        logger.info(f"VITS 音频输出目录: {os.path.abspath(self.user_result_vits_path)}，数量={len(ip_keys)}")

        for i in range(len(ip_keys)):

            text = PPT_Remake_values[i]
            output_path = os.path.normpath(os.path.join(self.user_result_vits_path, f"{ip_keys[i]}.wav"))

            print(f"VITS推理: 处理第{i+1}个音频，键: {ip_keys[i]}, 输出路径: {output_path}")
            logger.info(f"VITS 生成音频 -> key={ip_keys[i]} path={output_path}")

            # 确保输出目录存在
            os.makedirs(self.user_result_vits_path, exist_ok=True)

            try:
                self.VITS.Initialize_Models()
                self.VITS.Perform_Inference(
                    ref_wav_path = ref_wav_path,
                    prompt_text = ref_text,
                    prompt_languageself = self.VITS.i18n("中文"),
                    target_text = text,
                    target_text_language = self.VITS.i18n("中文"),
                    cut = self.VITS.i18n("凑50字一切"),
                    output_path=output_path
                )

                # 验证音频文件是否成功生成
                if os.path.exists(output_path):
                    print(f"VITS推理: 音频文件生成成功 {output_path}")
                    logger.info(f"VITS 生成成功: {output_path}")
                    dict[ip_keys[i]] = output_path
                else:
                    print(f"VITS推理: 警告 - 音频文件未生成 {output_path}")
                    logger.warning(f"VITS 未生成文件: {output_path}")
                    dict[ip_keys[i]] = output_path  # 仍然记录路径，让后续流程继续

            except Exception as e:
                print(f"VITS推理: 处理音频 {ip_keys[i]} 时出错: {e}")
                logger.exception(f"VITS 生成异常 key={ip_keys[i]} path={output_path} error={e}")
                # 即使出错也记录路径，让后续流程可以继续
                dict[ip_keys[i]] = output_path

        # 规范化所有路径，确保使用正确的路径分隔符
        normalized_dict = {}
        for key, path in dict.items():
            normalized_dict[key] = os.path.normpath(path)

        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Audio_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(normalized_dict, json_file)

        print(f"VITS推理: 已保存音频路径到 {json_file_path}")
        print(f"VITS推理: 音频路径数据 {normalized_dict}")
        logger.info(f"已写入 Audio_save_path.json -> {json_file_path}，keys={list(normalized_dict.keys())}")

        Save_Tiem(self.user_data_save_path, self.user_result_vits_path)

    #生成数字人视频
    def Inference_SadTalker(self,image):
        path = os.path.join(self.user_data_save_path,"Audio_save_path.json")
        #读取Audio_save_path.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            logger.info(f"读取音频路径清单: {path}，数量={len(data) if isinstance(data, dict) else 'N/A'}")
        except Exception:
            pass

        #把键跟值分别保存list
        ip_keys = list(data.keys())
        audio_path_values = list(data.values())

        # 规范化音频路径，确保使用正确的路径分隔符
        audio_path_values = [os.path.normpath(path) for path in audio_path_values]

        dict = {}

        for i in range(len(ip_keys)):
            save_path = os.path.join(self.user_result_sadtalker_path,ip_keys[i])
            audio_i = audio_path_values[i]
            if not os.path.exists(audio_i):
                logger.warning(f"音频文件不存在，仍尝试推理: {audio_i}")
            else:
                logger.info(f"SadTalker 输入音频: {audio_i} -> 输出: {save_path}.mp4")

            self.Sad.Perform_Inference(image, audio_i, save_path)

            dict[ip_keys[i]] = os.path.normpath(save_path + ".mp4")
            
        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Video_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w',encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(dict, json_file)

    #根据备注推理音频_test
    def Inference_VITS_test(self, ref_wav_path, ref_text, text):
        # 修复：使用VITS输出路径而不是数据保存路径
        output_path = os.path.normpath(os.path.join(self.user_result_vits_path, "Test_VITS.wav"))

        # 确保输出目录存在
        os.makedirs(self.user_result_vits_path, exist_ok=True)

        self.VITS.Initialize_Models()
        self.VITS.Perform_Inference(
            ref_wav_path = ref_wav_path,
            prompt_text = ref_text,
            prompt_languageself = self.VITS.i18n("中文"),
            target_text = text,
            target_text_language = self.VITS.i18n("中文"),
            cut = self.VITS.i18n("凑50字一切"),
            output_path=output_path
        )
        return output_path
    
    
    #生成数字人视频_test
    def Inference_SadTalker_test(self, image, audio_path, save_path):
        self.Sad.Perform_Inference(image, audio_path, save_path)

#VITS跟Wav2Lip结合
class VITS_Wav2Lip_Join():
    def __init__(self,vits,wav2lip,save):
        
        #声明模型
        self.VITS = None
        self.WM = None
        
        #实例化
        self.VITS = GPT_SoVITS_Model()
        self.WM = Wav2Lip_Model()
        
        #全局变量
        self.user_result_vits_path = vits
        self.user_result_wav2lip_path = wav2lip
        self.user_data_save_path = save
        
        
        #清空文件
        Clear_File(self.user_result_vits_path)
        Clear_File(self.user_result_wav2lip_path)
        

    #预加载参数和模型
    def Set_Params_and_Model(self,w2l_parames_yaml_path=None,vits_parames_yaml_path=None):
        self.VITS.Initialize_Parames(vits_parames_yaml_path)
        
        self.WM.Initialize_Parames(self.user_data_save_path, w2l_parames_yaml_path)
        Clear_File(self.WM.wav2lip_temp)
        self.WM.Initialize_Models()
        
    #根据备注推理音频
    def Inference_VITS(self, ref_wav_path, ref_text): 
        path = os.path.join(self.user_data_save_path,"PPT_Remake.json")
        #读取Json_Data/PPT_Remake.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 自然排序键，确保按 Slide 序号顺序生成
        def _key_to_index(k):
            try:
                import re
                m = re.search(r"(\d+)", str(k))
                return int(m.group(1)) if m else float('inf')
            except Exception:
                return float('inf')
        ordered_keys = sorted(list(data.keys()), key=_key_to_index)
        ip_keys = ordered_keys
        PPT_Remake_values = [data[k] for k in ordered_keys]
        
        dict = {}

        for i in range(len(ip_keys)):
            
            text = PPT_Remake_values[i]
            output_path = os.path.normpath(os.path.join(self.user_result_vits_path, f"{ip_keys[i]}.wav")) 
            
            self.VITS.Initialize_Models()
            self.VITS.Perform_Inference(
                ref_wav_path = ref_wav_path,
                prompt_text = ref_text,
                prompt_languageself = self.VITS.i18n("中文"),
                target_text = text,
                target_text_language = self.VITS.i18n("中文"),
                cut = self.VITS.i18n("凑50字一切"),
                output_path=output_path
            )
            
            dict[ip_keys[i]] = output_path

        # 规范化所有路径，确保使用正确的路径分隔符
        normalized_dict = {}
        for key, path in dict.items():
            normalized_dict[key] = os.path.normpath(path)

        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Audio_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(normalized_dict, json_file)

        Save_Tiem(self.user_data_save_path, self.user_result_vits_path)

    #生成数字人视频带动作
    def Inference_Wav2Lip(self, video_path):
        #修改FPS
        fps_video = os.path.join(self.WM.wav2lip_temp, "fps_video.mp4")
        print("Change Video Fps")
        self.WM.Change_Video_Fps(video_path, fps_video, 25.0)     
        
        path = os.path.join(self.user_data_save_path,"Audio_save_path.json")
        #读取Audio_save_path.json里面的键值对
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        #把键跟值分别保存list
        ip_keys = list(data.keys())
        audio_path_values = list(data.values())

        # 规范化音频路径，确保使用正确的路径分隔符
        audio_path_values = [os.path.normpath(path) for path in audio_path_values]

        dict = {}

        for i in range(len(ip_keys)):
            save_path = os.path.join(self.user_result_wav2lip_path,ip_keys[i] + ".mp4")

            shear_video = self.WM.Shear_Video(fps_video, audio_path_values[i])
            self.WM.Perform_Inference(shear_video, audio_path_values[i], save_path)

            dict[ip_keys[i]] = save_path
            
        # 指定要写入的JSON文件路径
        json_file_path =os.path.join(self.user_data_save_path,"Video_save_path.json")

        # 使用with语句打开文件，确保在写入完成后自动关闭文件
        with open(json_file_path, 'w',encoding='utf-8') as json_file:
            # 使用json.dump()函数将数据写入JSON文件
            json.dump(dict, json_file)
    
if __name__ == '__main__':

    result_vits_user_path, result_sadtalker_user_path, result_wav2lip_user_path, user_data_save_path = Create_File("Hui")
    
   