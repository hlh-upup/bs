import shutil
import torch
import os, sys, time
from argparse import ArgumentParser

import yaml

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


class SadTalker_Model():
    def __init__(self):
        
        #全局变量
        self.parser = None
        self.args = None
        self.sadtalker_paths = None #绝对路径
        
        #模型
        self.preprocess = None
        self.audio_to = None
        self.animate_from = None
    
    @staticmethod
    def _normalize_enhancer(val):
        """将布尔或大小写不规范的增强器参数规范化为期望的字符串或 None。
        允许: 'gfpgan' | 'RestoreFormer' | 'codeformer' | None。
        True -> 'gfpgan'; False/空 -> None。
        """
        if val is None:
            return None
        # 处理布尔
        if isinstance(val, bool):
            return 'gfpgan' if val else None
        # 处理字符串
        s = str(val).strip()
        if s == '':
            return None
        sl = s.lower()
        if sl in ('true', '1', 'yes', 'y', 'gfpgan'):
            return 'gfpgan'
        if sl in ('false', '0', 'no', 'n', 'none', 'null'):
            return None
        if sl == 'restoreformer':
            return 'RestoreFormer'
        if sl == 'codeformer':
            return 'codeformer'
        # 其他字符串原样返回，交由下游校验
        return s

    @staticmethod
    def _normalize_bg_upsampler(val):
        """背景放大器的规范化: True -> 'realesrgan'; False/空 -> None; 其他字符串仅保留 'realesrgan'。"""
        if val is None:
            return None
        if isinstance(val, bool):
            return 'realesrgan' if val else None
        s = str(val).strip()
        if s == '':
            return None
        sl = s.lower()
        if sl in ('true', '1', 'yes', 'y', 'realesrgan'):
            return 'realesrgan'
        if sl in ('false', '0', 'no', 'n', 'none', 'null'):
            return None
        # 非支持值置空，避免后续报错
        return None
    
    #初始化其他变量
    def Initialize_Parames(self,user_data_save_path, parames=None): 
        self.parser = ArgumentParser()
        self.user_data_save_path = user_data_save_path
        
        if parames == None:
            # 添加默认参数
            with open('SadTalker/SadTalker_config.yaml', 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
                
            self.args = self.parser.parse_args()
        
        else:
            with open(parames, 'r') as stream:
                config = yaml.safe_load(stream)
            for key, value in config.items():
                self.parser.add_argument(f"--{key}", default=value)
                
            self.args = self.parser.parse_args()
        
        # 统一增强器/背景增强器设置，避免传入布尔导致模型报错
        try:
            self.args.enhancer = SadTalker_Model._normalize_enhancer(getattr(self.args, 'enhancer', None))
            self.args.background_enhancer = SadTalker_Model._normalize_bg_upsampler(getattr(self.args, 'background_enhancer', None))
            print(f"SadTalker config normalized: enhancer={self.args.enhancer}, background_enhancer={self.args.background_enhancer}")
        except Exception as _norm_e:
            print(f"SadTalker config normalize failed (ignore): {_norm_e}")
        
        # 设置设备
        if torch.cuda.is_available() and not self.args.cpu:
            self.args.device = "cuda"
        else:
            self.args.device = "cpu"
            
        # 初始化路径
        # current_root_path = os.path.split(sys.argv[0])[0]
        current_root_path = os.path.dirname(os.path.abspath(__file__))
        # self.sadtalker_paths = init_path(self.args.checkpoint_dir, os.path.join(current_root_path, 'SadTalker/src/config'), self.args.size, self.args.old_version, self.args.preprocess)
        self.sadtalker_paths = init_path(self.args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), self.args.size, self.args.old_version, self.args.preprocess)
    
    #初始化模型
    def Initialize_Models(self):
        self.preprocess = CropAndExtract(self.sadtalker_paths, self.args.device)
        self.audio_to = Audio2Coeff(self.sadtalker_paths, self.args.device)
        self.animate_from = AnimateFromCoeff(self.sadtalker_paths, self.args.device)

        
    #推理
    def Perform_Inference(self,image_save_path,audio_save_path,save_path_name):
        #torch.backends.cudnn.enabled = False
        args = self.args

        pic_path = image_save_path #照片位置
        audio_path = audio_save_path #音频位置
        save_dir = save_path_name #保存位置   
        temp_path = os.path.join(self.user_data_save_path, save_dir)
        
        os.makedirs(temp_path, exist_ok=True)
        pose_style = args.pose_style
        device = args.device
        batch_size = args.batch_size
        input_yaw_list = args.input_yaw
        input_pitch_list = args.input_pitch
        input_roll_list = args.input_roll
        ref_eyeblink = args.ref_eyeblink
        ref_pose = args.ref_pose

        #初始化模型
        preprocess_model = self.preprocess
        audio_to_coeff = self.audio_to
        animate_from_coeff = self.animate_from

        ##裁剪图像并从图像中提取3dmm
        first_frame_dir = os.path.join(temp_path, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, args.preprocess,\
                                                                                source_image_flag=True, pic_size=args.size)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(temp_path, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_eyeblink_coeff_path=None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink: 
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(temp_path, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess, source_image_flag=False)
        else:
            ref_pose_coeff_path=None

        #audio2ceoff
        batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
        coeff_path = audio_to_coeff.generate(batch, temp_path, pose_style, ref_pose_coeff_path)

        # 3dface render
        if args.face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(temp_path, '3dface.mp4'))
        
        #coeff2video
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                    batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                    expression_scale=args.expression_scale, still_mode=args.still, preprocess=args.preprocess, size=args.size)
        
        #save_dir
        result = animate_from_coeff.generate(data, temp_path, pic_path, crop_info, self.user_data_save_path, \
                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer, preprocess=args.preprocess, img_size=args.size)
        
        shutil.move(result, save_dir+'.mp4')
        print('生成的视频命名为:', save_dir+'.mp4')

        if not args.verbose:
            shutil.rmtree(temp_path)



if __name__ == '__main__':
    
    # 创建一个线程池
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)  # 根据需求设置最大工作线程数量

    Sta = SadTalker_Model()
    Sta.Initialize_Parames("Data\\Hui")
    Sta.Initialize_Models()
    
    Sta.Perform_Inference("Data\\Hui\\Image.png","6.WAV","AAA")
    
    # 提交任务给线程池处理
    # executor.submit(perform_inference, args, preprocess, audio_to, animate_from)

