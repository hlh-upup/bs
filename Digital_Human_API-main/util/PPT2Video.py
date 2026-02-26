from rembg import remove, new_session
import cv2
import os
import shutil
import subprocess
from moviepy.video.io.VideoFileClip import VideoFileClip
import win32com.client
import pythoncom


class Ppt_2_Video():

    def __init__(self, output_frames, input_frames, Mov_Video):
        self.outputFolder = output_frames
        self.inputFolder = input_frames
        self.outputVideo = Mov_Video
        self.time = 0
        self.fps = 25

        # 初始化PowerPoint应用程序
        self.powerpoint = None
        self.presentation = None
    def Export_Slides_As_Images(self):
        """将每页幻灯片导出为图片，保存到 input_frames 目录"""
        if not self.presentation:
            print("PPT文件未打开，无法导出图片")
            return False
        input_folder_abs = os.path.abspath(self.inputFolder)
        os.makedirs(input_folder_abs, exist_ok=True)
        for i in range(1, self.presentation.Slides.Count + 1):
            slide = self.presentation.Slides(i)
            filename = os.path.join(input_folder_abs, f"frame_{i-1:05d}.png")
            slide.Export(filename, "PNG")
            print(f"导出幻灯片 {i} 为图片: {filename}")
        return True
    def _normalize_ppt_path(self, ppt_path: str) -> str:
        if ppt_path is None:
            raise ValueError("ppt_path 不能为空")
        p = str(ppt_path).strip().strip('"').strip("'")
        p = os.path.expandvars(os.path.expanduser(p))
        # 统一为反斜杠，减少转义干扰
        p = p.replace('/', '\\')
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        # 处理极长路径
        try:
            if len(p) >= 260 and not p.startswith('\\\\?\\'):
                if p.startswith('\\\\'):  # UNC: \\server\share -> \\?\UNC\server\share\...
                    p = '\\\\?\\UNC\\' + p.lstrip('\\')
                else:
                    p = '\\\\?\\' + p
        except Exception:
            pass
        return p

    def _ensure_existing_path(self, p: str) -> str:
        """确保路径存在；如带有 \\?\ 前缀尝试去前缀再判断一遍。返回可用路径。"""
        if os.path.exists(p):
            return p
        # 如果是长路径前缀，尝试去掉再判断
        try:
            if p.startswith('\\\\?\\UNC\\'):
                alt = '\\\\' + p[len('\\\\?\\UNC\\'):]
            elif p.startswith('\\\\?\\'):
                alt = p[4:]
            else:
                alt = None
            if alt and os.path.exists(alt):
                return alt
        except Exception:
            pass
        raise FileNotFoundError(f"PPT 路径不存在: {p}")

    def Open_Ppt(self, ppt_path):
        """打开PPT文件（自动规范化路径并显式带窗口打开）。"""
        try:
            # 初始化COM（需与 Quit 成对）
            pythoncom.CoInitialize()

            # 预处理路径
            resolved = self._normalize_ppt_path(ppt_path)
            resolved = self._ensure_existing_path(resolved)

            # 启动PowerPoint应用程序
            self.powerpoint = win32com.client.Dispatch("PowerPoint.Application")
            self.powerpoint.Visible = True  # 设置为可见以便调试

            # 显式参数：ReadOnly=False, Untitled=False, WithWindow=True
            self.presentation = self.powerpoint.Presentations.Open(resolved, False, False, True)
            print(f"成功打开PPT文件: {resolved}")
            # 打开后自动导出所有幻灯片为图片
            self.Export_Slides_As_Images()
            return True

        except Exception as e:
            try:
                print(f"打开PPT文件失败: {e}")
                print(f"当前工作目录: {os.getcwd()}")
                print(f"传入原始路径: {ppt_path}")
            except Exception:
                pass
            return False

    def Set_Ppt_Transtion_Speed(self, SlideIndex, transition_time):
        """设置幻灯片切换时间"""
        try:
            if not self.presentation:
                print("PPT文件未打开")
                return False

            # 获取指定索引的幻灯片对象（注意：索引从1开始）
            slide = self.presentation.Slides(SlideIndex + 1)

            # 获取幻灯片的切换设置
            slide_show_transition = slide.SlideShowTransition

            # 设置切换时间
            slide_show_transition.AdvanceOnTime = True
            slide_show_transition.AdvanceTime = transition_time

            print(f"设置第{SlideIndex + 1}页切换时间为: {transition_time}秒")
            return True

        except Exception as e:
            print(f"设置幻灯片切换时间失败: {e}")
            return False

    def Ppt_Add_Replace(self, SlideIndex, VideoPath):
        """在幻灯片中添加视频"""
        try:
            if not self.presentation:
                print("PPT文件未打开")
                return False

            # 获取指定索引的幻灯片对象
            slide = self.presentation.Slides(SlideIndex + 1)

            # 获取幻灯片尺寸
            left = self.presentation.PageSetup.SlideWidth - 100
            top = self.presentation.PageSetup.SlideHeight - 100

            # 添加视频对象
            video = slide.Shapes.AddMediaObject2(
                FileName=VideoPath,
                LinkToFile=False,
                Left=left,
                Top=top,
                Width=10,
                Height=10
            )

            # 设置视频播放选项
            video.AnimationSettings.PlaySettings.PlayOnEntry = True
            video.AnimationSettings.PlaySettings.LoopUntilStopped = True

            print(f"在第{SlideIndex + 1}页添加视频: {VideoPath}")
            return True

        except Exception as e:
            print(f"添加视频到幻灯片失败: {e}")
            return False

    def Create_Replace_Video(self, end_time):
        """创建替换视频"""
        try:
            start_time = 0  # 剪辑起始时间（秒）
            video_clip = VideoFileClip(self.m_ReplaceMainPath)
            trimmed_video = video_clip.subclip(start_time, end_time)
            trimmed_video.write_videofile(self.m_ReplacePath, codec="libx264", fps=24)
            return True

        except Exception as e:
            print(f"创建替换视频失败: {e}")
            return False

    def Create_Ppt_Base_Video(self, OutPptMainBaseVideoPath):
        """将PPT导出为视频"""
        try:
            if not self.presentation:
                print("PPT文件未打开")
                return False

            video_output_path = os.path.abspath(OutPptMainBaseVideoPath)
            print(f"开始导出PPT视频到: {video_output_path}")

            # 导出为视频
            self.presentation.CreateVideo(video_output_path)

            # 等待导出完成
            while self.presentation.CreateVideoStatus == 1:
                print(f"导出进度: {self.presentation.CreateVideoStatus}")
                import time
                time.sleep(1)

            print("PPT视频导出完成")
            return True

        except Exception as e:
            print(f"创建PPT基础视频失败: {e}")
            return False

    def Get_Video_Time(self, video_path):
        """获取视频时长"""
        try:
            # 使用OpenCV打开视频文件
            video = cv2.VideoCapture(video_path)

            # 获取视频的帧数和帧率
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算视频时长（单位：秒）
            Time = frame_count / 25

            video.release()
            return Time

        except Exception as e:
            print(f"获取视频时长失败: {e}")
            return 0

    def Insert_Video(self, MainVideo, AuxiliaryVideo, OutPutVideoName, InsertionTime, Time):
        """将辅助视频插入到主视频中"""
        try:
            # 基础检查与准备
            if not os.path.exists(MainVideo):
                print(f"插入视频失败: 主视频不存在 -> {MainVideo}")
                return False
            if not os.path.exists(AuxiliaryVideo):
                print(f"插入视频失败: 辅助视频不存在 -> {AuxiliaryVideo}")
                return False
            out_dir = os.path.dirname(OutPutVideoName)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            # 获取主视频的高度
            ffprobe_command = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=height",
                "-of", "csv=s=x:p=0",
                MainVideo
            ]
            try:
                process = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                out = (process.stdout or b"").decode().strip()
                main_height = int(out) if out.isdigit() else None
            except Exception as e:
                print(f"ffprobe 读取高度失败，尝试OpenCV回退: {e}")
                main_height = None

            if not main_height or main_height <= 0:
                try:
                    cap = cv2.VideoCapture(MainVideo)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    main_height = h if h and h > 0 else 1080
                except Exception as e:
                    print(f"OpenCV 回退获取高度失败，使用默认1080: {e}")
                    main_height = 1080

            # 计算辅助视频的缩放尺寸
            auxiliary_height = main_height // 5 * 4
            auxiliary_width = -2  # 使用-2表示宽度按比例缩放

            print(f"Insert_Video: 计算参数 -> main_height={main_height}, insertion_time={InsertionTime}, duration={Time}")

            # 说明：
            # - 移除 Windows 下会失效的 enable='between(...)' 单引号写法，改用 trim+setpts 精确控制叠加时段；
            # - 避免使用 -itsoffset，直接在滤镜里 setpts=PTS+offset/TB，减少时基错位；
            # - format=rgba 确保辅助流的透明度保留；
            # - 不复制音频（通常无音频流），统一输出像素格式，提升兼容性。
            # 修复：不能使用 shortest=1，否则输出会在较短流结束时被截断
            # 例如主视频10s，叠加6s 延迟3s 开始，shortest=1 会在 9s 处结束（10 与 3+6 的较短者）。
            # 使用 eof_action=pass 让叠加流结束后继续输出主视频剩余部分。
            filter_str = (
                f"[1:v]scale={auxiliary_width}:{auxiliary_height}:flags=lanczos,setsar=1,format=rgba,"
                f"trim=start=0:end={Time},setpts=PTS+{InsertionTime}/TB[ovr];"
                f"[0:v][ovr]overlay=W-w-10:H-h:format=auto:eof_action=pass"
            )

            ffmpeg_command = [
                "ffmpeg",
                "-i", MainVideo,
                "-i", AuxiliaryVideo,
                "-filter_complex", filter_str,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-an",
                OutPutVideoName
            ]

            # 调用 FFmpeg 命令
            print(f"Insert_Video: 执行FFmpeg命令 -> {' '.join(ffmpeg_command)}")
            proc = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0:
                stderr_text = (proc.stderr or b"").decode(errors='ignore')
                print(f"Insert_Video: FFmpeg 退出码={proc.returncode}, stderr前500字符: {stderr_text[:500]}")
            else:
                print(f"Insert_Video: FFmpeg 成功退出，返回码=0")

            # 结果文件检查
            ok = os.path.exists(OutPutVideoName)
            size = os.path.getsize(OutPutVideoName) if ok else -1
            print(f"插入视频完成: {OutPutVideoName}, exists={ok}, size={size}")
            if not ok or size <= 0:
                return False
            return True

        except Exception as e:
            print(f"插入视频失败: {e}")
            return False

    def Remove_Background(self):
        """去除图片背景"""
        try:
            session = new_session()

            if os.path.exists(self.inputFolder):
                shutil.rmtree(self.inputFolder)
            os.makedirs(self.inputFolder)

            # 处理每张图像
            for file in os.listdir(self.outputFolder):
                input_path = os.path.join(self.outputFolder, file)
                output_path = os.path.join(self.inputFolder, file)

                with open(input_path, 'rb') as i:
                    input_image = i.read()
                    output_image = remove(input_image, session=session)

                # 将处理后的图像写入输出文件
                with open(output_path, 'wb') as o:
                    o.write(output_image)

            print("背景去除完成")
            return True

        except Exception as e:
            print(f"去除背景失败: {e}")
            return False

    def Create_Video(self, key):
        """创建视频"""
        try:
            Mov_Video = os.path.join(self.outputVideo, f"{key}_Mov.mov")

            ffmpeg_command = [
                "ffmpeg",
                "-framerate", str(self.fps),
                "-i", f"{self.inputFolder}/frame_%05d.png",
                "-c:v", "qtrle",
                Mov_Video
            ]

            # 调用 FFmpeg 命令
            subprocess.run(ffmpeg_command)
            print(f"视频创建完成: {Mov_Video}")
            return True

        except Exception as e:
            print(f"创建视频失败: {e}")
            return False

    def Video_To_Frames(self, video_path):
        """将视频转换为帧"""
        try:
            if os.path.exists(self.outputFolder):
                shutil.rmtree(self.outputFolder)
            os.makedirs(self.outputFolder)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)

            # 获取视频帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 获取视频帧率
            self.fps = cap.get(cv2.CAP_PROP_FPS)

            # 获取视频帧尺寸
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 输出视频信息
            print("Frame count:", frame_count)
            print("FPS:", self.fps)
            print("Frame size:", (frame_width, frame_height))

            aspect_ratio_width = 9
            aspect_ratio_height = 16

            # 计算截取区域的宽度，保持高度不变

            crop_width = int(frame_height * (aspect_ratio_width / aspect_ratio_height))
            # 修正：宽度不足时不裁剪，直接保存原图
            if crop_width > frame_width:
                print("Warning: crop width larger than frame width, will not crop, save original frame.")
                start_x = 0
                end_x = frame_width
            else:
                start_x = (frame_width - crop_width) // 2
                end_x = start_x + crop_width

            # 逐帧读取视频，并保存成图片
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame[:, start_x:end_x]

                # 保存图片
                frame_filename = os.path.join(self.outputFolder, f"frame_{frame_number:05d}.png")
                cv2.imwrite(frame_filename, frame)

                frame_number += 1

                # 显示进度
                print(f"Processed frame {frame_number}/{frame_count}")

            # 释放视频对象
            cap.release()

            print("Frames extracted successfully!")
            return True

        except Exception as e:
            print(f"视频转帧失败: {e}")
            return False

    def Close(self):
        """关闭PowerPoint应用程序"""
        try:
            if self.presentation:
                self.presentation.Close()
                self.presentation = None

            if self.powerpoint:
                self.powerpoint.Quit()
                self.powerpoint = None

            # 释放COM
            pythoncom.CoUninitialize()
            print("PowerPoint应用程序已关闭")

        except Exception as e:
            print(f"关闭PowerPoint失败: {e}")

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.Close()


if __name__ == '__main__':
    # 测试代码
    Ppt = Ppt_2_Video("output_frames", "input_frames", "Mov_Video")

    # 测试打开PPT文件
    if Ppt.Open_Ppt(r"F:\hlh\Test.pptx"):
        print("PPT打开成功")
        Ppt.Close()