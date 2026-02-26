/**
 * 数字人视频生成服务
 * 基于C#原版的成功前后端配合逻辑设计
 */

import { digitalHumanApi } from './api'

export interface VideoGenerationConfig {
  user: string
  onProgress?: (step: string, progress: number) => void
  onError?: (error: string) => void
}

export class VideoGenerationService {
  /**
   * 完整的视频生成流程
   * 基于C#原版的CreatePptVideo逻辑
   */
  async generateCompleteVideo(
    config: VideoGenerationConfig,
    pptVideoFile: File,
    imageFile: File,
    refAudioFile?: File,
    refText?: string
  ): Promise<Blob | null> {
    const { user, onProgress, onError } = config

    try {
      // 1. 用户登录 (如果需要)
      onProgress?.('准备登录...', 0)

      // 2. 发送用户照片
      onProgress?.('上传用户照片...', 5)
      const imageBase64 = await this.fileToBase64(imageFile)
      const imageSuccess = await digitalHumanApi.sendImage({
        User: user,
        Img: imageBase64
      })
      if (!imageSuccess) {
        throw new Error('用户照片上传失败')
      }

      // 3. 配置参数
      onProgress?.('配置数字人参数...', 10)
      const configSuccess = await digitalHumanApi.sendConfig({
        User: user,
        VITS_Config: {
          // VITS配置参数
        },
        SadTalker_Config: {
          // SadTalker配置参数
        }
      })
      if (!configSuccess) {
        throw new Error('参数配置失败')
      }

      // 4. 发送参考音频和文字 (如果提供)
      if (refAudioFile && refText) {
        onProgress?.('上传参考音频...', 15)
        const refSuccess = await digitalHumanApi.sendRefWavAndText(
          { User: user, Ref_Text: refText },
          refAudioFile
        )
        if (!refSuccess) {
          throw new Error('参考音频上传失败')
        }
      }

      // 5. 开始推理 (异步任务)
      onProgress?.('开始数字人生成推理...', 20)
      const inferenceSuccess = await digitalHumanApi.getInference({ User: user })
      if (!inferenceSuccess) {
        throw new Error('推理启动失败')
      }

      // 6. 等待推理完成
      onProgress?.('正在进行数字人推理...', 30)
      const inferenceCompleted = await digitalHumanApi.waitForTask(
        user,
        'Audio_Video_Inference',
        60000, // 60秒间隔
        30 * 60 * 1000 // 30分钟超时
      )
      if (!inferenceCompleted) {
        throw new Error('推理超时或失败')
      }

      // 7. 获取音频时长数据
      onProgress?.('获取音频时长信息...', 50)
      const timeData = await digitalHumanApi.receiveWavTime({ User: user })
      if (!timeData) {
        throw new Error('获取音频时长失败')
      }

      // 8. 发送PPT视频
      onProgress?.('上传PPT视频...', 55)
      const videoSuccess = await digitalHumanApi.sendVideo(
        { User: user },
        pptVideoFile
      )
      if (!videoSuccess) {
        throw new Error('PPT视频上传失败')
      }

      // 9. 开始视频融合 (异步任务)
      onProgress?.('开始视频融合...', 60)
      const mergeSuccess = await digitalHumanApi.pptVideoMerge({ User: user })
      if (!mergeSuccess) {
        throw new Error('视频融合启动失败')
      }

      // 10. 等待视频融合完成
      onProgress?.('正在进行视频融合...', 70)
      const mergeCompleted = await digitalHumanApi.waitForTask(
        user,
        'Video_Merge',
        60000, // 60秒间隔
        30 * 60 * 1000 // 30分钟超时
      )
      if (!mergeCompleted) {
        throw new Error('视频融合超时或失败')
      }

      // 11. 下载最终视频
      onProgress?.('下载最终视频...', 90)
      const finalVideo = await digitalHumanApi.pullVideoMerge({ User: user })
      if (!finalVideo) {
        throw new Error('视频下载失败')
      }

      onProgress?.('视频生成完成！', 100)
      return finalVideo

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '视频生成失败'
      onError?.(errorMessage)
      return null
    }
  }

  /**
   * 快速生成模式 (仅推理，不包含PPT融合)
   */
  async generateInferenceOnly(
    config: VideoGenerationConfig,
    imageFile: File,
    refAudioFile?: File,
    refText?: string
  ): Promise<Blob | null> {
    const { user, onProgress, onError } = config

    try {
      // 1. 发送用户照片
      onProgress?.('上传用户照片...', 10)
      const imageBase64 = await this.fileToBase64(imageFile)
      await digitalHumanApi.sendImage({ User: user, Img: imageBase64 })

      // 2. 发送参考音频和文字 (如果提供)
      if (refAudioFile && refText) {
        onProgress?.('上传参考音频...', 20)
        await digitalHumanApi.sendRefWavAndText(
          { User: user, Ref_Text: refText },
          refAudioFile
        )
      }

      // 3. 开始推理
      onProgress?.('开始数字人推理...', 30)
      const inferenceSuccess = await digitalHumanApi.getInference({ User: user })
      if (!inferenceSuccess) {
        throw new Error('推理启动失败')
      }

      // 4. 等待推理完成
      onProgress?.('正在进行推理...', 40)
      const completed = await digitalHumanApi.waitForTask(
        user,
        'Audio_Video_Inference'
      )
      if (!completed) {
        throw new Error('推理失败')
      }

      onProgress?.('推理完成！', 100)

      // 这里可以返回推理结果或测试视频
      return new Blob() // 临时返回空blob

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '推理失败'
      onError?.(errorMessage)
      return null
    }
  }

  /**
   * 文件转Base64
   */
  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        // 移除data:image/...;base64,前缀
        const base64 = result.split(',')[1]
        resolve(base64)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  /**
   * 下载视频文件
   */
  downloadVideo(videoBlob: Blob, filename: string = 'digital-human-video.mp4') {
    const url = URL.createObjectURL(videoBlob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  /**
   * 预估视频生成时间
   */
  estimateGenerationTime(audioLength?: number): number {
    // 基于经验公式估算时间 (秒)
    const baseTime = 120 // 基础时间2分钟
    const audioTime = audioLength ? audioLength * 0.5 : 0 // 音频长度的一半
    const processingTime = 180 // 处理时间3分钟

    return baseTime + audioTime + processingTime
  }
}

// 单例模式
export const videoGenerationService = new VideoGenerationService()