import axios from 'axios'

// 调试开关：在 .env.[mode] 中设置 VITE_APP_DEBUG=true 即可输出详细日志
const DEBUG = import.meta.env.VITE_APP_DEBUG === 'true'

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 3600000, // 1 hour timeout for long operations
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求/响应拦截器用于调试
api.interceptors.request.use((config) => {
  if (DEBUG) {
    ;(config as any).metadata = { start: Date.now() }
    console.log(
      '%c[API REQUEST]',
      'color:#2563eb',
      config.method?.toUpperCase(),
      config.url,
      'data:',
      config.data,
    )
  }
  return config
})

api.interceptors.response.use(
  (response) => {
    if (DEBUG) {
      const meta = (response.config as any).metadata
      const duration = meta?.start ? Date.now() - meta.start : 'N/A'
      console.log(
        '%c[API RESPONSE]',
        'color:#16a34a',
        response.config.url,
        'status:',
        response.status,
        'time:',
        duration + 'ms',
        'data:',
        response.data,
      )
    }
    return response
  },
  (error) => {
    if (DEBUG) {
      if (error.response) {
        console.warn(
          '[API ERROR]',
          error.config?.url,
          'status:',
          error.response.status,
          'data:',
          error.response.data,
        )
      } else {
        console.warn('[API ERROR]', error.message)
      }
    }
    return Promise.reject(error)
  },
)

export interface LoginRequest {
  User: string
  Password: string
}

export interface ApiResponse {
  result: string
}

export interface ConfigRequest {
  User: string
  VITS_Config: any
  SadTalker_Config: any
}

export interface ImageRequest {
  User: string
  Img: string
}

export interface PPTRequest {
  User: string
  PPT_Remakes: string | Record<string, string> // 支持字符串或对象
}

export interface TrainRequest {
  User: string
  Label: string
}

export interface TaskRequest {
  User: string
  Task: string
}

export interface ModelRequest {
  User: string
  Index?: string
}

export interface AudioRequest {
  User: string
  Audio_Name: string
}

export interface RefRequest {
  User: string
  Ref_Text: string
}

class DigitalHumanApiService {
  private async handleResponse<T>(response: any): Promise<T> {
    if (response.data?.result === 'Failed') {
      throw new Error('API request failed')
    }
    return response.data
  }

  async login(credentials: LoginRequest): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Login', credentials)
    return response.data?.result === 'Success'
  }

  async sendImage(request: ImageRequest): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_Image', request)
    return response.data?.result === 'Success'
  }

  async sendPPTRemakes(request: PPTRequest): Promise<boolean> {
    // 智能处理PPT备注数据格式
    let processedRequest = { ...request }

    if (typeof request.PPT_Remakes === 'string') {
      // 如果是字符串，尝试解析为JSON对象
      try {
        const parsed = JSON.parse(request.PPT_Remakes)
        processedRequest.PPT_Remakes = parsed
      } catch (e) {
        // 如果解析失败，保持原样（可能是后端期望的字符串格式）
        console.warn('PPT_Remakes is not valid JSON, sending as string:', e)
      }
    }

    console.log('发送PPT备注数据:', processedRequest)
    const response = await api.post<ApiResponse>('/Send_PPT_Remakes', processedRequest)
    return response.data?.result === 'Success'
  }

  async sendConfig(request: ConfigRequest): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_Config', request)
    return response.data?.result === 'Success'
  }

  // 发送Wav2Lip相关配置（若后端需要）
  async sendWav2LipConfig(request: { User: string; [k: string]: any }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_Wav2Lip_Config', request)
    return response.data?.result === 'Success'
  }

  async sendVideo(request: any, videoFile: File): Promise<boolean> {
    const formData = new FormData()
    // 对齐C#格式：将整个request对象作为JSON字符串放入"Json"字段
    formData.append('Json', JSON.stringify(request))
    formData.append('File', videoFile)

    const response = await api.post<ApiResponse>('/Send_Video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 7200000, // 2小时超时，对齐C#设置
    })
    return response.data?.result === 'Success'
  }

  // 发送教师视频（C#: SendTeacherVideo -> /Send_Teacher_Video）
  async sendTeacherVideo(request: any, videoFile: File): Promise<boolean> {
    const formData = new FormData()
    formData.append('Json', JSON.stringify(request))
    formData.append('File', videoFile)

    const response = await api.post<ApiResponse>('/Send_Teacher_Video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 7200000,
    })
    return response.data?.result === 'Success'
  }

  // 发送PPT单页音频（可选：用户自录音频场景）
  async sendPPTAudio(request: { User: string; Slide: number }, audioFile: File): Promise<boolean> {
    const formData = new FormData()
    formData.append('Json', JSON.stringify(request))
    formData.append('File', audioFile)
    const response = await api.post<ApiResponse>('/Send_PPT_Audio', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 7200000,
    })
    return response.data?.result === 'Success'
  }

  async sendTrainAudio(request: AudioRequest, audioFile: File): Promise<boolean> {
    const formData = new FormData()
    // 对齐C#格式：将整个request对象作为JSON字符串放入"Json"字段
    formData.append('Json', JSON.stringify(request))
    formData.append('File', audioFile)

    const response = await api.post<ApiResponse>('/Send_Tarin_Audio', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 7200000, // 2小时超时，对齐C#设置
    })
    return response.data?.result === 'Success'
  }

  async sendRefWavAndText(request: RefRequest, audioFile: File): Promise<boolean> {
    const formData = new FormData()
    // 对齐C#格式：将整个request对象作为JSON字符串放入"Json"字段
    formData.append('Json', JSON.stringify(request))
    formData.append('File', audioFile)

    const response = await api.post<ApiResponse>('/Send_Ref_Wav_And_Text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 7200000, // 2小时超时，对齐C#设置
    })
    return response.data?.result === 'Success'
  }

  async uploadPPTParseRemakes(
    request: { User: string },
    pptFile: File,
  ): Promise<{ success: boolean; data?: any; error?: string }> {
    const formData = new FormData()
    // 对齐C#格式：将整个request对象作为JSON字符串放入"Json"字段
    formData.append('Json', JSON.stringify(request))
    formData.append('File', pptFile)

    const response = await api.post('/Upload_PPT_Parse_Remakes', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 300000, // 5分钟超时，用于PPT解析
    })

    if (response.data?.result === 'Success') {
      return {
        success: true,
        data: response.data.data,
      }
    } else {
      return {
        success: false,
        error: response.data?.error || 'PPT解析失败',
      }
    }
  }

  async trainVITSModel(request: TrainRequest): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Train_VITS_Model', request)
    return response.data?.result === 'Success'
  }

  // === 推理入口（不同后端分支） ===
  async getInference(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference', request)
    // 后端返回任务ID（如 "Audio_Video_Inference"），只要不是 "Failed" 就算成功
    return response.data?.result !== 'Failed'
  }

  async getInferenceVITS(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_VITS', request)
    return response.data?.result === 'Success'
  }

  // VITS推理（多音频）—— C#: GetInferenceVits -> /Get_Inference_VITS_Multiple
  async getInferenceVITS_Multiple(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_VITS_Multiple', request)
    // C# 中使用 FlaskResponseState，result 返回任务名；不是 "Failed" 即表示触发成功
    return response.data?.result !== 'Failed'
  }

  async getInferenceVITS_Sadtalker(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_VITS_Sadtalker', request)
    return response.data?.result !== 'Failed'
  }

  async getInferenceUser_Sadtalker(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_User_Audio_Sadtalker', request)
    return response.data?.result !== 'Failed'
  }

  async getInferenceVITS_Wav2Lip(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_VITS_Wav2Lip', request)
    return response.data?.result !== 'Failed'
  }

  async getInferenceUser_Wav2Lip(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Get_Inference_User_Audio_Wav2Lip', request)
    return response.data?.result !== 'Failed'
  }

  async pptVideoMerge(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/PPT_Video_Merge', request)
    // 后端返回任务ID（如 "Video_Merge"），只要不是 "Failed" 就算成功
    return response.data?.result !== 'Failed'
  }

  async pptVideoMergeSelectInto(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/PPT_Video_Merge_Select_Into', request)
    return response.data?.result !== 'Failed'
  }

  async pptVideoMergeNoInto(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/PPT_Video_Merge_No_Into', request)
    return response.data?.result !== 'Failed'
  }

  async pullVideoMerge(request: { User: string }): Promise<Blob> {
    const response = await api.post('/Pull_Video_Merge', request, {
      responseType: 'blob',
    })
    return response.data
  }

  // 获取生成的视频列表
  async getVideoList(request: { User: string }): Promise<
    {
      id: string
      name: string
      url: string
      duration: number
      size: number
      createTime: string
    }[]
  > {
    const response = await api.get('/Get_Video_List', { params: request })
    // 后端返回 { result: [...] }
    return response.data?.result || []
  }

  // 删除视频文件
  async deleteVideo(request: { User: string; VideoId: string }): Promise<boolean> {
    const response = await api.post('/Delete_Video', request)
    return response.data?.result === 'Success'
  }

  async pullVITSAudio(request: { User: string }): Promise<Blob> {
    const response = await api.post('/Pull_VITS_Audio', request, {
      responseType: 'blob',
    })
    return response.data
  }

  async getState(request: TaskRequest): Promise<string> {
    const response = await api.post<ApiResponse>('/Get_State', request)
    return response.data?.result || 'Failed'
  }

  async getTestInference(request: { User: string }): Promise<Blob> {
    const response = await api.post('/Get_Test_Inference', request, {
      responseType: 'blob',
    })
    return response.data
  }

  async receiveWavTime(request: { User: string }): Promise<any> {
    const response = await api.post('/Recive_Wav_Time', request)
    return response.data?.result
  }

  async receiveUserWavTime(request: { User: string }): Promise<any> {
    const response = await api.post('/Recive_User_Wav_Time', request)
    return response.data?.result
  }

  async selectVITSModel(request: ModelRequest): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_Select_VITS_Model', request)
    return response.data?.result === 'Success'
  }

  // 图片二次元化（后端）
  async cartoonizeImage(request: {
    User: string
    Img: string // base64-raw，无前缀
    Mode?: 'animegan_v2' | 'wbc' | 'cv_stylize' | 'bilateral'
    Style?:
      | 'hayao'
      | 'shinkai'
      | 'paprika'
      | 'celeba'
      | 'animeganv3_paprika'
      | 'paprika_v3'
      | 'animeganv3_hayao'
      | 'hayao_v3'
      | 'animeganv3_shinkai'
      | 'shinkai_v3'
    Params?: Record<string, any>
  }): Promise<{
    success: boolean
    img?: string
    error?: string
    mode_used?: string
    requested_mode?: string
    style_used?: string
    fallback?: { mode: string; error: string }[]
    debug?: any
  }> {
    console.info('[api] /Cartoonize_Image request', request)
    const response = await api.post('/Cartoonize_Image', request)
    if (response.data?.result === 'Success') {
      return {
        success: true,
        img: response.data.Img,
        mode_used: response.data.mode_used,
        requested_mode: response.data.requested_mode,
        style_used: response.data.style_used,
        fallback: response.data.fallback,
        debug: response.data.debug,
      }
    } else {
      console.error('[api] /Cartoonize_Image failed', response.data)
      return { success: false, error: response.data?.error || 'Cartoonize failed' }
    }
  }

  async selectTrainVITSModel(request: { User: string }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_Select_Train_VITS_Model', request)
    return response.data?.result === 'Success'
  }

  async getTrainVitsModelName(request: { User: string }): Promise<string | null> {
    const response = await api.post<ApiResponse>('/Get_Train_VITS_Model_Name', request)
    const result = response.data?.result
    if (result && result !== 'Failed') return result
    return null
  }

  async sendPeopleLocation(request: { User: string; Pages: number[] }): Promise<boolean> {
    const response = await api.post<ApiResponse>('/Send_People_Location', request)
    return response.data?.result === 'Success'
  }

  async generatePPTVideo(request: {
    User: string
  }): Promise<{ success: boolean; data?: any; error?: string }> {
    const response = await api.post('/Generate_PPT_Video', request, {
      timeout: 600000, // 10分钟超时，用于PPT视频生成
    })

    if (response.data?.result === 'Success') {
      return {
        success: true,
        data: response.data.data,
      }
    } else {
      return {
        success: false,
        error: response.data?.error || 'PPT视频生成失败',
      }
    }
  }

  async waitForTask(
    user: string,
    task: string,
    interval = 10000, // 10秒间隔
    timeout = 30 * 60 * 1000, // 默认 30 分钟
  ): Promise<boolean> {
    const start = Date.now()
    let attempt = 0
    console.log(`[TASK] 开始轮询任务: ${task}`)
    while (true) {
      attempt++
      console.log(`[TASK] 第${attempt}次查询任务状态: ${task}`)
      const state = await this.getState({ User: user, Task: task })
      console.log(`[TASK] 查询结果: ${task} = ${state}`)

      if (DEBUG) {
        console.log(`[TASK][${task}] attempt=${attempt} state=${state}`)
      }
      if (state === 'True') {
        console.log(`[TASK] 任务完成: ${task}`)
        return true
      }
      if (state === 'Failed') {
        console.warn(`[TASK] 任务失败: ${task}`)
        if (DEBUG) console.warn(`[TASK][${task}] 失败返回 Failed`)
        return false
      }
      // 处理新的"Processing"状态
      if (state === 'Processing' || state === 'False') {
        console.log(`[TASK] 任务进行中: ${task} = ${state}`)
        if (DEBUG) console.log(`[TASK][${task}] 进行中状态: ${state}`)
      }
      if (Date.now() - start > timeout) {
        console.warn(`[TASK] 任务超时: ${task}`)
        if (DEBUG) console.warn(`[TASK][${task}] 超时 (${timeout} ms)`)
        return false
      }
      console.log(`[TASK] 等待${interval / 1000}秒后继续查询: ${task}`)
      await new Promise((res) => setTimeout(res, interval))
    }
  }
}

export const digitalHumanApi = new DigitalHumanApiService()
export default api
