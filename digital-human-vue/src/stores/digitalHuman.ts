import { defineStore } from 'pinia'
import { digitalHumanApi } from '@/services/api'

export interface DigitalHumanState {
  isProcessing: boolean
  currentTask: string | null
  progress: number
  config: {
    enhancer: string | null
    expressionScale: number
    modelIndex: string
    resolution: string
    fps: string
  }
  uploadedFiles: {
    image: File | null
    audio: File | null
    ppt: File | null
    video: File | null
  }
  trainingData: {
    audioFiles: File[]
    labels: Record<string, string>
    refAudio: File | null
    refText: string
  }
  // 配置状态
  isPersonConfigured: boolean // 是否已配置数字人
  isVoiceConfigured: boolean // 是否已配置语音模型

  // 新增：视频生成相关选项（对齐 C# 全局 global 变量）
  videoOptions: {
    // 1：无动作(SadTalker) | 2：有动作(Wav2Lip)
    digitalMotion: 1 | 2
    // 1：全部插入 | 2：全部不插入 | 3：部分插入
    intoDigitalOperation: 1 | 2 | 3
    // 是否使用模型声音（true=模型生成，false=用户音频）
    useModelAudio: boolean
    // 选择插入的页码（部分插入时）
    selectedSlides: number[]
  }

  // 前端像素化选项（默认关闭，不影响现有行为）
  pixelateEnabled: boolean
  pixelBlockSize: number

  // 卡通化选项
  cartoonizeEnabled: boolean
  cartoonLevels: number
  cartoonEdge: number
  cartoonBlurPasses: number
  cartoonProcessMode?: 'frontend' | 'backend'
  cartoonBackendMode?: 'animegan_v2' | 'wbc' | 'cv_stylize' | 'bilateral'
  cartoonBackendStyle?:
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
  // Anime frame overlay
  animeFrameEnabled: boolean
  animeFrameStyle: 'panel' | 'glow' | 'film'
}

// 读取持久化设置（若存在）
function loadImageProcessSettings() {
  try {
    const raw = localStorage.getItem('digitalHuman:imageProcess')
    if (!raw) return {}
    return JSON.parse(raw)
  } catch {
    return {}
  }
}

function saveImageProcessSettings(patch: Record<string, any>) {
  try {
    const current = loadImageProcessSettings()
    const merged = { ...current, ...patch }
    localStorage.setItem('digitalHuman:imageProcess', JSON.stringify(merged))
  } catch {
    // ignore
  }
}

export const useDigitalHumanStore = defineStore('digitalHuman', {
  state: (): DigitalHumanState => ({
    isProcessing: false,
    currentTask: null,
    progress: 0,
    config: {
      enhancer: null,
      expressionScale: 1.0,
      modelIndex: '0',
      resolution: '1080p',
      fps: '30',
    },
    uploadedFiles: {
      image: null,
      audio: null,
      ppt: null,
      video: null,
    },
    trainingData: {
      audioFiles: [],
      labels: {},
      refAudio: null,
      refText: '',
    },
    // 配置状态初始值
    isPersonConfigured: false,
    isVoiceConfigured: false,

    // 视频生成选项默认值（与 C# 默认一致）
    videoOptions: {
      digitalMotion: 1,
      intoDigitalOperation: 1,
      useModelAudio: true,
      selectedSlides: [],
    },

    // 读取持久化：像素化 & 卡通化
    ...(() => {
      const p: any = loadImageProcessSettings()
      return {
        pixelateEnabled: p.pixelateEnabled ?? false,
        pixelBlockSize: p.pixelBlockSize ?? 10,
        cartoonizeEnabled: p.cartoonizeEnabled ?? false,
        cartoonLevels: p.cartoonLevels ?? 12,
        cartoonEdge: p.cartoonEdge ?? 80,
        cartoonBlurPasses: p.cartoonBlurPasses ?? 2,
        cartoonProcessMode: p.cartoonProcessMode ?? 'backend',
        cartoonBackendMode: p.cartoonBackendMode ?? 'animegan_v2',
        cartoonBackendStyle: p.cartoonBackendStyle ?? 'hayao',
        animeFrameEnabled: p.animeFrameEnabled ?? false,
        animeFrameStyle: p.animeFrameStyle ?? 'panel',
      }
    })(),
  }),

  actions: {
    setConfig(config: Partial<DigitalHumanState['config']>) {
      this.config = { ...this.config, ...config }
    },

    setVideoOptions(options: Partial<DigitalHumanState['videoOptions']>) {
      this.videoOptions = { ...this.videoOptions, ...options }
    },

    setPixelateEnabled(v: boolean) {
      this.pixelateEnabled = v
      saveImageProcessSettings({ pixelateEnabled: v })
    },

    setPixelBlockSize(v: number) {
      this.pixelBlockSize = v
      saveImageProcessSettings({ pixelBlockSize: v })
    },

    // 卡通化设置
    setCartoonizeEnabled(v: boolean) {
      this.cartoonizeEnabled = v
      saveImageProcessSettings({ cartoonizeEnabled: v })
    },
    setCartoonLevels(v: number) {
      this.cartoonLevels = v
      saveImageProcessSettings({ cartoonLevels: v })
    },
    setCartoonEdge(v: number) {
      this.cartoonEdge = v
      saveImageProcessSettings({ cartoonEdge: v })
    },
    setCartoonBlurPasses(v: number) {
      this.cartoonBlurPasses = v
      saveImageProcessSettings({ cartoonBlurPasses: v })
    },
    setCartoonProcessMode(v: 'frontend'|'backend') {
      this.cartoonProcessMode = v
      saveImageProcessSettings({ cartoonProcessMode: v })
    },
    setCartoonBackendMode(v: 'animegan_v2' | 'wbc' | 'cv_stylize' | 'bilateral') {
      this.cartoonBackendMode = v
      saveImageProcessSettings({ cartoonBackendMode: v })
    },
    setCartoonBackendStyle(
      v:
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
    ) {
      this.cartoonBackendStyle = v
      saveImageProcessSettings({ cartoonBackendStyle: v })
    },
    // Anime frame overlay setters
    setAnimeFrameEnabled(v: boolean) {
      this.animeFrameEnabled = v
      saveImageProcessSettings({ animeFrameEnabled: v })
    },
    setAnimeFrameStyle(v: 'panel' | 'glow' | 'film') {
      this.animeFrameStyle = v
      saveImageProcessSettings({ animeFrameStyle: v })
    },

    setUploadedFile(type: keyof DigitalHumanState['uploadedFiles'], file: File | null) {
      this.uploadedFiles[type] = file
    },

    addTrainingAudio(file: File) {
      this.trainingData.audioFiles.push(file)
    },

    removeTrainingAudio(index: number) {
      this.trainingData.audioFiles.splice(index, 1)
    },

    setTrainingLabel(filename: string, text: string) {
      this.trainingData.labels[filename] = text
    },

    setRefAudio(file: File | null) {
      this.trainingData.refAudio = file
    },

    setRefText(text: string) {
      this.trainingData.refText = text
    },

    setPersonConfigured(configured: boolean) {
      this.isPersonConfigured = configured
    },

    setVoiceConfigured(configured: boolean) {
      this.isVoiceConfigured = configured
    },

    async uploadImage(user: string, imageFile: File) {
      this.isProcessing = true
      this.currentTask = '上传图片'
      try {
        const reader = new FileReader()
        reader.readAsDataURL(imageFile)
        let base64Image = await new Promise<string>((resolve) => {
          reader.onload = () => resolve(reader.result as string)
        })

        // 若启用卡通化或像素化，先在前端处理
        if (this.cartoonizeEnabled && this.cartoonProcessMode === 'backend') {
          // 按所选后端模式调用（animegan_v2 / wbc / cv_stylize / bilateral）
          try {
            const raw = base64Image.split(',')[1]
            const mode = this.cartoonBackendMode || 'animegan_v2'
            const style = this.cartoonBackendStyle
            const params: any = { max_side: 1600 }
            console.info('[cartoonize-backend] request begin', { mode, style, params })
            const res = await digitalHumanApi.cartoonizeImage({
              User: user,
              Img: raw,
              Mode: mode as any,
              // 仅 animegan 需要 Style；其他模式忽略
              Style: mode === 'animegan_v2' ? (style as any) : undefined,
              Params: params,
            })
            if (res.success && res.img) {
              console.info('[cartoonize-backend] success', {
                mode_used: res.mode_used,
                style_used: res.style_used,
                debug: res.debug,
              })
              base64Image = 'data:image/png;base64,' + res.img
            } else {
              console.error('[cartoonize-backend] failed (no fallback)', res.error)
              // 失败后不再处理，保留原始图像
            }
          } catch (e) {
            console.error('[cartoonize-backend] exception (no fallback)', e)
          }
        } else if (this.pixelateEnabled) {
          try {
            const { pixelateImage } = await import('@/utils/pixelate')
            base64Image = await pixelateImage(base64Image, {
              blockSize: this.pixelBlockSize,
              levels: 16,
              dithering: true,
            })
          } catch (e) {
            // 处理失败则退回原图，避免中断上传
            console.warn('[pixelate] 像素化失败，退回原图：', e)
          }
        }

        const base64Data = base64Image.split(',')[1]
        const result = await digitalHumanApi.sendImage({
          User: user,
          Img: base64Data,
        })

        // 上传成功后标记数字人已配置
        if (result) {
          this.isPersonConfigured = true
        }

        return result
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    async sendPPTRemakes(user: string, remakes: string | Record<string, string>) {
      this.isProcessing = true
      this.currentTask = '发送PPT备注'
      try {
        return await digitalHumanApi.sendPPTRemakes({
          User: user,
          PPT_Remakes: remakes,
        })
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    async sendConfig(user: string) {
      this.isProcessing = true
      this.currentTask = '配置参数'
      try {
        const vitsConfig = {
          model_index: this.config.modelIndex,
        }
        const sadTalkerConfig = {
          enhancer: this.config.enhancer,
          expression_scale: this.config.expressionScale,
        }

        return await digitalHumanApi.sendConfig({
          User: user,
          VITS_Config: vitsConfig,
          SadTalker_Config: sadTalkerConfig,
        })
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    // 旧流程（保留兼容）
    async generateDigitalHuman(user: string) {
      this.isProcessing = true
      this.currentTask = '生成数字人视频'
      try {
        // Step 1: 触发推理
        this.currentTask = '触发推理任务'
        const startInference = await digitalHumanApi.getInference({ User: user })
        if (!startInference) throw new Error('阶段[触发推理]：后端返回 Failed')

        // Step 2: 等待推理完成
        this.currentTask = '等待音视频推理'
        const inferenceDone = await digitalHumanApi.waitForTask(user, 'Audio_Video_Inference')
        if (!inferenceDone) throw new Error('阶段[等待推理]：任务未成功完成 (Failed/超时)')

        // Step 2.5: 生成PPT视频 (调用后端API)
        this.currentTask = '生成PPT视频'
        const pptVideoGenerated = await this.generatePptVideo(user)
        if (!pptVideoGenerated) throw new Error('阶段[PPT视频生成]：生成失败')

        // Step 3: 合并 PPT + 视频
        this.currentTask = '请求合并视频'
        const mergeTrigger = await digitalHumanApi.pptVideoMerge({ User: user })
        if (!mergeTrigger) throw new Error('阶段[触发合并]：后端返回 Failed')

        // Step 4: 等待合并完成
        this.currentTask = '等待视频合并'
        const mergeDone = await digitalHumanApi.waitForTask(user, 'Video_Merge')
        if (!mergeDone) throw new Error('阶段[等待合并]：任务未成功完成 (Failed/超时)')

        this.currentTask = '完成'
        return true
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    // 新流程：完整分支逻辑（无动作/有动作 + 全插入/不插入/部分插入 + 是否使用模型音频）
    async generateDigitalHumanAdvanced(user: string) {
      const { digitalMotion, intoDigitalOperation, useModelAudio, selectedSlides } =
        this.videoOptions

      this.isProcessing = true
      try {
        // 1) 推理阶段（根据分支调用不同后端）
        this.currentTask = '启动推理'
        let inferenceOk = false
        if (digitalMotion === 1) {
          // SadTalker 分支
          inferenceOk = useModelAudio
            ? await digitalHumanApi.getInferenceVITS_Sadtalker({ User: user })
            : await digitalHumanApi.getInferenceUser_Sadtalker({ User: user })
        } else {
          // Wav2Lip 分支
          inferenceOk = useModelAudio
            ? await digitalHumanApi.getInferenceVITS_Wav2Lip({ User: user })
            : await digitalHumanApi.getInferenceUser_Wav2Lip({ User: user })
        }
        if (!inferenceOk) throw new Error('推理启动失败')

        // 2) 等待推理完成（C# 里返回的是任务名再轮询；这里沿用固定任务名）
        this.currentTask = '等待推理完成'
        const inferenceDone = await digitalHumanApi.waitForTask(user, 'Audio_Video_Inference')
        if (!inferenceDone) throw new Error('推理超时或失败')

        // 3) 获取音频时长（根据是否使用模型音频）
        this.currentTask = '获取音频时长'
        const wavTime = useModelAudio
          ? await digitalHumanApi.receiveWavTime({ User: user })
          : await digitalHumanApi.receiveUserWavTime({ User: user })
        if (!wavTime) throw new Error('获取音频时长失败')

        // 4) 生成PPT视频
        this.currentTask = '生成PPT视频'
        const pptOk = await this.generatePptVideo(user)
        if (!pptOk) throw new Error('PPT视频生成失败')

        // 5) 合并方式：全部/不插入/部分
        if (intoDigitalOperation === 3) {
          // 部分插入：需要提前发送插入页
          if (!selectedSlides || selectedSlides.length === 0) {
            throw new Error('请选择需要插入数字人的页码')
          }
          this.currentTask = '发送插入页信息'
          const ok = await digitalHumanApi.sendPeopleLocation({ User: user, Pages: selectedSlides })
          if (!ok) throw new Error('发送插入页失败')
        }

        // 6) 触发合并
        this.currentTask = '触发视频合并'
        let mergeOk = false
        if (intoDigitalOperation === 1) {
          mergeOk = await digitalHumanApi.pptVideoMerge({ User: user })
        } else if (intoDigitalOperation === 2) {
          mergeOk = await digitalHumanApi.pptVideoMergeNoInto({ User: user })
        } else {
          mergeOk = await digitalHumanApi.pptVideoMergeSelectInto({ User: user })
        }
        if (!mergeOk) throw new Error('触发合并失败')

        // 7) 等待合并完成
        this.currentTask = '等待视频合并'
        const mergeDone = await digitalHumanApi.waitForTask(user, 'Video_Merge')
        if (!mergeDone) throw new Error('视频合并失败或超时')

        this.currentTask = '完成'
        return true
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    async downloadVideo(user: string): Promise<Blob> {
      return await digitalHumanApi.pullVideoMerge({ User: user })
    },

    // 透传：获取VITS推理（单次）
    async getInferenceVITS(user: string): Promise<boolean> {
      return await digitalHumanApi.getInferenceVITS({ User: user })
    },

    // 透传：拉取VITS音频
    async pullVITSAudio(user: string): Promise<Blob> {
      return await digitalHumanApi.pullVITSAudio({ User: user })
    },

    async trainVoiceModel(user: string) {
      this.isProcessing = true
      this.currentTask = '训练语音模型'
      try {
        // Upload training audio files
        for (const file of this.trainingData.audioFiles) {
          const success = await digitalHumanApi.sendTrainAudio(
            { User: user, Audio_Name: file.name },
            file,
          )
          if (!success) throw new Error(`上传训练音频失败: ${file.name}`)
        }

        // Upload reference audio and text
        if (this.trainingData.refAudio) {
          const success = await digitalHumanApi.sendRefWavAndText(
            { User: user, Ref_Text: this.trainingData.refText },
            this.trainingData.refAudio,
          )
          if (!success) throw new Error('上传参考音频失败')
        }

        // Start training
        const trainSuccess = await digitalHumanApi.trainVITSModel({
          User: user,
          Label: JSON.stringify(this.trainingData.labels),
        })
        if (!trainSuccess) throw new Error('训练失败')

        // Wait for training completion
        await digitalHumanApi.waitForTask(user, 'VITS_Train')

        return true
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    // 新增：语音模型选择方法
    async selectVITSModel(user: string, modelIndex: string) {
      this.isProcessing = true
      this.currentTask = '选择预训练语音模型'
      try {
        const result = await digitalHumanApi.selectVITSModel({
          User: user,
          Index: modelIndex,
        })

        // 选择成功后标记语音模型已配置
        if (result) {
          this.isVoiceConfigured = true
        }
        return result
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    // 新增：选择训练得到的VITS模型（如果存在）
    async selectTrainVITSModel(user: string) {
      this.isProcessing = true
      this.currentTask = '选择训练模型'
      try {
        const result = await digitalHumanApi.selectTrainVITSModel({ User: user })
        if (result) {
          this.isVoiceConfigured = true
        }
        return result
      } finally {
        this.isProcessing = false
        this.currentTask = null
      }
    },

    // 生成PPT视频（封装为独立方法，避免重复代码）
    // 注意：该方法不修改 isProcessing，仅返回成功与否，外层流程负责设置进度与状态
    async generatePptVideo(user: string): Promise<boolean> {
      const res = await digitalHumanApi.generatePPTVideo({ User: user })
      // 某些后端实现可能是同步完成，此时直接以返回值为准
      if (!res.success) return false
      // 如果后端为异步任务，也可在此追加等待逻辑；暂不等待以兼容两种实现
      return true
    },

    // 获取训练模型名称（可用于UI展示）
    async getTrainVitsModelName(user: string): Promise<string | null> {
      return await digitalHumanApi.getTrainVitsModelName({ User: user })
    },
  },
})
