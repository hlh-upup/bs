<template>
  <div class="voice-trainer">
    <div class="section-header">
      <h2>语音训练</h2>
      <p>训练专属语音模型，让数字人拥有您的声音</p>
    </div>

    <div class="training-workflow">
      <!-- Step 1: Upload Training Audio -->
      <div class="training-step" :class="{ active: currentStep >= 1, completed: currentStep > 1 }">
        <div class="step-header">
          <div class="step-number">1</div>
          <div class="step-content">
            <h3>上传训练音频</h3>
            <p>上传3-10秒的WAV音频文件，系统将自动进行音频分段和预处理</p>
          </div>
        </div>

        <div class="step-content">
          <div class="upload-area" @drop="handleAudioDrop" @dragover.prevent>
            <input
              ref="audioInput"
              type="file"
              accept="audio/wav"
              multiple
              @change="handleAudioSelect"
              style="display: none"
            />

            <div v-if="trainingData.audioFiles.length === 0" class="upload-placeholder" @click="$refs.audioInput.click()">
              <div class="upload-icon">🎵</div>
              <p>点击或拖拽上传音频文件</p>
              <span class="upload-hint">支持WAV格式，建议3-10秒/段，总时长建议2-10分钟</span>
            </div>

            <div v-else class="audio-list">
              <div class="audio-list-header">
                <span>已上传音频文件 ({{ trainingData.audioFiles.length }}段)</span>
                <button class="add-more-btn" @click="$refs.audioInput.click()">+ 添加更多</button>
              </div>
              <div v-for="(audio, index) in trainingData.audioFiles" :key="index" class="audio-item">
                <div class="audio-info">
                  <div class="audio-icon">🎵</div>
                  <div class="audio-details">
                    <span class="audio-name">{{ audio.name }}</span>
                    <span class="audio-size">{{ formatFileSize(audio.size) }}</span>
                    <span class="audio-duration" v-if="audioDurations[audio.name]">{{ audioDurations[audio.name] }}s</span>
                  </div>
                </div>
                <div class="audio-actions">
                  <button class="preview-btn" @click="previewAudio(audio)" title="预览音频">▶</button>
                  <button class="segment-btn" @click="segmentAudio(audio)" title="智能分段">✂</button>
                  <button class="remove-btn" @click="removeAudio(index)" title="删除">×</button>
                </div>
              </div>
            </div>
          </div>

          <!-- Audio Segmentation Results -->
          <div v-if="segmentedAudios.length > 0" class="segmented-audios">
            <h4>智能分段结果 ({{ segmentedAudios.length }}段)</h4>
            <div class="segment-list">
              <div v-for="(segment, index) in segmentedAudios" :key="index" class="segment-item">
                <div class="segment-info">
                  <span class="segment-name">{{ segment.name }}</span>
                  <span class="segment-duration">{{ segment.duration }}s</span>
                </div>
                <button class="remove-btn" @click="removeSegment(index)">×</button>
              </div>
            </div>
          </div>

          <div v-if="trainingData.audioFiles.length > 0" class="audio-labels">
            <h4>音频文本标注</h4>
            <p class="labels-hint">请为每个音频片段标注对应的文字内容，这是训练高质量语音模型的关键</p>
            <div v-for="audio in trainingData.audioFiles" :key="audio.name" class="label-item">
              <div class="label-header">
                <span class="audio-name">{{ audio.name }}</span>
                <span class="audio-duration" v-if="audioDurations[audio.name]">{{ audioDurations[audio.name] }}s</span>
              </div>
              <textarea
                v-model="trainingData.labels[audio.name]"
                :placeholder="'请输入此段音频对应的文字内容...'"
                class="label-textarea"
                rows="2"
              />
              <div class="label-actions">
                <button class="auto-label-btn" @click="autoLabelAudio(audio)">智能标注</button>
                <button class="preview-label-btn" @click="previewLabel(audio)">预览</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 2: Reference Audio -->
      <div class="training-step" :class="{ active: currentStep >= 2 }">
        <div class="step-header">
          <div class="step-number">2</div>
          <div class="step-content">
            <h3>上传参考音频</h3>
            <p>用于目标语音的参考音频</p>
          </div>
        </div>

        <div class="step-content">
          <div class="upload-area" @drop="handleRefAudioDrop" @dragover.prevent>
            <input
              ref="refAudioInput"
              type="file"
              accept="audio/wav"
              @change="handleRefAudioSelect"
              style="display: none"
            />

            <div v-if="!trainingData.refAudio" class="upload-placeholder" @click="$refs.refAudioInput.click()">
              <div class="upload-icon">🎤</div>
              <p>上传参考音频</p>
              <span class="upload-hint">WAV格式，清晰人声</span>
            </div>

            <div v-else class="audio-item">
              <div class="audio-info">
                <div class="audio-icon">🎤</div>
                <div class="audio-details">
                  <span class="audio-name">{{ trainingData.refAudio.name }}</span>
                  <span class="audio-size">{{ formatFileSize(trainingData.refAudio.size) }}</span>
                </div>
              </div>
              <button class="remove-btn" @click="removeRefAudio">×</button>
            </div>
          </div>

          <div v-if="trainingData.refAudio" class="ref-text-section">
            <label>参考文字内容</label>
            <textarea
              v-model="trainingData.refText"
              placeholder="请输入参考音频对应的文字内容..."
              rows="3"
              class="text-input"
            />
          </div>
        </div>
      </div>

      <!-- Step 3: Training Configuration -->
      <div class="training-step" :class="{ active: currentStep >= 3, completed: currentStep > 3 }">
        <div class="step-header">
          <div class="step-number">3</div>
          <div class="step-content">
            <h3>高级训练配置</h3>
            <p>自定义VITS模型训练参数，优化语音质量</p>
          </div>
        </div>

        <div class="step-content">
          <div class="config-tabs">
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'basic' }"
              @click="activeTab = 'basic'"
            >
              基础配置
            </button>
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'advanced' }"
              @click="activeTab = 'advanced'"
            >
              高级配置
            </button>
            <button
              class="tab-btn"
              :class="{ active: activeTab === 'quality' }"
              @click="activeTab = 'quality'"
            >
              质量优化
            </button>
          </div>

          <div class="config-content">
            <!-- Basic Configuration -->
            <div v-if="activeTab === 'basic'" class="config-grid">
              <div class="config-item">
                <label>训练轮数 <span class="help-tooltip" title="训练轮数越多，模型质量越高，但训练时间更长">?</span></label>
                <select v-model="trainingConfig.epochs">
                  <option value="10">10轮（快速预览）</option>
                  <option value="20">20轮（标准质量）</option>
                  <option value="50">50轮（高质量）</option>
                  <option value="100">100轮（专业级）</option>
                </select>
              </div>

              <div class="config-item">
                <label>批处理大小</label>
                <select v-model="trainingConfig.batchSize">
                  <option value="16">16（推荐）</option>
                  <option value="32">32（更快）</option>
                  <option value="8">8（显存友好）</option>
                </select>
              </div>

              <div class="config-item">
                <label>保存频率</label>
                <select v-model="trainingConfig.saveInterval">
                  <option value="5">每5轮保存</option>
                  <option value="10">每10轮保存</option>
                  <option value="20">每20轮保存</option>
                </select>
              </div>
            </div>

            <!-- Advanced Configuration -->
            <div v-if="activeTab === 'advanced'" class="config-grid">
              <div class="config-item">
                <label>学习率</label>
                <div class="lr-input-group">
                  <input
                    type="number"
                    v-model.number="trainingConfig.learningRate"
                    min="0.00001"
                    max="0.01"
                    step="0.00001"
                    class="number-input"
                  />
                  <select v-model="trainingConfig.lrScheduler" class="lr-scheduler">
                    <option value="exponential">指数衰减</option>
                    <option value="linear">线性衰减</option>
                    <option value="step">阶梯衰减</option>
                  </select>
                </div>
              </div>

              <div class="config-item">
                <label>优化器</label>
                <select v-model="trainingConfig.optimizer">
                  <option value="AdamW">AdamW（推荐）</option>
                  <option value="Adam">Adam</option>
                  <option value="SGD">SGD</option>
                </select>
              </div>

              <div class="config-item">
                <label>权重衰减</label>
                <input
                  type="number"
                  v-model.number="trainingConfig.weightDecay"
                  min="0"
                  max="0.1"
                  step="0.0001"
                  class="number-input"
                />
              </div>

              <div class="config-item">
                <label>梯度裁剪</label>
                <input
                  type="number"
                  v-model.number="trainingConfig.gradClip"
                  min="0"
                  max="10"
                  step="0.1"
                  class="number-input"
                />
              </div>
            </div>

            <!-- Quality Optimization -->
            <div v-if="activeTab === 'quality'" class="config-grid">
              <div class="config-item">
                <label>音频采样率</label>
                <select v-model="trainingConfig.samplingRate">
                  <option value="22050">22.05 kHz（标准）</option>
                  <option value="24000">24 kHz（高质量）</option>
                  <option value="44100">44.1 kHz（超高质量）</option>
                </select>
              </div>

              <div class="config-item">
                <label>FFT滤波器大小</label>
                <select v-model="trainingConfig.nFFT">
                  <option value="1024">1024（标准）</option>
                  <option value="2048">2048（高质量）</option>
                  <option value="512">512（快速）</option>
                </select>
              </div>

              <div class="config-item">
                <label>说话人嵌入维度</label>
                <select v-model="trainingConfig.speakerEmbedding">
                  <option value="256">256维（标准）</option>
                  <option value="512">512维（高质量）</option>
                  <option value="128">128维（轻量）</option>
                </select>
              </div>

              <div class="config-item">
                <label>数据增强</label>
                <div class="checkbox-group">
                  <label class="checkbox-item">
                    <input type="checkbox" v-model="trainingConfig.usePitchShift" />
                    音调变化
                  </label>
                  <label class="checkbox-item">
                    <input type="checkbox" v-model="trainingConfig.useTimeStretch" />
                    时间拉伸
                  </label>
                  <label class="checkbox-item">
                    <input type="checkbox" v-model="trainingConfig.useNoiseInjection" />
                    噪声注入
                  </label>
                </div>
              </div>
            </div>
          </div>

          <!-- Configuration Summary -->
          <div class="config-summary">
            <h4>配置摘要</h4>
            <div class="summary-grid">
              <div class="summary-item">
                <span class="summary-label">预计训练时间:</span>
                <span class="summary-value">{{ estimatedTrainingTime }}分钟</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">显存需求:</span>
                <span class="summary-value">{{ estimatedMemoryUsage }}GB</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">模型质量:</span>
                <span class="summary-value">{{ estimatedQuality }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 4: Start Training -->
      <div class="training-step" :class="{ active: currentStep >= 4 }">
        <div class="step-header">
          <div class="step-number">4</div>
          <div class="step-content">
            <h3>开始训练</h3>
            <p>启动语音模型训练</p>
          </div>
        </div>

        <div class="step-content">
          <div class="training-info">
            <div class="info-item">
              <span class="label">音频数量:</span>
              <span class="value">{{ trainingData.audioFiles.length }} 段</span>
            </div>
            <div class="info-item">
              <span class="label">预计时间:</span>
              <span class="value">{{ estimatedTrainingTime }} 分钟</span>
            </div>
          </div>

          <button
            class="train-button"
            @click="startTraining"
            :disabled="!canTrain || isTraining"
          >
            {{ isTraining ? '训练中...' : '开始训练' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Training Progress -->
    <div v-if="isTraining" class="training-progress">
      <div class="progress-header">
        <h3>训练进行中...</h3>
        <div class="progress-stats">
          <span class="current-epoch">第 {{ currentEpoch }}/{{ totalEpochs }} 轮</span>
          <span class="elapsed-time">已用时: {{ formatElapsedTime(elapsedTime) }}</span>
        </div>
      </div>

      <div class="progress-main">
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: trainingProgress + '%' }"></div>
          </div>
          <span class="progress-percentage">{{ trainingProgress.toFixed(1) }}%</span>
        </div>

        <div class="current-step">{{ trainingStatus }}</div>
      </div>

      <!-- Detailed Progress Information -->
      <div class="progress-details">
        <div class="detail-card">
          <h4>训练指标</h4>
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">损失值</span>
              <span class="metric-value">{{ trainingMetrics.loss?.toFixed(4) || 'N/A' }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">学习率</span>
              <span class="metric-value">{{ trainingMetrics.learningRate?.toFixed(6) || 'N/A' }}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">梯度范数</span>
              <span class="metric-value">{{ trainingMetrics.gradNorm?.toFixed(4) || 'N/A' }}</span>
            </div>
          </div>
        </div>

        <div class="detail-card">
          <h4>实时日志</h4>
          <div class="log-container">
            <div v-for="(log, index) in trainingLogs" :key="index" class="log-entry">
              <span class="log-timestamp">{{ log.timestamp }}</span>
              <span class="log-message">{{ log.message }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Training Actions -->
      <div class="training-actions">
        <button class="pause-btn" @click="pauseTraining" :disabled="isPaused">
          {{ isPaused ? '继续训练' : '暂停训练' }}
        </button>
        <button class="stop-btn" @click="stopTraining">停止训练</button>
        <button class="export-btn" @click="exportTrainingLogs">导出日志</button>
      </div>
    </div>

    <!-- Training Results -->
    <div v-if="trainingCompleted" class="training-results">
      <h3>✅ 训练完成！</h3>
      <div class="results-info">
        <p>您的专属语音模型已训练完成</p>
        <button class="test-button" @click="testModel">测试模型</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'

const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

const currentStep = ref(1)
const isTraining = ref(false)
const trainingCompleted = ref(false)
const isPaused = ref(false)
const trainingProgress = ref(0)
const trainingStatus = ref('')
const audioInput = ref<HTMLInputElement>()
const refAudioInput = ref<HTMLInputElement>()
const activeTab = ref('basic')

// Enhanced training data
const audioDurations = ref<Record<string, number>>({})
const segmentedAudios = ref<Array<{ name: string; duration: number; blob: Blob }>>([])
const currentEpoch = ref(0)
const totalEpochs = ref(0)
const elapsedTime = ref(0)
const startTime = ref(0)

// Training metrics and logs
const trainingMetrics = ref({
  loss: 0,
  learningRate: 0,
  gradNorm: 0
})

const trainingLogs = ref<Array<{ timestamp: string; message: string }>>([])
const progressInterval = ref<number | null>(null)
const metricsInterval = ref<number | null>(null)

const trainingData = computed(() => digitalHumanStore.trainingData)

const trainingConfig = ref({
  // Basic config
  epochs: 20,
  batchSize: 16,
  saveInterval: 10,
  learningRate: 0.0002,

  // Advanced config
  optimizer: 'AdamW',
  weightDecay: 0.01,
  gradClip: 1.0,
  lrScheduler: 'exponential',

  // Quality config
  samplingRate: 22050,
  nFFT: 1024,
  speakerEmbedding: 256,

  // Data augmentation
  usePitchShift: false,
  useTimeStretch: false,
  useNoiseInjection: false
})

const canTrain = computed(() => {
  return (
    trainingData.value.audioFiles.length > 0 &&
    trainingData.value.refAudio &&
    trainingData.value.refText.trim() &&
    Object.keys(trainingData.value.labels).length === trainingData.value.audioFiles.length &&
    allLabelsComplete.value
  )
})

const allLabelsComplete = computed(() => {
  return trainingData.value.audioFiles.every(audio =>
    trainingData.value.labels[audio.name]?.trim().length > 0
  )
})

const estimatedTrainingTime = computed(() => {
  const baseTime = trainingData.value.audioFiles.length * trainingConfig.value.epochs * 0.5
  const complexityFactor = trainingConfig.value.samplingRate / 22050
  const batchSizeFactor = 16 / trainingConfig.value.batchSize
  return Math.round(baseTime * complexityFactor * batchSizeFactor)
})

const estimatedMemoryUsage = computed(() => {
  const baseMemory = 4 // GB base
  const batchSizeFactor = trainingConfig.value.batchSize / 16
  const fftFactor = trainingConfig.value.nFFT / 1024
  const embeddingFactor = trainingConfig.value.speakerEmbedding / 256
  return (baseMemory * batchSizeFactor * fftFactor * embeddingFactor).toFixed(1)
})

const estimatedQuality = computed(() => {
  const epochs = trainingConfig.value.epochs
  const samplingRate = trainingConfig.value.samplingRate
  const hasDataAugmentation = trainingConfig.value.usePitchShift ||
                               trainingConfig.value.useTimeStretch ||
                               trainingConfig.value.useNoiseInjection

  if (epochs >= 100 && samplingRate >= 24000 && hasDataAugmentation) {
    return '专业级'
  } else if (epochs >= 50 && samplingRate >= 22050) {
    return '高质量'
  } else if (epochs >= 20) {
    return '标准质量'
  } else {
    return '快速预览'
  }
})

const handleAudioSelect = (event: Event) => {
  const files = (event.target as HTMLInputElement).files
  if (files) {
    for (let i = 0; i < files.length; i++) {
      if (files[i].type === 'audio/wav') {
        digitalHumanStore.addTrainingAudio(files[i])
        getAudioDuration(files[i])
      }
    }
    if (currentStep.value === 1) currentStep.value = 2
  }
}

const handleAudioDrop = (event: DragEvent) => {
  const files = event.dataTransfer?.files
  if (files) {
    for (let i = 0; i < files.length; i++) {
      if (files[i].type === 'audio/wav') {
        digitalHumanStore.addTrainingAudio(files[i])
        getAudioDuration(files[i])
      }
    }
    if (currentStep.value === 1) currentStep.value = 2
  }
}

const getAudioDuration = (file: File) => {
  const audio = new Audio()
  const url = URL.createObjectURL(file)
  audio.addEventListener('loadedmetadata', () => {
    audioDurations.value[file.name] = Math.round(audio.duration)
    URL.revokeObjectURL(url)
  })
  audio.src = url
}

const previewAudio = (audio: File) => {
  const audioUrl = URL.createObjectURL(audio)
  const audioElement = new Audio(audioUrl)
  audioElement.play()
  setTimeout(() => URL.revokeObjectURL(audioUrl), 5000)
}

const segmentAudio = async (audio: File) => {
  // Simulate audio segmentation (in real implementation, this would call backend API)
  const segments = Math.ceil(audioDurations.value[audio.name] / 5) // 5-second segments
  for (let i = 0; i < segments; i++) {
    segmentedAudios.value.push({
      name: `${audio.name}_segment_${i + 1}.wav`,
      duration: Math.min(5, audioDurations.value[audio.name] - i * 5),
      blob: audio // In real implementation, this would be the actual segmented blob
    })
  }
  addTrainingLog(`音频 ${audio.name} 已分为 ${segments} 段`)
}

const removeSegment = (index: number) => {
  segmentedAudios.value.splice(index, 1)
}

import { digitalHumanApi } from '@/services/api'

const autoLabelAudio = async (audio: File) => {
  // 调用后端接口进行语音识别
  if (!authStore.currentUser) return
  try {
    const req = { User: authStore.currentUser, Ref_Text: '' }
    const resp = await digitalHumanApi.sendRefWavAndText(req, audio)
    // 这里假设 resp 为 true，实际可根据后端返回内容调整
    // 若后端返回识别文本，需在 resp 中取用并赋值
    addTrainingLog(`自动标注完成: ${audio.name}`)
    // 可选：刷新 trainingData.value.labels[audio.name] = 后端返回的识别文本
  } catch (e) {
    addTrainingLog(`自动标注失败: ${audio.name}`)
  }
}

const previewLabel = (audio: File) => {
  const text = trainingData.value.labels[audio.name]
  if (text) {
    // Simulate text-to-speech preview
    addTrainingLog(`预览文本: "${text}"`)
  }
}

const removeAudio = (index: number) => {
  digitalHumanStore.removeTrainingAudio(index)
  // Clean up duration entry
  const audioName = trainingData.value.audioFiles[index]?.name
  if (audioName && audioDurations.value[audioName]) {
    delete audioDurations.value[audioName]
  }
}

const handleRefAudioSelect = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file && authStore.currentUser) {
    digitalHumanStore.setRefAudio(file)
    currentStep.value = 3
    // 自动调用后端识别接口
    try {
      const req = { User: authStore.currentUser, Ref_Text: trainingData.value.refText || '' }
      await digitalHumanApi.sendRefWavAndText(req, file)
      // 可选：弹窗提示识别成功
    } catch (e) {
      addTrainingLog('参考音频智能标注失败')
    }
  }
}

const handleRefAudioDrop = (event: DragEvent) => {
  const file = event.dataTransfer?.files[0]
  if (file && file.type === 'audio/wav') {
    digitalHumanStore.setRefAudio(file)
    currentStep.value = 3
  }
}

const removeRefAudio = () => {
  digitalHumanStore.setRefAudio(null)
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const startTraining = async () => {
  if (!canTrain.value || !authStore.currentUser) return

  isTraining.value = true
  isPaused.value = false
  trainingCompleted.value = false
  trainingProgress.value = 0
  currentEpoch.value = 0
  totalEpochs.value = trainingConfig.value.epochs
  startTime.value = Date.now()
  trainingLogs.value = []

  try {
    addTrainingLog('开始训练语音模型...')
    trainingStatus.value = '初始化训练环境...'

    // Step 1: Upload training audio files
    trainingStatus.value = '上传训练音频文件...'
    for (const file of trainingData.value.audioFiles) {
      const success = await digitalHumanStore.trainVoiceModel(authStore.currentUser)
      if (!success) throw new Error(`上传训练音频失败: ${file.name}`)
      addTrainingLog(`上传完成: ${file.name}`)
    }

    // Step 2: Upload reference audio and text
    if (trainingData.value.refAudio) {
      trainingStatus.value = '上传参考音频和文本...'
      const success = await digitalHumanStore.trainVoiceModel(authStore.currentUser)
      if (!success) throw new Error('上传参考音频失败')
      addTrainingLog('参考音频上传完成')
    }

    // Step 3: Start training with progress monitoring
    trainingStatus.value = '开始模型训练...'
    startProgressMonitoring()

    // Start actual training
    const trainSuccess = await digitalHumanStore.trainVoiceModel(authStore.currentUser)
    if (trainSuccess) {
      addTrainingLog('训练任务提交成功')
      // Monitor training progress
      await monitorTrainingProgress()
      trainingCompleted.value = true
      addTrainingLog('🎉 训练完成！')
    }

  } catch (error) {
    console.error('训练失败:', error)
    addTrainingLog(`❌ 训练失败: ${error}`)
    alert('训练失败，请重试')
  } finally {
    stopProgressMonitoring()
    isTraining.value = false
  }
}

const startProgressMonitoring = () => {
  // Update elapsed time every second
  progressInterval.value = window.setInterval(() => {
    if (!isPaused.value) {
      elapsedTime.value = Math.floor((Date.now() - startTime.value) / 1000)
    }
  }, 1000)

  // Update training metrics every 5 seconds
  metricsInterval.value = window.setInterval(() => {
    if (!isPaused.value && isTraining.value) {
      updateTrainingMetrics()
    }
  }, 5000)
}

const stopProgressMonitoring = () => {
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
    progressInterval.value = null
  }
  if (metricsInterval.value) {
    clearInterval(metricsInterval.value)
    metricsInterval.value = null
  }
}

const updateTrainingMetrics = async () => {
  // Simulate metrics update (in real implementation, this would fetch from backend)
  const progress = Math.min(100, (currentEpoch.value / totalEpochs.value) * 100)
  trainingProgress.value = progress

  // Simulate training metrics
  trainingMetrics.value = {
    loss: Math.random() * 2 + 0.1, // Random loss between 0.1 and 2.1
    learningRate: trainingConfig.value.learningRate * Math.pow(0.99, currentEpoch.value),
    gradNorm: Math.random() * 5 + 0.5 // Random gradient norm
  }

  // Simulate epoch progress
  if (currentEpoch.value < totalEpochs.value) {
    currentEpoch.value += 1
    if (currentEpoch.value % 5 === 0) {
      addTrainingLog(`完成第 ${currentEpoch.value} 轮训练，损失: ${trainingMetrics.value.loss.toFixed(4)}`)
    }
  }
}

const monitorTrainingProgress = async () => {
  // Monitor training progress via backend API
  const maxWaitTime = trainingConfig.value.epochs * 60000 // 1 minute per epoch max
  const startTime = Date.now()

  while (Date.now() - startTime < maxWaitTime) {
    if (!isTraining.value) break

    try {
      // Simulate checking training status
      // In real implementation, this would call backend API to check training state
      await new Promise(resolve => setTimeout(resolve, 2000)) // Check every 2 seconds

      if (trainingProgress.value >= 100) {
        break
      }

    } catch (error) {
      console.error('检查训练进度失败:', error)
    }
  }
}

const pauseTraining = () => {
  isPaused.value = !isPaused.value
  addTrainingLog(isPaused.value ? '训练已暂停' : '训练已恢复')
}

const stopTraining = () => {
  isTraining.value = false
  isPaused.value = false
  stopProgressMonitoring()
  addTrainingLog('训练已停止')
}

const addTrainingLog = (message: string) => {
  const timestamp = new Date().toLocaleTimeString()
  trainingLogs.value.unshift({ timestamp, message })
  // Keep only last 50 logs
  if (trainingLogs.value.length > 50) {
    trainingLogs.value = trainingLogs.value.slice(0, 50)
  }
}

const exportTrainingLogs = () => {
  const logContent = trainingLogs.value
    .map(log => `[${log.timestamp}] ${log.message}`)
    .join('\n')

  const blob = new Blob([logContent], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `training_logs_${new Date().toISOString().slice(0, 10)}.txt`
  a.click()
  URL.revokeObjectURL(url)
}

const testModel = async () => {
  if (!authStore.currentUser) return

  try {
    addTrainingLog('开始测试训练好的语音模型...')
    const success = await digitalHumanStore.getInferenceVITS(authStore.currentUser)
    if (success) {
      addTrainingLog('语音推理成功')
      const audioBlob = await digitalHumanStore.pullVITSAudio(authStore.currentUser)
      const url = URL.createObjectURL(audioBlob)
      const audio = new Audio(url)
      audio.play()
      addTrainingLog('🔊 音频播放中...')
    }
  } catch (error) {
    console.error('测试失败:', error)
    addTrainingLog(`❌ 模型测试失败: ${error}`)
    alert('模型测试失败')
  }
}

const formatElapsedTime = (seconds: number) => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  } else {
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }
}

// Lifecycle hooks
onMounted(() => {
  // Cleanup on unmount
  onUnmounted(() => {
    stopProgressMonitoring()
  })
})

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey) {
    if (e.key === 'Enter' && !isTraining.value && canTrain.value) {
      startTraining()
    }
    if (e.key === 's' && isTraining.value) {
      e.preventDefault()
      stopTraining()
    }
  }
})
</script>

<style scoped>
.voice-trainer {
  max-width: 800px;
  margin: 0 auto;
}

.section-header {
  text-align: center;
  margin-bottom: 40px;
}

.section-header h2 {
  font-size: 28px;
  color: #333;
  margin-bottom: 10px;
}

.section-header p {
  color: #666;
  font-size: 16px;
}

.training-workflow {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.training-step {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  transition: all 0.3s;
}

.training-step.active {
  border-left: 4px solid #667eea;
}

.step-header {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.step-number {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
}

.training-step.active .step-number {
  background: #667eea;
  color: white;
}

.step-content h3 {
  margin: 0;
  color: #333;
}

.step-content p {
  margin: 5px 0 0 0;
  color: #666;
  font-size: 14px;
}

.upload-area {
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 30px;
  text-align: center;
  transition: all 0.3s;
  cursor: pointer;
}

.upload-area:hover {
  border-color: #667eea;
  background: #f8f9ff;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.upload-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.upload-hint {
  font-size: 14px;
  color: #666;
}

/* Enhanced Audio Management */
.audio-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  font-weight: 600;
  color: #333;
}

.add-more-btn {
  background: #667eea;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.3s;
}

.add-more-btn:hover {
  background: #5a6fd8;
}

.audio-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.preview-btn,
.segment-btn {
  background: #17a2b8;
  color: white;
  border: none;
  border-radius: 50%;
  width: 28px;
  height: 28px;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.3s;
}

.preview-btn:hover,
.segment-btn:hover {
  background: #138496;
}

.segment-btn {
  background: #6f42c1;
}

.segment-btn:hover {
  background: #5a3a8a;
}

.audio-duration {
  font-size: 12px;
  color: #888;
  background: #f0f0f0;
  padding: 2px 6px;
  border-radius: 4px;
}

/* Audio Segmentation */
.segmented-audios {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.segmented-audios h4 {
  margin-bottom: 12px;
  color: #333;
  font-size: 16px;
}

.segment-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.segment-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  background: white;
  border-radius: 6px;
  border-left: 3px solid #28a745;
}

.segment-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.segment-name {
  font-weight: 500;
  color: #333;
}

.segment-duration {
  font-size: 12px;
  color: #666;
  background: #e9ecef;
  padding: 2px 6px;
  border-radius: 4px;
}

/* Enhanced Labels */
.labels-hint {
  font-size: 14px;
  color: #666;
  margin-bottom: 15px;
}

.label-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.label-textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  resize: vertical;
  font-family: inherit;
  margin-bottom: 10px;
}

.label-actions {
  display: flex;
  gap: 10px;
}

.auto-label-btn,
.preview-label-btn {
  padding: 6px 12px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s;
}

.auto-label-btn {
  background: #28a745;
  color: white;
}

.auto-label-btn:hover {
  background: #218838;
}

.preview-label-btn {
  background: #6c757d;
  color: white;
}

.preview-label-btn:hover {
  background: #5a6268;
}

.audio-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.audio-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.audio-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.audio-icon {
  font-size: 24px;
}

.audio-details {
  display: flex;
  flex-direction: column;
}

.audio-name {
  font-weight: 600;
}

.audio-size {
  font-size: 12px;
  color: #666;
}

.remove-btn {
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  cursor: pointer;
  font-size: 14px;
}

.audio-labels {
  margin-top: 20px;
}

.audio-labels h4 {
  margin-bottom: 15px;
  color: #333;
}

.label-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.label-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.ref-text-section {
  margin-top: 20px;
}

.ref-text-section label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: #333;
}

.text-input {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  resize: vertical;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-item label {
  font-weight: 600;
  color: #333;
}

.number-input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

/* Training Step Completion */
.training-step.completed {
  background: #f8fff8;
}

.training-step.completed .step-number {
  background: #28a745;
  color: white;
}

/* Configuration Tabs */
.config-tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  border-bottom: 2px solid #e9ecef;
}

.tab-btn {
  padding: 12px 20px;
  background: none;
  border: none;
  font-size: 14px;
  font-weight: 600;
  color: #666;
  cursor: pointer;
  transition: all 0.3s;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
}

.tab-btn:hover {
  color: #333;
}

.tab-btn.active {
  color: #667eea;
  border-bottom-color: #667eea;
}

.config-content {
  min-height: 200px;
}

/* Enhanced Configuration */
.lr-input-group {
  display: flex;
  gap: 10px;
  align-items: center;
}

.lr-scheduler {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: white;
}

.checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 8px;
}

.checkbox-item {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 14px;
}

.checkbox-item input[type="checkbox"] {
  margin: 0;
}

/* Configuration Summary */
.config-summary {
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #17a2b8;
}

.config-summary h4 {
  margin: 0 0 12px 0;
  color: #333;
  font-size: 16px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.summary-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}

.summary-value {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

/* Help Tooltip */
.help-tooltip {
  display: inline-block;
  width: 16px;
  height: 16px;
  background: #6c757d;
  color: white;
  border-radius: 50%;
  font-size: 11px;
  line-height: 16px;
  text-align: center;
  cursor: help;
  margin-left: 4px;
}

.training-info {
  display: flex;
  gap: 30px;
  margin-bottom: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.info-item .label {
  font-size: 14px;
  color: #666;
}

.info-item .value {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.train-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 15px 40px;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.train-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.train-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.training-progress {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.progress-header h3 {
  margin: 0;
  color: #333;
}

.progress-stats {
  display: flex;
  gap: 20px;
  font-size: 14px;
  color: #666;
}

.current-epoch,
.elapsed-time {
  background: #f0f0f0;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: 500;
}

.progress-main {
  margin-bottom: 25px;
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 10px;
}

.progress-bar {
  flex: 1;
  height: 12px;
  background: #e0e0e0;
  border-radius: 6px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  transition: width 0.5s ease;
  border-radius: 6px;
}

.progress-percentage {
  font-size: 16px;
  font-weight: 600;
  color: #667eea;
  min-width: 50px;
  text-align: right;
}

.current-step {
  text-align: center;
  font-size: 14px;
  color: #666;
  font-style: italic;
}

/* Progress Details */
.progress-details {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 25px;
}

.detail-card {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
}

.detail-card h4 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 16px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.metric-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.metric-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  background: white;
  padding: 4px 8px;
  border-radius: 4px;
  min-width: 60px;
}

/* Log Container */
.log-container {
  max-height: 150px;
  overflow-y: auto;
  background: white;
  border-radius: 6px;
  padding: 10px;
  border: 1px solid #e9ecef;
}

.log-entry {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  margin-bottom: 6px;
  font-size: 12px;
  line-height: 1.4;
}

.log-timestamp {
  color: #888;
  white-space: nowrap;
  font-family: 'Courier New', monospace;
  min-width: 60px;
}

.log-message {
  color: #333;
  flex: 1;
}

/* Training Actions */
.training-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.pause-btn,
.stop-btn,
.export-btn {
  padding: 10px 20px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.pause-btn {
  background: #ffc107;
  color: #333;
}

.pause-btn:hover:not(:disabled) {
  background: #e0a800;
}

.pause-btn:disabled {
  background: #d6d6d6;
  color: #999;
  cursor: not-allowed;
}

.stop-btn {
  background: #dc3545;
  color: white;
}

.stop-btn:hover {
  background: #c82333;
}

.export-btn {
  background: #17a2b8;
  color: white;
}

.export-btn:hover {
  background: #138496;
}

.training-progress h3 {
  margin-bottom: 20px;
  color: #333;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 15px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
}

.progress-text {
  color: #666;
  font-size: 14px;
}

.training-results {
  background: #e8f5e8;
  border-radius: 12px;
  padding: 30px;
  text-align: center;
}

.training-results h3 {
  margin-bottom: 15px;
  color: #27ae60;
}

.test-button {
  background: #27ae60;
  color: white;
  border: none;
  padding: 12px 30px;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  margin-top: 15px;
}

.test-button:hover {
  background: #219a52;
}
</style>