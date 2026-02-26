<template>
  <div class="person-customizer">
    <div class="section-header">
      <h2>数字人定制</h2>
      <p>自定义数字人形象和声音，打造专属虚拟讲师</p>
    </div>

    <div class="customizer-content">
      <!-- Image Upload Section -->
      <div class="upload-section">
        <h3>真人照片</h3>
        <div class="upload-area" @drop="handleImageDrop" @dragover.prevent>
          <input
            ref="imageInput"
            type="file"
            accept="image/*"
            @change="handleImageSelect"
            style="display: none"
          />

          <div v-if="!imagePreview" class="upload-placeholder" @click="$refs.imageInput.click()">
            <div class="upload-icon">👤</div>
            <p>点击或拖拽上传真人照片</p>
            <span class="upload-hint">清晰正面人脸，光线充足</span>
          </div>

          <div v-else class="image-preview">
            <img :src="imagePreview" alt="真人照片" />
            <button class="remove-btn" @click="removeImage">×</button>
          </div>
        </div>

        <!-- 像素化开关与块大小滑条 -->
        <div class="pixelate-controls">
          <label class="switch-label">
            <input type="checkbox" v-model="pixelateEnabled" />
            <span>启用像素化</span>
          </label>
          <div v-if="pixelateEnabled" class="pixelate-slider">
            <label>像素块大小：<b>{{ pixelBlockSize }}</b></label>
            <input type="range" min="4" max="32" step="1" v-model.number="pixelBlockSize" @input="debouncedUpdatePreview" />
          </div>
        </div>
      </div>

      <!-- Voice Model Selection -->
      <div class="voice-section">
        <h3>声音模型选择</h3>

        <!-- Model Type Selection -->
        <div class="model-type-selector">
          <label class="radio-label">
            <input
              type="radio"
              v-model="modelType"
              value="pretrained"
              @change="onModelTypeChange"
            />
            <span>预训练模型</span>
          </label>
          <label class="radio-label">
            <input
              type="radio"
              v-model="modelType"
              value="trained"
              @change="onModelTypeChange"
            />
            <span>自训练模型</span>
          </label>
        </div>

        <!-- Pretrained Models -->
        <div v-if="modelType === 'pretrained'" class="model-grid">
          <div
            v-for="model in pretrainedModels"
            :key="model.id"
            class="model-card"
            :class="{ active: selectedPretrainedModel === model.id }"
            @click="selectPretrainedModel(model.id)"
          >
            <div class="model-icon">🎤</div>
            <h4>{{ model.name }}</h4>
            <p>{{ model.description }}</p>
            <div class="model-tag">{{ model.gender }}</div>
          </div>
        </div>

        <!-- Trained Models -->
        <div v-if="modelType === 'trained'" class="trained-model-section">
          <div class="model-info">
            <p>使用您自己训练的语音模型</p>
            <small v-if="!hasTrainedModel" class="warning">
              请先在语音训练页面训练模型
            </small>
          </div>
          <div v-if="hasTrainedModel" class="trained-model-card">
            <div class="model-icon">🎵</div>
            <h4>我的语音模型</h4>
            <p>基于您上传的音频训练的专属模型</p>
            <div class="model-tag">自定义</div>
          </div>
        </div>
      </div>

      <!-- Configuration Settings -->
      <div class="config-section">
        <h3>高级设置</h3>
        <div class="config-grid">
          <div class="config-item">
            <label>表情强度</label>
            <div class="range-container">
              <input
                type="range"
                v-model.number="config.expressionScale"
                min="0.5"
                max="2.0"
                step="0.1"
                class="slider"
              />
              <span class="range-value">{{ config.expressionScale }}</span>
            </div>
            <small>数值越高表情越丰富</small>
          </div>

          <div class="config-item">
            <label>面部增强</label>
            <label class="switch">
              <input type="checkbox" v-model="config.enhancer" />
              <span class="slider-toggle"></span>
            </label>
            <small>启用GF面部增强算法</small>
          </div>

          <div class="config-item">
            <label>视频分辨率</label>
            <select v-model="config.resolution">
              <option value="720p">720p (HD)</option>
              <option value="1080p">1080p (Full HD)</option>
              <option value="4k">4K (Ultra HD)</option>
            </select>
            <small>视频输出分辨率</small>
          </div>

          <div class="config-item">
            <label>帧率</label>
            <select v-model="config.fps">
              <option value="24">24 FPS (电影标准)</option>
              <option value="30">30 FPS (电视标准)</option>
              <option value="60">60 FPS (流畅)</option>
            </select>
            <small>视频帧率设置</small>
          </div>
        </div>
      </div>

      <!-- Preview Section -->
      <div class="preview-section" v-if="imagePreview">
        <h3>效果预览</h3>
        <div class="preview-container">
          <div class="preview-item">
            <h4>原始照片</h4>
            <img :src="imagePreview" alt="原始" class="preview-image" />
          </div>
          <div class="preview-item">
            <h4>像素化预览</h4>
            <div v-if="pixelateEnabled">
              <div v-if="pixelating" class="preview-placeholder">
                <div class="placeholder-icon">⏳</div>
                <p>像素化处理中...</p>
              </div>
              <img v-else-if="pixelPreview" :src="pixelPreview" alt="像素化" class="preview-image" />
              <div v-else class="preview-placeholder">
                <div class="placeholder-icon">🧩</div>
                <p>调整参数以预览像素化效果</p>
              </div>
            </div>
            <div v-else class="preview-placeholder">
              <div class="placeholder-icon">🎭</div>
              <p>关闭像素化时无预览</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Configuration Status -->
      <div class="status-section">
        <h3>当前配置状态</h3>
        <div class="status-indicators">
          <div class="status-indicator" :class="{ configured: digitalHumanStore.isPersonConfigured }">
            <div class="indicator-icon">
              <span v-if="digitalHumanStore.isPersonConfigured">✓</span>
              <span v-else>✗</span>
            </div>
            <span>数字人形象</span>
          </div>
          <div class="status-indicator" :class="{ configured: digitalHumanStore.isVoiceConfigured }">
            <div class="indicator-icon">
              <span v-if="digitalHumanStore.isVoiceConfigured">✓</span>
              <span v-else>✗</span>
            </div>
            <span>语音模型</span>
          </div>
        </div>
        <p class="status-note">
          {{ digitalHumanStore.isPersonConfigured && digitalHumanStore.isVoiceConfigured
             ? '配置已完成，可以生成视频'
             : '请完成所有配置项' }}
        </p>
      </div>

      <!-- Save Configuration -->
      <div class="action-section">
        <button
          class="save-button"
          @click="saveConfiguration"
          :disabled="!imagePreview"
        >
          保存配置
        </button>
        <button
          class="debug-button"
          @click="forceSetConfigured"
          style="margin-left: 10px; padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer;"
        >
          调试：强制设为已配置
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import { pixelateImage } from '@/utils/pixelate'

const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

const imageInput = ref<HTMLInputElement>()
const imageFile = ref<File | null>(null)
const imagePreview = ref('')
const pixelPreview = ref('')
const pixelating = ref(false)
const selectedModel = ref('0')

// 新增：语音模型类型和选择状态
const modelType = ref<'pretrained' | 'trained'>('pretrained')
const selectedPretrainedModel = ref('0')
const hasTrainedModel = ref(false) // 这里应该从store获取实际状态

const config = computed({
  get: () => digitalHumanStore.config,
  set: (value) => digitalHumanStore.setConfig(value),
})

// 像素化 UI 双向绑定（持久与上传使用 store 的值）
const pixelateEnabled = computed({
  get: () => digitalHumanStore.pixelateEnabled,
  set: (v: boolean) => {
    digitalHumanStore.setPixelateEnabled(v)
    updatePixelPreview()
  },
})
const pixelBlockSize = computed({
  get: () => digitalHumanStore.pixelBlockSize,
  set: (v: number) => digitalHumanStore.setPixelBlockSize(v),
})

const pretrainedModels = [
  { id: '0', name: '男声', description: '成熟稳重的男性声音', gender: '男' },
  { id: '1', name: '女声', description: '温柔悦耳的女性声音', gender: '女' },
]

// 原有的voiceModels保留兼容性
const voiceModels = [
  {
    id: '0',
    name: '标准女声',
    description: '清晰自然的女性声音，适合教育内容',
    language: '中文'
  },
  {
    id: '1',
    name: '标准男声',
    description: '沉稳专业的男性声音，适合商务内容',
    language: '中文'
  },
  {
    id: '2',
    name: '温柔女声',
    description: '柔和亲切的女性声音，适合儿童内容',
    language: '中文'
  },
  {
    id: 'custom',
    name: '自定义模型',
    description: '使用您训练的专属声音模型',
    language: '自定义'
  }
]

const handleImageSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    imageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target?.result as string
      updatePixelPreview()
    }
    reader.readAsDataURL(file)
  }
}

const handleImageDrop = (event: DragEvent) => {
  const file = event.dataTransfer?.files[0]
  if (file && file.type.startsWith('image/')) {
    imageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target?.result as string
      updatePixelPreview()
    }
    reader.readAsDataURL(file)
  }
}

const removeImage = () => {
  imageFile.value = null
  imagePreview.value = ''
  pixelPreview.value = ''
}

const selectModel = (modelId: string) => {
  selectedModel.value = modelId
  digitalHumanStore.setConfig({ modelIndex: modelId })
}

// 新增：语音模型选择相关函数
const onModelTypeChange = () => {
  // 切换模型类型时的处理
  console.log('模型类型切换为:', modelType.value)
}

const selectPretrainedModel = (modelId: string) => {
  selectedPretrainedModel.value = modelId
  selectedModel.value = modelId
  digitalHumanStore.setConfig({ modelIndex: modelId })
  console.log('选择预训练模型:', modelId)
}

const saveConfiguration = async () => {
  if (!imageFile.value || !authStore.currentUser) return

  try {
    // 1. 上传图片
    await digitalHumanStore.uploadImage(authStore.currentUser, imageFile.value)

    // 2. 根据模型类型选择对应的API调用
    if (modelType.value === 'pretrained') {
      // 预训练模型：调用selectVITSModel
      await digitalHumanStore.selectVITSModel(authStore.currentUser, selectedPretrainedModel.value)
    } else if (modelType.value === 'trained') {
      // 自训练模型：调用selectTrainVITSModel
      await digitalHumanStore.selectTrainVITSModel(authStore.currentUser)
    }

    // 3. 发送配置参数
    await digitalHumanStore.sendConfig(authStore.currentUser)

    alert('配置保存成功！')
  } catch (error) {
    console.error('保存失败:', error)
    alert('配置保存失败，请重试')
  }
}

// 调试方法：强制设置配置状态
const forceSetConfigured = () => {
  digitalHumanStore.setPersonConfigured(true)
  digitalHumanStore.setVoiceConfigured(true)
  alert('已强制设置为配置状态（仅用于调试）')
}
// 生成像素化预览（防抖避免频繁计算）
let lastReq = 0
function updatePixelPreview() {
  if (!pixelateEnabled.value || !imagePreview.value) {
    pixelPreview.value = ''
    pixelating.value = false
    return
  }
  const req = ++lastReq
  pixelating.value = true
  pixelateImage(imagePreview.value, {
    blockSize: pixelBlockSize.value,
    levels: 16,
    dithering: true,
  })
    .then((url) => {
      if (req === lastReq) pixelPreview.value = url
    })
    .finally(() => {
      if (req === lastReq) pixelating.value = false
    })
}

let debounceTimer: any
function debouncedUpdatePreview() {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(updatePixelPreview, 150)
}

</script>

<style scoped>
.person-customizer {
  max-width: 1000px;
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

.customizer-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.upload-section,
.voice-section,
.config-section,
.preview-section {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.upload-section h3,
.voice-section h3,
.config-section h3,
.preview-section h3 {
  margin-bottom: 20px;
  color: #333;
}

.upload-area {
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 40px;
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
  font-size: 48px;
  margin-bottom: 10px;
}

.upload-hint {
  font-size: 14px;
  color: #666;
}

.image-preview {
  position: relative;
  display: inline-block;
}

.image-preview img {
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
}

.pixelate-controls {
  margin-top: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.switch-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
}

.pixelate-slider {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* 新增：语音模型选择器样式 */
.model-type-selector {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.radio-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: 500;
}

.radio-label input[type="radio"] {
  margin: 0;
}

.trained-model-section {
  padding: 20px;
  border: 2px dashed #ddd;
  border-radius: 8px;
  text-align: center;
}

.model-info {
  margin-bottom: 15px;
}

.model-info .warning {
  color: #e74c3c;
  font-weight: 500;
}

.trained-model-card {
  padding: 20px;
  border: 2px solid #667eea;
  border-radius: 8px;
  background: #f8f9ff;
  cursor: pointer;
  transition: all 0.3s;
}

.trained-model-card:hover {
  border-color: #5a6fd8;
  background: #eef2ff;
}

.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
}

.model-card {
  border: 2px solid #ddd;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
}

.model-card:hover {
  border-color: #667eea;
  transform: translateY(-2px);
}

.model-card.active {
  border-color: #667eea;
  background: #f8f9ff;
}

.model-icon {
  font-size: 36px;
  margin-bottom: 10px;
}

.model-card h4 {
  margin-bottom: 8px;
  color: #333;
}

.model-card p {
  font-size: 14px;
  color: #666;
  margin-bottom: 10px;
}

.model-tag {
  background: #667eea;
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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

.config-item small {
  color: #666;
  font-size: 12px;
}

.range-container {
  display: flex;
  align-items: center;
  gap: 10px;
}

.slider {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: #ddd;
  outline: none;
}

.range-value {
  font-weight: 600;
  min-width: 30px;
}

.switch {
  position: relative;
  display: inline-block;
  width: 50px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider-toggle {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 24px;
}

.slider-toggle:before {
  position: absolute;
  content: "";
  height: 16px;
  width: 16px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider-toggle {
  background-color: #667eea;
}

input:checked + .slider-toggle:before {
  transform: translateX(26px);
}

.preview-container {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
}

.preview-item {
  text-align: center;
}

.preview-item h4 {
  margin-bottom: 15px;
  color: #333;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  border-radius: 8px;
}

.preview-placeholder {
  border: 2px dashed #ddd;
  border-radius: 8px;
  padding: 40px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.placeholder-icon {
  font-size: 48px;
}

.save-button {
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

.save-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.save-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 配置状态样式 */
.status-section {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  margin-bottom: 30px;
}

.status-section h3 {
  margin: 0 0 16px 0;
  font-size: 18px;
  color: #333;
}

.status-indicators {
  display: flex;
  gap: 20px;
  margin-bottom: 12px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
}

.status-indicator.configured {
  background: #d4edda;
  border-color: #c3e6cb;
  color: #155724;
}

.indicator-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 12px;
}

.status-indicator:not(.configured) .indicator-icon {
  background: #f8d7da;
  color: #721c24;
}

.status-indicator.configured .indicator-icon {
  background: #28a745;
  color: white;
}

.status-note {
  margin: 0;
  font-size: 14px;
  color: #666;
  font-style: italic;
}
</style>