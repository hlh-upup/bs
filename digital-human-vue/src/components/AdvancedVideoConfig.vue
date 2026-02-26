<template>
  <div class="advanced-video-config">
    <!-- 配置标题 -->
    <div class="config-header">
      <h2>高级视频配置</h2>
      <p>精细控制数字人在PPT中的出现位置和参数</p>
    </div>

    <!-- PPT页面预览和配置 -->
    <div class="ppt-slides-config">
      <div class="section-header">
        <h3>PPT页面配置</h3>
        <p>选择哪些页面需要显示数字人，以及对应的设置</p>
      </div>

      <div class="slides-grid">
        <div
          v-for="(slide, index) in slideConfig"
          :key="index"
          class="slide-card"
          :class="{ active: slide.enabled }"
        >
          <div class="slide-number">
            <span>第{{ index + 1 }}页</span>
          </div>

          <div class="slide-controls">
            <!-- 启用数字人 -->
            <div class="control-group">
              <label class="switch-label">
                <input
                  type="checkbox"
                  v-model="slide.enabled"
                  @change="updateSlideConfig(index, 'enabled', $event.target.checked)"
                />
                <span class="switch-slider"></span>
              </label>
              <span>显示数字人</span>
            </div>

            <!-- 数字人位置 -->
            <div class="control-group" v-if="slide.enabled">
              <label>位置:</label>
              <select
                v-model="slide.position"
                @change="updateSlideConfig(index, 'position', $event.target.value)"
                class="position-select"
              >
                <option value="bottom-right">右下角</option>
                <option value="bottom-left">左下角</option>
                <option value="bottom-center">底部居中</option>
                <option value="top-right">右上角</option>
                <option value="top-left">左上角</option>
                <option value="top-center">顶部居中</option>
              </select>
            </div>

            <!-- 数字人大小 -->
            <div class="control-group" v-if="slide.enabled">
              <label>大小:</label>
              <div class="size-control">
                <input
                  type="range"
                  min="50"
                  max="150"
                  v-model="slide.size"
                  @input="updateSlideConfig(index, 'size', parseInt($event.target.value))"
                  class="size-slider"
                />
                <span class="size-value">{{ slide.size }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 全局设置 -->
    <div class="global-settings">
      <div class="section-header">
        <h3>全局设置</h3>
        <p>应用所有页面的通用配置</p>
      </div>

      <div class="settings-grid">
        <!-- 数字人模型选择 -->
        <div class="setting-item">
          <label>数字人模型:</label>
          <select v-model="globalConfig.digitalHumanModel" class="model-select">
            <option value="sadtalker">SadTalker（高质量，推荐）</option>
            <option value="wav2lip">Wav2Lip（支持动作视频）</option>
          </select>
        </div>

        <!-- 音频模式 -->
        <div class="setting-item">
          <label>音频模式:</label>
          <select v-model="globalConfig.audioMode" class="audio-mode-select">
            <option value="vits">VITS语音合成</option>
            <option value="user-audio">用户自定义音频</option>
          </select>
        </div>

        <!-- 动作视频上传 -->
        <div class="setting-item" v-if="globalConfig.digitalHumanModel === 'wav2lip'">
          <label>动作视频:</label>
          <div class="video-upload">
            <input
              ref="videoInput"
              type="file"
              accept="video/*"
              @change="handleVideoSelect"
              style="display: none"
            />
            <div
              class="upload-area"
              :class="{ 'has-video': globalConfig.motionVideo }"
              @click="$refs.videoInput.click()"
            >
              <div v-if="!globalConfig.motionVideo" class="upload-placeholder">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M8 5v14M4 12h16M16 8l-4-4M16 16l-4-4"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
                <p>点击或拖拽上传动作视频</p>
                <span>支持 MP4、AVI 格式</span>
              </div>
              <div v-else class="video-preview">
                <video :src="globalConfig.motionVideoUrl" class="preview-video" controls></video>
                <button class="remove-video-btn" @click.stop="removeVideo">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M18 6L6 18M6 6L18 18"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </button>
              </div>
            </div>
            <div class="video-info">
              <p class="upload-hint">上传带有肢体动作的视频，让数字人更加生动</p>
              <p class="format-tips">建议视频时长：5-15秒，清晰度：720p或1080p</p>
            </div>
          </div>
        </div>

        <!-- 背景处理 -->
        <div class="setting-item">
          <label>背景处理:</label>
          <select v-model="globalConfig.backgroundMode" class="bg-mode-select">
            <option value="keep">保持原背景</option>
            <option value="remove">移除背景</option>
            <option value="blur">模糊背景</option>
          </select>
        </div>

        <!-- 表情强度 -->
        <div class="setting-item">
          <label>表情强度:</label>
          <div class="expression-control">
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              v-model="globalConfig.expressionScale"
              class="expression-slider"
            />
            <span class="expression-value">{{ globalConfig.expressionScale.toFixed(1) }}</span>
          </div>
        </div>

        <!-- 说话速度 -->
        <div class="setting-item">
          <label>说话速度:</label>
          <select v-model="globalConfig.speakingSpeed" class="speed-select">
            <option value="slow">慢速</option>
            <option value="normal">正常</option>
            <option value="fast">快速</option>
          </select>
        </div>
      </div>
    </div>

    <!-- 配置预览 -->
    <div class="config-preview">
      <div class="section-header">
        <h3>配置预览</h3>
        <p>预览当前配置的效果</p>
      </div>

      <div class="preview-container">
        <div class="preview-canvas">
          <div class="mock-slide">
            <div class="slide-content">
              <h4>PPT页面示例</h4>
              <p>这是示例PPT页面内容，数字人将出现在{{ getEnabledSlidesCount() }}个页面中</p>
            </div>
            <div v-if="getPreviewSlide()" class="digital-person" :style="getDigitalPersonStyle()">
              <div class="person-avatar">👤</div>
              <div class="speech-bubble" v-if="globalConfig.audioMode !== 'user-audio'">
                <span>语音合成示例</span>
              </div>
            </div>
          </div>
        </div>

        <div class="preview-controls">
          <button class="preview-btn" @click="testConfig">
            <svg viewBox="0 0 24 24" fill="none">
              <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
              <path
                d="M5 12v9M3 9l2-2h14l2 2-2 2v-4z"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <span>测试配置</span>
          </button>
        </div>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="action-buttons">
      <button class="save-btn" @click="saveConfiguration" :disabled="!isConfigValid">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 3.46957 3.21071 2.96086 3.58579 2.58579C3.96086 2.21071 4.46957 2 5V19C2 20.5304 3.21071 21.0391 3.58579 21.4142C3.96086 21.7893 4.46957 22 5 21H15C15.5304 22 16.0391 21.7893 16.4142 21C16.7893 20.0391 17 19.5304 17 19V12.5H21L15.5 14.5V21Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <span>保存配置</span>
      </button>

      <button class="reset-btn" @click="resetConfiguration">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M1 4C0.552285 4 0 3.44772 0 3V1C0 0.44772 0.552285 0 1H13C13.5523 1 14 1.44772 14 2V8C14 8.55228 13.5523 8 13 8H4C4 3.44772 4 2.94772 4 2.23607V4Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M19 11C19.5523 11 20 10.5523 20 10C20 10.4477 19.5523 10 19 9H9V19C9 20.4477 8.44772 21 8 21H17C16.4477 21 16 20.4477 16 20V17C16 17.4477 16.4477 17 17 19H18V20C18 20.4477 18.4477 19 20 19H19C19.5523 19 20 19.4477 20 19.4477V17C20 16.4477 20.4477 20 16 20H19Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <span>重置为默认</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

interface SlideConfig {
  enabled: boolean
  position: string
  size: number
}

interface GlobalConfig {
  digitalHumanModel: string
  audioMode: string
  motionVideo: File | null
  motionVideoUrl: string
  backgroundMode: string
  expressionScale: number
  speakingSpeed: string
}

const authStore = useAuthStore()

// 响应式数据
const slideCount = ref(10) // 假设有10页PPT
const slideConfig = ref<SlideConfig[]>(
  Array.from({ length: 10 }, (_, index) => ({
    enabled: index < 5, // 默认前5页启用数字人
    position: 'bottom-right',
    size: 80,
  })),
)

const globalConfig = ref<GlobalConfig>({
  digitalHumanModel: 'sadtalker',
  audioMode: 'vits',
  motionVideo: null,
  motionVideoUrl: '',
  backgroundMode: 'remove',
  expressionScale: 1.0,
  speakingSpeed: 'normal',
})

// 计算属性
const isConfigValid = computed(() => {
  return slideConfig.value.some((slide) => slide.enabled)
})

const getEnabledSlidesCount = computed(() => {
  return slideConfig.value.filter((slide) => slide.enabled).length
})

const getPreviewSlide = () => {
  return slideConfig.value.find((slide) => slide.enabled)
}

const getDigitalPersonStyle = () => {
  const slide = getPreviewSlide()
  if (!slide) return {}

  const positions: {
    [key: string]: {
      bottom?: string
      right?: string
      left?: string
      top?: string
      transform?: string
    }
  } = {
    'bottom-right': { bottom: '20px', right: '20px' },
    'bottom-left': { bottom: '20px', left: '20px' },
    'bottom-center': { bottom: '20px', left: '50%', transform: 'translateX(-50%)' },
    'top-right': { top: '20px', right: '20px' },
    'top-left': { top: '20px', left: '20px' },
    'top-center': { top: '20px', left: '50%', transform: 'translateX(-50%)' },
  }

  return {
    width: slide.size + '%',
    ...positions[slide.position],
    transform: slide.position.includes('center') ? positions[slide.position]?.transform : undefined,
  }
}

// 生命周期
onMounted(() => {
  loadConfiguration()
})

// 方法
const loadConfiguration = () => {
  if (!authStore.currentUser) return

  try {
    const storageKey = `advancedVideoConfig_${authStore.currentUser}`
    const savedConfig = localStorage.getItem(storageKey)

    if (savedConfig) {
      const config = JSON.parse(savedConfig)
      slideConfig.value = config.slideConfig || slideConfig.value
      globalConfig.value = { ...globalConfig.value, ...config.globalConfig }
      if (config.globalConfig.motionVideoUrl) {
        globalConfig.value.motionVideoUrl = config.globalConfig.motionVideoUrl
      }
    }
  } catch (error) {
    console.error('加载配置失败:', error)
  }
}

const saveConfiguration = () => {
  if (!authStore.currentUser) return

  try {
    const config = {
      slideConfig: slideConfig.value,
      globalConfig: {
        ...globalConfig.value,
        motionVideoUrl: globalConfig.value.motionVideoUrl,
      },
    }

    const storageKey = `advancedVideoConfig_${authStore.currentUser}`
    localStorage.setItem(storageKey, JSON.stringify(config))

    // 这里可以调用API保存到后端
    alert('配置保存成功！')
  } catch (error) {
    console.error('保存配置失败:', error)
    alert('保存配置失败，请重试')
  }
}

const resetConfiguration = () => {
  if (confirm('确定要重置所有配置为默认值吗？')) {
    slideConfig.value = Array.from({ length: slideCount.value }, (_, index) => ({
      enabled: index < 5,
      position: 'bottom-right',
      size: 80,
    }))

    globalConfig.value = {
      digitalHumanModel: 'sadtalker',
      audioMode: 'vits',
      motionVideo: null,
      motionVideoUrl: '',
      backgroundMode: 'remove',
      expressionScale: 1.0,
      speakingSpeed: 'normal',
    }

    alert('配置已重置为默认值')
  }
}

const updateSlideConfig = (index: number, field: string, value: any) => {
  if (slideConfig.value[index]) {
    ;(slideConfig.value[index] as any)[field] = value
  }
}

const handleVideoSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file && file.type.startsWith('video/')) {
    globalConfig.value.motionVideo = file
    const url = URL.createObjectURL(file)
    globalConfig.value.motionVideoUrl = url
  }
}

const removeVideo = () => {
  globalConfig.value.motionVideo = null
  globalConfig.value.motionVideoUrl = ''
}

const testConfig = async () => {
  if (!authStore.currentUser) {
    alert('请先登录')
    return
  }

  try {
    console.log('测试配置:', {
      slides: slideConfig.value,
      global: globalConfig.value,
    })

    // 这里可以调用API进行测试
    alert('配置测试功能开发中...')
  } catch (error) {
    console.error('测试配置失败:', error)
    alert('测试配置失败，请重试')
  }
}
</script>

<style scoped>
.advanced-video-config {
  padding: 32px;
  min-height: 600px;
}

.config-header {
  text-align: center;
  margin-bottom: 32px;
}

.config-header h2 {
  font-size: 28px;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 8px 0;
}

.config-header p {
  font-size: 16px;
  color: #64748b;
  margin: 0;
}

.ppt-slides-config {
  margin-bottom: 40px;
}

.section-header {
  margin-bottom: 24px;
}

.section-header h3 {
  font-size: 20px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 8px 0;
}

.section-header p {
  font-size: 14px;
  color: #64748b;
  margin: 0;
}

.slides-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.slide-card {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 20px;
  transition: all 0.3s ease;
}

.slide-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.slide-card.active {
  border-color: #667eea;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
}

.slide-number {
  font-size: 16px;
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 12px;
}

.slide-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.control-group label {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  min-width: 60px;
}

.position-select,
.size-slider,
.expression-slider {
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  font-size: 14px;
}

.size-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.size-value,
.expression-value {
  font-size: 14px;
  font-weight: 600;
  color: #1a202c;
  min-width: 50px;
}

.global-settings {
  margin-bottom: 40px;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  max-width: 1000px;
  margin: 0 auto;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.setting-item label {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
}

.model-select,
.audio-mode-select,
.bg-mode-select,
.speed-select {
  padding: 10px 12px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  background: white;
}

.video-upload {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.upload-area {
  border: 2px dashed #e2e8f0;
  border-radius: 8px;
  padding: 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-area:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.02);
}

.upload-area.has-video {
  border-style: solid;
  border-color: #667eea;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.upload-placeholder svg {
  width: 32px;
  height: 32px;
  color: #94a3b8;
}

.upload-placeholder p {
  font-size: 14px;
  font-weight: 500;
  color: #334155;
  margin: 0;
}

.upload-placeholder span {
  font-size: 12px;
  color: #64748b;
}

.video-preview {
  position: relative;
}

.preview-video {
  max-width: 100%;
  max-height: 200px;
  border-radius: 8px;
}

.remove-video-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 24px;
  height: 24px;
  border: none;
  background: #ef4444;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.remove-video-btn:hover {
  background: #dc2626;
}

.remove-video-btn svg {
  width: 14px;
  height: 14px;
}

.video-info {
  text-align: left;
}

.upload-hint,
.format-tips {
  font-size: 12px;
  color: #64748b;
  margin: 4px 0 0 0;
}

.expression-control {
  display: flex;
  align-items: center;
  gap: 12px;
}

.config-preview {
  margin-bottom: 40px;
}

.preview-container {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
}

.preview-canvas {
  position: relative;
  height: 300px;
  background: #f8fafc;
  border-radius: 8px;
  overflow: hidden;
}

.mock-slide {
  padding: 20px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.slide-content h4 {
  font-size: 18px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 16px 0;
}

.slide-content p {
  font-size: 14px;
  color: #64748b;
  margin: 0;
  flex: 1;
}

.digital-person {
  position: absolute;
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.9);
  padding: 12px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.person-avatar {
  font-size: 24px;
}

.speech-bubble {
  background: #667eea;
  color: white;
  padding: 8px 12px;
  border-radius: 16px;
  font-size: 12px;
  position: relative;
}

.speech-bubble::before {
  content: '';
  position: absolute;
  left: -8px;
  top: 50%;
  transform: translateY(-50%);
  width: 0;
  height: 0;
  border-style: solid;
  border-width: 8px 8px 0 8px;
  border-color: transparent #667eea transparent transparent transparent;
}

.preview-controls {
  text-align: center;
  margin-top: 16px;
}

.preview-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.preview-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.preview-btn svg {
  width: 16px;
  height: 16px;
}

.action-buttons {
  display: flex;
  gap: 16px;
  justify-content: center;
  margin-top: 32px;
}

.save-btn,
.reset-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.save-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.save-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.save-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.reset-btn {
  background: #f1f5f9;
  color: #475569;
  border: 1px solid #e2e8f0;
}

.reset-btn:hover {
  background: #e2e8f0;
  color: #334155;
}

.save-btn svg,
.reset-btn svg {
  width: 16px;
  height: 16px;
}

/* 开关样式 */
.switch-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.switch-label input[type='checkbox'] {
  display: none;
}

.switch-slider {
  position: relative;
  width: 44px;
  height: 24px;
  background: #cbd5e1;
  border-radius: 12px;
  transition: all 0.3s ease;
}

.switch-slider::before {
  content: '';
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background: white;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.switch-label input[type='checkbox']:checked + .switch-slider {
  background: #667eea;
}

.switch-label input[type='checkbox']:checked + .switch-slider::before {
  transform: translateX(20px);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .advanced-video-config {
    padding: 16px;
  }

  .slides-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .settings-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }

  .preview-canvas {
    height: 200px;
  }

  .action-buttons {
    flex-direction: column;
    gap: 12px;
  }

  .save-btn,
  .reset-btn {
    width: 100%;
    justify-content: center;
  }
}
</style>
