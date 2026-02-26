<template>
  <div class="video-config-view">
    <!-- 页面标题 -->
    <div class="page-header">
      <h1>视频生成配置</h1>
      <p>上传PPT和配置参数，快速生成专业的数字人教学视频</p>
    </div>

    <!-- 步骤指示器 -->
    <div class="steps-container">
      <div class="steps-wrapper">
        <div
          v-for="(step, index) in steps"
          :key="index"
          :class="[
            'step-item',
            { active: currentStep > index, current: currentStep === index + 1 },
          ]"
        >
          <div class="step-number">{{ index + 1 }}</div>
          <div class="step-content">
            <h4>{{ step.title }}</h4>
            <p>{{ step.description }}</p>
          </div>
          <div v-if="index < steps.length - 1" class="step-connector"></div>
        </div>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="content-grid">
      <!-- 左侧：上传和配置区域 -->
      <div class="upload-section">
        <!-- 先选择数字人动作模式，再进行素材上传 -->
        <div class="option-group" style="margin-bottom: 16px;">
          <label class="option-label">数字人动作模式</label>
          <div class="radio-group">
            <label class="radio-option">
              <input type="radio" v-model="digitalMotion" value="sad" />
              <span>无动作（SadTalker）- 仅口型与表情</span>
            </label>
            <label class="radio-option">
              <input type="radio" v-model="digitalMotion" value="wav" />
              <span>有动作（Wav2Lip）- 面部动作更丰富</span>
            </label>
          </div>
        </div>
        <!-- 配置状态检查 -->
        <div class="status-grid">
          <div class="status-card" :class="{ configured: digitalHumanStore.isPersonConfigured }">
            <div class="status-icon">
              <svg
                v-if="digitalHumanStore.isPersonConfigured"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="2"
              >
                <path d="M20 6L9 17L4 12" />
              </svg>
              <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <div class="status-info">
              <h4>数字人形象</h4>
              <p>{{ digitalHumanStore.isPersonConfigured ? '已配置' : '请前往形象管理设置' }}</p>
            </div>
          </div>

          <div
            v-if="digitalMotion === 'sad'"
            class="upload-area"
            :class="{ 'has-image': imagePreview, 'is-dragover': dragOverImage }"
            @drop="handleImageDrop"
            @dragover.prevent="dragOverImage = true"
            @dragleave="dragOverImage = false"
          >
            <input
              ref="imageInput"
              type="file"
              accept="image/*"
              @change="handleImageSelect"
              style="display: none"
            />

            <div v-if="!imagePreview" class="upload-placeholder" @click="$refs.imageInput.click()">
              <div class="upload-icon-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p>点击或拖拽上传照片</p>
              <span>支持 JPG、PNG 格式</span>
            </div>

            <div v-else class="image-preview">
              <img :src="imagePreview" alt="预览" />
              <button class="remove-btn" @click="removeImage">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
            <!-- 图片上传后：卡通化与形象框选项 -->
            <div v-if="imagePreview" class="img-preprocess-panel">
              <div class="pre-row" style="justify-content: space-between;">
                <div style="display:flex; align-items:center; gap:8px;">
                  <label class="pre-label">卡通化</label>
                  <label class="switch">
                    <input type="checkbox" v-model="cartoonizeEnabled" />
                    <span class="slider"></span>
                  </label>
                </div>
                <div v-if="cartoonizeEnabled" style="display:flex; align-items:center; gap:8px;">
                  <select class="mode-select" v-model="cartoonBackendMode">
                    <option value="animegan_v2">AnimeGAN v2/v3</option>
                    <option value="wbc">White-box Cartoonization</option>
                    <!-- 可选：<option value="cv_stylize">OpenCV 卡通</option> -->
                  </select>
                  <button v-if="cartoonBackendMode==='animegan_v2'" class="style-btn" type="button" @click="styleListVisible = !styleListVisible">
                    {{ styleLabelMap[cartoonBackendStyle] }} ▼
                  </button>
                </div>
              </div>
              <transition name="fade">
                <div v-if="cartoonizeEnabled && cartoonBackendMode==='animegan_v2' && styleListVisible" class="style-list-panel">
                  <div class="style-list-grid">
                    <div
                      v-for="s in styleOptions"
                      :key="s"
                      :class="['style-list-item', { active: cartoonBackendStyle === s }]"
                      @click="cartoonBackendStyle = s; styleListVisible = false"
                    >
                      <span class="style-name">{{ styleLabelMap[s] }}</span>
                      <span v-if="cartoonBackendStyle === s" class="style-check">✓</span>
                    </div>
                  </div>
                  <div class="style-hint">选择 AnimeGAN 风格（包含 v2 / v3）</div>
                </div>
              </transition>

              <div class="pre-row" style="justify-content: space-between; margin-top:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                  <label class="pre-label">形象框</label>
                  <label class="switch">
                    <input type="checkbox" v-model="animeFrameEnabled" />
                    <span class="slider"></span>
                  </label>
                </div>
                <div v-if="animeFrameEnabled" class="frame-style-chips">
                  <button
                    v-for="f in ['panel','glow','film']"
                    :key="f"
                    type="button"
                    class="frame-chip"
                    :class="{ active: animeFrameStyle === f }"
                    @click="animeFrameStyle = f as any"
                  >
                    {{ frameStyleLabelMap[f] }}
                  </button>
                </div>
              </div>
              <div class="pre-row" v-if="cartoonizeEnabled && imagePreview" style="justify-content:flex-end; margin-top:4px;">
                <button class="btn-primary" type="button" :disabled="previewLoading" @click="previewCartoonize">
                  {{ previewLoading ? '生成中…' : '生成卡通化预览' }}
                </button>
              </div>
              <div class="preview-note" v-if="cartoonizeEnabled">
                上传后将调用后端 {{ cartoonBackendMode==='animegan_v2' ? 'AnimeGAN' : (cartoonBackendMode==='wbc' ? 'WBC' : cartoonBackendMode) }} 进行卡通化。
              </div>
              <div v-if="cartoonInfo" class="preview-note" style="margin-top:4px;">
                后端实际模式：{{ cartoonInfo.mode || '-' }}
                <span v-if="cartoonInfo.style">，风格：{{ cartoonInfo.style }}</span>
              </div>
            </div>
          </div>
          <div
            v-else
            class="upload-area"
            :class="{ 'has-file': teacherVideoFile, 'is-dragover': dragOverTeacherVideo }"
          >
            <input
              ref="teacherVideoInput"
              type="file"
              accept="video/mp4,video/*"
              @change="handleTeacherVideoSelect"
              style="display: none"
            />

            <div
              v-if="!teacherVideoFile"
              class="upload-placeholder"
              @click="$refs.teacherVideoInput.click()"
            >
              <div class="upload-icon-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polygon points="23 7 16 12 23 17 23 7" />
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                </svg>
              </div>
              <p>点击选择教师视频（mp4）</p>
              <span>Wav2Lip 模式用于更丰富的面部动作</span>
            </div>

            <div v-else class="file-preview">
              <div class="file-info">
                <div class="file-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="23 7 16 12 23 17 23 7" />
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                  </svg>
                </div>
                <div>
                  <p class="file-name">{{ teacherVideoFile?.name }}</p>
                  <p class="file-size">{{ formatFileSize(teacherVideoFile?.size || 0) }}</p>
                </div>
              </div>
              <button class="remove-btn" @click="removeTeacherVideo">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <!-- PPT文件上传 -->
        <div class="upload-card">
          <div class="upload-header">
            <div class="upload-icon document">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
            </div>
            <div>
              <h3>教学PPT</h3>
              <p>上传包含批注的PPT文件</p>
            </div>
          </div>

          <div
            class="upload-area"
            :class="{ 'has-file': pptFile, 'is-dragover': dragOverPPT }"
            @drop="handlePPTDrop"
            @dragover.prevent="dragOverPPT = true"
            @dragleave="dragOverPPT = false"
          >
            <input
              ref="pptInput"
              type="file"
              accept=".ppt,.pptx"
              @change="handlePPTSelect"
              style="display: none"
            />

            <div v-if="!pptFile" class="upload-placeholder" @click="$refs.pptInput.click()">
              <div class="upload-icon-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <p>点击或拖拽上传PPT</p>
              <span>支持 PPT、PPTX 格式</span>
            </div>

            <div v-else class="file-preview">
              <div class="file-info">
                <div class="file-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <div>
                  <p class="file-name">{{ pptFile.name }}</p>
                  <p class="file-size">{{ formatFileSize(pptFile.size) }}</p>
                </div>
              </div>
              <button class="remove-btn" @click="removePPT">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>

        <!-- PPT备注编辑区 -->
        <div class="notes-section">
          <div class="notes-header">
            <h3>PPT备注内容</h3>
            <div class="notes-status">
              <span v-if="pptRemakes.trim().length > 0" class="status-badge success"
                >✅ 已解析</span
              >
              <span v-else class="status-badge pending">⏳ 等待上传PPT</span>
            </div>
          </div>
          <textarea
            v-model="pptRemakes"
            placeholder="PPT备注内容将在这里显示，您也可以手动编辑..."
            class="notes-textarea"
          ></textarea>
          <div class="notes-footer">
            <p class="notes-hint">💡 上传PPT后会自动解析备注内容，您也可以手动编辑JSON格式的备注</p>
            <button
              v-if="pptRemakes.trim().length > 0"
              @click="formatPPTRemakes"
              class="format-btn"
            >
              格式化JSON
            </button>
          </div>
        </div>

        <!-- 配置选项 -->
        <div class="config-options">
          <h3>配置选项</h3>

          

          <div class="option-group">
            <label class="option-label">数字人插入方式</label>
            <div class="radio-group">
              <label class="radio-option">
                <input type="radio" v-model="insertionMode" value="all" />
                <span>全部插入 - 在所有页面插入数字人讲解</span>
              </label>
              <label class="radio-option">
                <input type="radio" v-model="insertionMode" value="none" />
                <span>全部不插入 - 生成纯讲解音频</span>
              </label>
              <label class="radio-option">
                <input type="radio" v-model="insertionMode" value="select" />
                <span>部分插入 - 选择特定页面插入数字人</span>
              </label>
            </div>
          </div>

          <div class="option-group" v-if="insertionMode === 'select'">
            <label class="option-label">选择插入页</label>
            <div class="slides-grid">
              <button
                v-for="n in availableSlides"
                :key="n"
                type="button"
                class="slide-chip"
                :class="{ selected: selectedSlideNumbers.includes(n) }"
                @click="toggleSlide(n)"
              >
                第 {{ n }} 页
              </button>
            </div>
            <p class="slides-hint">提示：至少选择一页用于插入数字人</p>
          </div>

          <div class="option-group">
            <label class="option-label">音频模式</label>
            <div class="radio-group">
              <label class="radio-option">
                <input type="radio" v-model="audioMode" value="generate" />
                <span>AI生成音频 - 使用AI根据PPT内容生成语音</span>
              </label>
              <label class="radio-option">
                <input type="radio" v-model="audioMode" value="upload" />
                <span>上传音频 - 使用预先录制好的音频文件</span>
              </label>
            </div>
          </div>

          <!-- 当选择上传音频时，提供每页音频上传 -->
          <div class="option-group" v-if="audioMode === 'upload'">
            <label class="option-label">上传每页音频</label>
            <div class="slides-audio-list" v-if="availableSlides.length > 0">
              <div class="slide-audio-item" v-for="n in availableSlides" :key="n">
                <div class="slide-audio-left">
                  <span class="slide-tag">第 {{ n }} 页</span>
                  <span class="slide-audio-name" v-if="slideAudios[n]">{{
                    slideAudios[n]?.name
                  }}</span>
                  <span class="slide-audio-missing" v-else>未选择音频</span>
                </div>
                <div class="slide-audio-right">
                  <input
                    :ref="setSlideAudioRef(n)"
                    type="file"
                    accept="audio/*"
                    class="hidden-file"
                    @change="onSlideAudioChange(n, $event)"
                  />
                  <button type="button" class="pick-audio-btn" @click="triggerSlideAudio(n)">
                    选择音频
                  </button>
                </div>
              </div>
            </div>
            <p class="slides-hint">为每一页选择对应的音频文件；将用于“用户音频”推理与合并</p>
          </div>
        </div>

        <!-- 生成按钮 -->
        <div class="action-section">
          <button
            class="generate-btn"
            :class="{ 'can-generate': canGenerate, 'is-generating': isGenerating }"
            :disabled="!canGenerate || isGenerating"
            @click="generateVideo"
          >
            <div v-if="!isGenerating" class="btn-content">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              <span>开始生成视频</span>
            </div>
            <div v-else class="loading-content">
              <div class="loading-spinner"></div>
              <span>生成中...</span>
            </div>
          </button>

          <div v-if="isGenerating" class="progress-section">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
            </div>
            <p class="progress-text">{{ currentStepText }}</p>
          </div>

          <div v-if="!canGenerate" class="requirements-hint">
            <p>请完成以下配置后再生成视频：</p>
            <ul>
              <li v-if="!digitalHumanStore.isPersonConfigured">⚠️ 配置数字人形象</li>
              <li v-if="!digitalHumanStore.isVoiceConfigured">⚠️ 配置语音模型</li>
              <li v-if="!pptFile">⚠️ 上传PPT文件</li>
              <li v-if="!pptRemakes.trim()">⚠️ 确保PPT包含备注内容</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- 右侧：预览和提示区域 -->
      <div class="preview-section">
        <div class="preview-card">
          <div class="preview-header">
            <h3>预览区域</h3>
            <div class="preview-status">
              <div class="status-dot"></div>
              <span>就绪</span>
            </div>
          </div>

          <div class="preview-content">
            <div v-if="!imagePreview" class="empty-preview">
              <div class="empty-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="2" y="2" width="20" height="20" rx="2.18" />
                  <line x1="7" y1="2" x2="7" y2="22" />
                  <line x1="17" y1="2" x2="17" y2="22" />
                  <line x1="2" y1="12" x2="22" y2="12" />
                  <line x1="2" y1="7" x2="7" y2="7" />
                  <line x1="2" y1="17" x2="7" y2="17" />
                  <line x1="17" y1="7" x2="22" y2="7" />
                  <line x1="17" y1="17" x2="22" y2="17" />
                </svg>
              </div>
              <p>上传照片后将显示预览</p>
            </div>

            <div v-else class="image-preview-container">
              <div class="preview-label">原始照片</div>
              <div :class="['anime-frame', animeFrameEnabled ? animeFrameStyle : '']">
                <img :src="imagePreview" alt="预览" class="preview-image" />
              </div>
              <div v-if="cartoonPreviewUrl" class="image-preview-container" style="margin-top:12px;">
                <div class="preview-label">
                  卡通化预览（
                  {{ cartoonBackendMode==='animegan_v2' ? styleLabelMap[cartoonBackendStyle] : (cartoonBackendMode==='wbc' ? 'WBC' : cartoonBackendMode) }}
                  ）
                </div>
                <div :class="['anime-frame', animeFrameEnabled ? animeFrameStyle : '']">
                  <img :src="cartoonPreviewUrl" alt="卡通预览" class="preview-image" />
                </div>
                <div class="preview-actions">
                  <button type="button" class="small-btn" @click="clearCartoonPreview">清除预览</button>
                  <button type="button" class="small-btn primary" :disabled="uploadPreviewLoading" @click="useCartoonPreview">
                    {{ uploadPreviewLoading ? '上传中…' : '设为数字人形象' }}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 提示信息 -->
        <div class="tips-card">
          <div class="tips-header">
            <div class="tips-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 16v-4M12 8h.01" />
              </svg>
            </div>
            <h4>使用提示</h4>
          </div>
          <ul class="tips-list">
            <li>确保照片为正面人脸，光线充足</li>
            <li>PPT文件每页都需要有批注内容</li>
            <li>生成过程可能需要几分钟，请耐心等待</li>
            <li>建议使用Chrome或Edge浏览器获得最佳体验</li>
          </ul>
        </div>

        <!-- 快速操作 -->
        <div class="quick-actions">
          <h3>快速操作</h3>
          <div class="action-buttons">
            <router-link to="/dashboard/person-manager" class="action-btn secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
              管理形象
            </router-link>
            <router-link to="/dashboard/voice-trainer" class="action-btn secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1v22M5 12h14" />
              </svg>
              语音训练
            </router-link>
            <router-link to="/dashboard/video-list" class="action-btn secondary">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path
                  d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"
                />
              </svg>
              视频列表
            </router-link>
            <button
              type="button"
              class="action-btn"
              @click="quickSynthesizeTest"
              :disabled="!digitalHumanStore.isVoiceConfigured"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 3l14 9L5 21V3z" />
              </svg>
              语音合成试听
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import { digitalHumanApi } from '@/services/api'

const router = useRouter()
const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

// Template refs
const imageInput = ref<HTMLInputElement>()
const teacherVideoInput = ref<HTMLInputElement>()
const pptInput = ref<HTMLInputElement>()

// 状态管理
const currentStep = ref(1)
const dragOverImage = ref(false)
const dragOverPPT = ref(false)
const dragOverTeacherVideo = ref(false)
const imageFile = ref<File | null>(null)
const imagePreview = ref('')
const teacherVideoFile = ref<File | null>(null)
const pptFile = ref<File | null>(null)
const pptRemakes = ref('')
const isGenerating = ref(false)

// 配置选项
const digitalMotion = ref<'sad' | 'wav'>('sad')
const insertionMode = ref('all')
const audioMode = ref('generate')
const selectedSlideNumbers = ref<number[]>([])
const slideAudios = ref<Record<number, File | null>>({})
const slideFileInputs = ref<Record<number, HTMLInputElement | null>>({})

// 步骤配置
const steps = [
  { title: '上传照片', description: '上传清晰的正面人脸照片' },
  { title: '上传PPT', description: '上传包含批注的PPT文件' },
  { title: '配置参数', description: '设置数字人表现参数' },
  { title: '生成视频', description: 'AI自动生成数字人视频' },
]

// 计算属性
const canGenerate = computed(() => {
  const needImage = digitalMotion.value === 'sad'
  const hasFaceOrVideo = needImage ? imageFile.value : teacherVideoFile.value
  return !!(
    hasFaceOrVideo &&
    pptFile.value &&
    pptRemakes.value.trim() &&
    digitalHumanStore.isPersonConfigured &&
    digitalHumanStore.isVoiceConfigured
  )
})

const progressPercent = computed(() => ((currentStep.value - 1) / 3) * 100)

const currentStepText = computed(() => {
  const texts = ['准备中...', '上传图片中', '上传PPT中', '配置参数中', '生成视频中']
  return texts[currentStep.value] || texts[0]
})

// 文件处理方法
const handleImageSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    imageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
    digitalHumanStore.setUploadedFile('image', file)
    currentStep.value = Math.max(currentStep.value, 2)
  }
}

const handleImageDrop = (event: DragEvent) => {
  dragOverImage.value = false
  const file = event.dataTransfer?.files[0]
  if (file && file.type.startsWith('image/')) {
    imageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
    digitalHumanStore.setUploadedFile('image', file)
    currentStep.value = Math.max(currentStep.value, 2)
  }
}

const removeImage = () => {
  imageFile.value = null
  imagePreview.value = ''
  digitalHumanStore.setUploadedFile('image', null)
  if (!pptFile.value) {
    currentStep.value = 1
  }
}

const handlePPTSelect = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    pptFile.value = file
    digitalHumanStore.setUploadedFile('ppt', file)
    await parsePPTRemakes(file)
    currentStep.value = Math.max(currentStep.value, 3)
  }
}

const handleTeacherVideoSelect = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file || !authStore.currentUser) return
  teacherVideoFile.value = file
  try {
    const ok = await digitalHumanApi.sendTeacherVideo({ User: authStore.currentUser }, file)
    if (ok) {
      digitalHumanStore.setPersonConfigured(true)
      currentStep.value = Math.max(currentStep.value, 2)
    } else {
      alert('教师视频上传失败，请重试')
    }
  } catch (e) {
    console.error(e)
    alert('教师视频上传失败，请检查网络或后端')
  }
}

const removeTeacherVideo = () => {
  teacherVideoFile.value = null
}

const handlePPTDrop = async (event: DragEvent) => {
  dragOverPPT.value = false
  const file = event.dataTransfer?.files[0]
  if (file && (file.name.endsWith('.ppt') || file.name.endsWith('.pptx'))) {
    pptFile.value = file
    digitalHumanStore.setUploadedFile('ppt', file)
    await parsePPTRemakes(file)
    currentStep.value = Math.max(currentStep.value, 3)
  }
}

const removePPT = () => {
  pptFile.value = null
  pptRemakes.value = ''
  digitalHumanStore.setUploadedFile('ppt', null)
  if (!imageFile.value) {
    currentStep.value = 1
  }
}

// PPT解析方法
const parsePPTRemakes = async (file: File) => {
  if (!authStore.currentUser) return

  try {
    const result = await digitalHumanApi.uploadPPTParseRemakes(
      { User: authStore.currentUser },
      file,
    )

    if (result.success && result.data) {
      pptRemakes.value = JSON.stringify(result.data.remakes, null, 2)
    } else {
      const fallbackRemakes = {
        'Slide 1': '这是一个测试PPT的演讲内容',
        'Slide 2': '系统会自动根据备注生成语音',
      }
      pptRemakes.value = JSON.stringify(fallbackRemakes, null, 2)
    }
  } catch (error) {
    const fallbackRemakes = {
      'Slide 1': '这是一个测试PPT的演讲内容',
      'Slide 2': '系统会自动根据备注生成语音',
    }
    pptRemakes.value = JSON.stringify(fallbackRemakes, null, 2)
  }
}

const formatPPTRemakes = () => {
  try {
    const parsed = JSON.parse(pptRemakes.value)
    pptRemakes.value = JSON.stringify(parsed, null, 2)
  } catch (error) {
    alert('JSON格式错误，请检查格式')
  }
}

const availableSlides = computed<number[]>(() => {
  try {
    const obj = JSON.parse(pptRemakes.value || '{}')
    // 兼容形如 {"Slide 1": "...", "Slide 2": "..."}
    const keys = Object.keys(obj)
    const nums = keys
      .map((k) => {
        const m = k.match(/(\d+)/)
        return m ? parseInt(m[1], 10) : NaN
      })
      .filter((n) => !Number.isNaN(n))
      .sort((a, b) => a - b)
    return nums
  } catch (e) {
    return []
  }
})

const toggleSlide = (n: number) => {
  const idx = selectedSlideNumbers.value.indexOf(n)
  if (idx >= 0) selectedSlideNumbers.value.splice(idx, 1)
  else selectedSlideNumbers.value.push(n)
}

// 选择每页音频上传
const setSlideAudioRef = (n: number) => (el: HTMLInputElement | null) => {
  slideFileInputs.value[n] = el
}
const triggerSlideAudio = (n: number) => {
  slideFileInputs.value[n]?.click()
}
const onSlideAudioChange = (n: number, e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0] || null
  slideAudios.value[n] = file
}

// 视频生成方法
const generateVideo = async () => {
  if (!canGenerate.value || !authStore.currentUser) return

  if (!digitalHumanStore.isPersonConfigured) {
    alert('请先在形象管理页面配置数字人形象')
    return
  }

  if (!digitalHumanStore.isVoiceConfigured) {
    alert('请先在形象管理页面配置语音模型')
    return
  }

  isGenerating.value = true

  try {
    // 根据动作模式上传形象
    if (digitalMotion.value === 'sad') {
      if (imageFile.value) {
        await digitalHumanStore.uploadImage(authStore.currentUser, imageFile.value)
      }
    } else {
      if (teacherVideoFile.value) {
        await digitalHumanApi.sendTeacherVideo(
          { User: authStore.currentUser },
          teacherVideoFile.value,
        )
        digitalHumanStore.setPersonConfigured(true)
      }
    }

    await digitalHumanStore.sendPPTRemakes(authStore.currentUser, pptRemakes.value)
    await digitalHumanStore.sendConfig(authStore.currentUser)

    // 若选择“上传音频”，先逐页上传音频文件（对齐C# FormSet：/Send_PPT_Audio）
    if (audioMode.value === 'upload') {
      if (availableSlides.value.length === 0) {
        alert('未检测到可用的PPT页码，请检查备注解析')
        return
      }
      // 要求每一页都有音频
      for (const n of availableSlides.value) {
        if (!slideAudios.value[n]) {
          alert(`第 ${n} 页未选择音频，请补全`)
          return
        }
      }
      for (const n of availableSlides.value) {
        const file = slideAudios.value[n]!
        // 与C#约定：后端通常以0为起始索引存储如 0.wav、1.wav
        await digitalHumanApi.sendPPTAudio({ User: authStore.currentUser, Slide: n - 1 }, file)
      }
    }

    // 映射UI选项到Store的视频生成选项
    const into = insertionMode.value === 'all' ? 1 : insertionMode.value === 'none' ? 2 : 3
    const digital = digitalMotion.value === 'sad' ? 1 : 2
    const useModel = audioMode.value === 'generate'

    if (into === 3 && selectedSlideNumbers.value.length === 0) {
      alert('请选择至少一页用于“部分插入”')
      return
    }

    digitalHumanStore.setVideoOptions({
      digitalMotion: digital,
      intoDigitalOperation: into as 1 | 2 | 3,
      useModelAudio: useModel,
      selectedSlides: into === 3 ? [...selectedSlideNumbers.value] : [],
    })

    const success = await digitalHumanStore.generateDigitalHumanAdvanced(authStore.currentUser)
    if (success) {
      currentStep.value = 4
      // 可以跳转到视频列表或显示成功消息
    }
  } catch (error) {
    console.error('视频生成失败:', error)
  } finally {
    isGenerating.value = false
  }
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

// ===== 新增：卡通化 & 动漫形象框（与全局 store 绑定保持一致） =====
// 使用与 VideoGenerator.vue 相同的绑定策略，确保两个页面共享设置
const cartoonizeEnabled = computed<boolean>({
  get: () => digitalHumanStore.cartoonizeEnabled,
  set: (v) => digitalHumanStore.setCartoonizeEnabled(v),
})
const cartoonBackendStyle = computed<
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
>({
  get: () => digitalHumanStore.cartoonBackendStyle ?? 'hayao',
  set: (v) => digitalHumanStore.setCartoonBackendStyle(v),
})
const cartoonBackendMode = computed<'animegan_v2' | 'wbc' | 'cv_stylize' | 'bilateral'>( {
  get: () => (digitalHumanStore as any).cartoonBackendMode || 'animegan_v2',
  set: (v) => (digitalHumanStore as any).setCartoonBackendMode(v),
})
const styleLabelMap: Record<string, string> = {
  hayao: '宫崎骏(v2)',
  shinkai: '新海诚(v2)',
  paprika: '今敏(v2)',
  celeba: '人像(v2)',
  animeganv3_paprika: '今敏 Paprika(v3)',
  paprika_v3: '今敏 Paprika(v3)',
  animeganv3_hayao: '宫崎骏(v3)',
  hayao_v3: '宫崎骏(v3)',
  animeganv3_shinkai: '新海诚(v3)',
  shinkai_v3: '新海诚(v3)',
}
const styleOptions: Array<
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
> = ['hayao','shinkai','paprika','celeba','animeganv3_paprika','animeganv3_hayao','animeganv3_shinkai']
const styleListVisible = ref(true)

// 动漫形象框设置
const animeFrameEnabled = computed<boolean>({
  get: () => digitalHumanStore.animeFrameEnabled,
  set: (v) => digitalHumanStore.setAnimeFrameEnabled(v),
})
const animeFrameStyle = computed<'panel' | 'glow' | 'film'>({
  get: () => digitalHumanStore.animeFrameStyle,
  set: (v) => digitalHumanStore.setAnimeFrameStyle(v),
})
const frameStyleLabelMap: Record<string, string> = {
  panel: '面板描边',
  glow: '霓虹光效',
  film: '赛璐璐边框',
}

// 卡通化预览
const previewLoading = ref(false)
const cartoonPreviewUrl = ref('')
const cartoonInfo = ref<{ mode?: string; style?: string } | null>(null)
const previewCartoonize = async () => {
  if (!authStore.currentUser) {
    alert('请先登录')
    return
  }
  if (!imagePreview.value) {
    alert('请先上传图片')
    return
  }
  try {
    previewLoading.value = true
    const raw = imagePreview.value.split(',')[1]
    const res = await digitalHumanApi.cartoonizeImage({
      User: authStore.currentUser,
      Img: raw,
      Mode: cartoonBackendMode.value as any,
      Style: cartoonBackendMode.value === 'animegan_v2' ? (cartoonBackendStyle.value as any) : undefined,
      Params: { max_side: 1600 },
    })
    if (res.success && res.img) {
      cartoonPreviewUrl.value = 'data:image/png;base64,' + res.img
      cartoonInfo.value = { mode: res.mode_used, style: res.style_used }
    } else {
      alert('生成卡通化预览失败：' + (res.error || '未知错误'))
    }
  } catch (e) {
    console.error('[previewCartoonize] error', e)
    alert('生成卡通化预览异常，请稍后重试')
  } finally {
    previewLoading.value = false
  }
}
const clearCartoonPreview = () => {
  cartoonPreviewUrl.value = ''
  cartoonInfo.value = null
}

// 语音合成试听（快速）
const quickSynthesizeTest = async () => {
  if (!authStore.currentUser) return
  try {
    // 触发一次VITS推理（单次）
    const ok = await digitalHumanStore.getInferenceVITS(authStore.currentUser)
    if (!ok) {
      alert('语音合成触发失败，请检查配置')
      return
    }
    // 拉取合成音频
    const blob = await digitalHumanStore.pullVITSAudio(authStore.currentUser)
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.play()
  } catch (e) {
    console.error(e)
    alert('语音合成失败，请稍后重试')
  }
}
</script>

<style scoped>
/* 视频配置页面样式 */
.video-config-view {
  /* 让页面主体自行满幅，由外层 content-area 控制留白 */
  padding: 0;
  max-width: none;
  width: 100%;
}

/* 页面标题 */
.page-header {
  text-align: center;
  margin-bottom: var(--spacing-4xl);
}

.page-header h1 {
  font-size: var(--text-4xl);
  font-weight: var(--font-bold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-sm) 0;
}

.page-header p {
  font-size: var(--text-lg);
  color: var(--color-secondary);
  margin: 0;
}

/* 步骤指示器 */
.steps-container {
  margin-bottom: var(--spacing-5xl);
}

.steps-wrapper {
  display: flex;
  justify-content: space-between;
  position: relative;
  max-width: 800px;
  margin: 0 auto;
}

.step-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  flex: 1;
  z-index: 1;
}

.step-connector {
  position: absolute;
  top: 20px;
  left: 50%;
  width: 100%;
  height: 2px;
  background: var(--color-gray-200);
  z-index: -1;
}

.step-item.active ~ .step-connector {
  background: var(--color-primary);
}

.step-number {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--color-gray-200);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-semibold);
  color: var(--color-secondary);
  margin-bottom: var(--spacing-md);
  transition: all var(--transition-smooth);
}

.step-item.active .step-number,
.step-item.current .step-number {
  background: var(--gradient-primary);
  color: white;
  box-shadow: var(--shadow-primary);
}

.step-content {
  text-align: center;
}

.step-content h4 {
  font-size: var(--text-sm);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-xs) 0;
}

.step-content p {
  font-size: var(--text-xs);
  color: var(--color-secondary);
  margin: 0;
}

/* 主要内容网格 */
.content-grid {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: var(--spacing-4xl);
  max-width: none;
  width: 100%;
}

/* 状态网格 */
.status-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-lg);
  margin-bottom: var(--spacing-3xl);
}

.status-card {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  transition: all var(--transition-smooth);
}

.status-card.configured {
  background: var(--color-success-light);
  border-color: var(--color-success);
}

.status-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.status-card:not(.configured) .status-icon {
  background: var(--color-warning-light);
  color: var(--color-warning);
}

.status-card.configured .status-icon {
  background: var(--color-success);
  color: white;
}

.status-icon svg {
  width: 20px;
  height: 20px;
}

.status-info h4 {
  font-size: var(--text-sm);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-xs) 0;
}

.status-info p {
  font-size: var(--text-xs);
  color: var(--color-secondary);
  margin: 0;
}

/* 上传网格 */
.upload-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-xl);
  margin-bottom: var(--spacing-3xl);
}

.upload-card {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  overflow: hidden;
  transition: all var(--transition-smooth);
}

.upload-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-xl);
}

.upload-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  background: var(--color-gray-50);
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.upload-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.upload-icon.photo {
  background: var(--gradient-primary);
}

.upload-icon.document {
  background: var(--gradient-accent);
}

.upload-icon svg {
  width: 20px;
  height: 20px;
}

.upload-header h3 {
  font-size: var(--text-base);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-xs) 0;
}

.upload-header p {
  font-size: var(--text-sm);
  color: var(--color-secondary);
  margin: 0;
}

/* 上传区域 */
.upload-area {
  padding: var(--spacing-2xl);
  border: 2px dashed var(--color-gray-200);
  border-radius: var(--radius-lg);
  text-align: center;
  transition: all var(--transition-smooth);
  cursor: pointer;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-area:hover {
  border-color: var(--color-primary);
  background: var(--color-primary-lightest);
}

.upload-area.is-dragover {
  border-color: var(--color-primary);
  background: var(--color-primary-lightest);
  transform: scale(1.02);
}

.upload-area.has-image,
.upload-area.has-file {
  border-color: var(--color-primary);
  border-style: solid;
  background: var(--color-primary-lightest);
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
}

.upload-icon-placeholder {
  width: 48px;
  height: 48px;
  color: var(--color-secondary);
}

.upload-placeholder p {
  font-size: var(--text-base);
  font-weight: var(--font-medium);
  color: var(--color-gray-700);
  margin: 0;
}

.upload-placeholder span {
  font-size: var(--text-sm);
  color: var(--color-secondary);
}

/* 图片预览 */
.image-preview {
  position: relative;
  display: inline-block;
}

.image-preview img {
  max-width: 100%;
  max-height: 200px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
}
/* ===== 新增：卡通化面板与风格列表样式（与 VideoGenerator 保持风格统一的轻量版） ===== */
.img-preprocess-panel {
  margin-top: var(--spacing-lg);
  padding: 14px 16px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pre-row { display:flex; align-items:center; gap:12px; }
.pre-label { font-size:13px; font-weight:600; color:#334155; }
.preview-note { font-size:11px; color:#64748b; }
.style-btn { padding:6px 10px; background:#fff; border:1px solid #e2e8f0; border-radius:6px; font-size:12px; cursor:pointer; color:#334155; }
.style-btn:hover { background:#f1f5f9; }
.mode-select { padding:6px 10px; background:#fff; border:1px solid #e2e8f0; border-radius:6px; font-size:12px; cursor:pointer; color:#334155; }
.mode-select:focus { outline:none; border-color:#6366f1; box-shadow:0 0 0 2px rgba(99,102,241,.25); }
.style-list-panel { padding:8px 8px 10px; background:#fff; border:1px solid #e5e7eb; border-radius:10px; }
.style-list-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; }
.style-list-item { display:flex; align-items:center; justify-content:space-between; padding:8px 10px; border:1px solid #e5e7eb; border-radius:8px; cursor:pointer; font-size:12px; color:#374151; }
.style-list-item:hover { background:#f9fafb; }
.style-list-item.active { background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }
.style-check { color:#667eea; font-weight:600; }
.style-hint { margin-top:6px; font-size:11px; color:#64748b; }
.fade-enter-active,.fade-leave-active { transition: opacity .2s ease; }
.fade-enter-from,.fade-leave-to { opacity:0; }
/* Switch 通用样式 */
.switch { position:relative; display:inline-block; width:42px; height:22px; }
.switch input { opacity:0; width:0; height:0; }
.switch .slider { position:absolute; inset:0; background:#cbd5e1; border-radius:22px; cursor:pointer; transition:.3s; }
.switch .slider:before { content:''; position:absolute; width:16px; height:16px; left:3px; top:3px; background:#fff; border-radius:50%; box-shadow:0 2px 4px rgba(0,0,0,.15); transition:.3s; }
.switch input:checked + .slider { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); }
.switch input:checked + .slider:before { transform:translateX(20px); }
/* Frame style chips */
.frame-style-chips { display:flex; gap:6px; }
.frame-chip { padding:4px 8px; font-size:11px; border:1px solid #e2e8f0; background:#fff; border-radius:6px; cursor:pointer; color:#475569; }
.frame-chip.active { background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }
/* Anime frame preview (applied later in right preview area if reused) */
.anime-frame { padding:6px; border-radius:10px; transition: box-shadow .25s ease, border-color .25s ease; }
.anime-frame.panel { border:2px solid #6366f1; box-shadow:0 0 0 2px rgba(99,102,241,0.15) inset, 0 6px 14px rgba(99,102,241,0.2); }
.anime-frame.glow { border:1px solid rgba(147,197,253,0.6); box-shadow:0 0 18px rgba(59,130,246,0.45), 0 0 42px rgba(147,197,253,0.25); background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.12), transparent 70%); }
.anime-frame.film { border:2px dashed #94a3b8; background: repeating-linear-gradient(-45deg, rgba(148,163,184,0.1) 0 6px, rgba(255,255,255,0) 6px 12px); }

/* Buttons for preview actions */
.btn-primary {
  padding: 8px 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 13px;
  cursor: pointer;
}
.btn-primary:disabled { opacity: .7; cursor: not-allowed; }
.preview-actions { margin-top: 6px; }
.small-btn {
  padding: 6px 10px;
  border: 1px solid #e2e8f0;
  background: #fff;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  color: #334155;
}
.small-btn:hover { background:#f1f5f9; }

/* 文件预览 */
.file-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--spacing-md);
  width: 100%;
}

.file-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  flex: 1;
}

.file-icon {
  width: 40px;
  height: 40px;
  background: var(--color-accent);
  border-radius: var(--radius-lg);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.file-icon svg {
  width: 20px;
  height: 20px;
}

.file-name {
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-xs) 0;
}

.file-size {
  font-size: var(--text-xs);
  color: var(--color-secondary);
  margin: 0;
}

/* 删除按钮 */
.remove-btn {
  width: 28px;
  height: 28px;
  border: none;
  background: var(--color-error);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: all var(--transition-fast);
  flex-shrink: 0;
}

.remove-btn:hover {
  background: var(--color-error-dark);
  transform: scale(1.1);
}

.remove-btn svg {
  width: 14px;
  height: 14px;
}

/* 备注编辑区 */
.notes-section {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  padding: var(--spacing-2xl);
  margin-bottom: var(--spacing-3xl);
  transition: all var(--transition-smooth);
}

.notes-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--spacing-lg);
}

.notes-header h3 {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0;
}

.status-badge {
  padding: var(--spacing-xs) var(--spacing-sm);
  border-radius: var(--radius-full);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
}

.status-badge.success {
  background: var(--color-success-light);
  color: var(--color-success);
}

.status-badge.pending {
  background: var(--color-warning-light);
  color: var(--color-warning);
}

.notes-textarea {
  width: 100%;
  height: 120px;
  padding: var(--spacing-md);
  border: 1px solid var(--color-gray-200);
  border-radius: var(--radius-lg);
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  resize: vertical;
  transition: all var(--transition-smooth);
}

.notes-textarea:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.notes-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: var(--spacing-md);
}

.notes-hint {
  font-size: var(--text-xs);
  color: var(--color-secondary);
  margin: 0;
}

.format-btn {
  padding: var(--spacing-xs) var(--spacing-md);
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: var(--radius-lg);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
  cursor: pointer;
  transition: all var(--transition-smooth);
}

.format-btn:hover {
  background: var(--color-primary-dark);
}

/* 配置选项 */
.config-options {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  padding: var(--spacing-2xl);
  margin-bottom: var(--spacing-3xl);
}

.config-options h3 {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-xl) 0;
}

.option-group {
  margin-bottom: var(--spacing-2xl);
}

.option-group:last-child {
  margin-bottom: 0;
}

.option-label {
  display: block;
  font-size: var(--text-base);
  font-weight: var(--font-medium);
  color: var(--color-gray-900);
  margin-bottom: var(--spacing-md);
}

.radio-group {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.radio-option {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg);
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: all var(--transition-smooth);
}

.radio-option:hover {
  border-color: var(--color-primary);
  background: var(--color-primary-lightest);
}

.radio-option input[type='radio'] {
  margin: 0;
}

.radio-option span {
  font-size: var(--text-sm);
  color: var(--color-gray-700);
  line-height: var(--leading-relaxed);
}

/* 上传每页音频 */
.slides-audio-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm);
}
.slide-audio-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-md);
  border: 1px solid var(--color-gray-200);
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.8);
}
.slide-audio-left {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}
.slide-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: var(--radius-full);
  background: var(--color-gray-100);
  color: var(--color-gray-700);
  font-size: var(--text-xs);
}
.slide-audio-name {
  color: var(--color-gray-900);
  font-size: var(--text-sm);
}
.slide-audio-missing {
  color: var(--color-warning-dark);
  font-size: var(--text-sm);
}
.slide-audio-right {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}
.hidden-file {
  display: none;
}
.pick-audio-btn {
  padding: 8px 12px;
  border: 1px solid var(--color-gray-300);
  background: white;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.pick-audio-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

/* 选择页样式 */
.slides-grid {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
}

.slide-chip {
  padding: 6px 10px;
  border-radius: var(--radius-full);
  border: 1px solid var(--color-gray-300);
  background: white;
  color: var(--color-gray-800);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.slide-chip.selected {
  border-color: var(--color-primary);
  background: var(--color-primary-lightest);
  color: var(--color-primary);
  box-shadow: var(--shadow-sm);
}

.slides-hint {
  margin-top: var(--spacing-sm);
  font-size: var(--text-xs);
  color: var(--color-secondary);
}

/* 操作区域 */
.action-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
  align-items: center;
}

.generate-btn {
  width: 100%;
  max-width: 400px;
  padding: var(--spacing-xl) var(--spacing-3xl);
  border: none;
  border-radius: var(--radius-xl);
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  cursor: pointer;
  transition: all var(--transition-smooth);
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 56px;
}

.generate-btn:not(.can-generate) {
  background: var(--color-gray-200);
  color: var(--color-secondary);
  cursor: not-allowed;
}

.generate-btn.can-generate {
  background: var(--gradient-primary);
  color: white;
  box-shadow: var(--shadow-primary);
}

.generate-btn.can-generate:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-primary-hover);
}

.generate-btn.is-generating {
  background: var(--color-secondary);
  color: white;
  cursor: not-allowed;
}

.btn-content {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.btn-content svg {
  width: 20px;
  height: 20px;
}

.loading-content {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 进度条 */
.progress-section {
  width: 100%;
  max-width: 400px;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: var(--color-gray-200);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--gradient-primary);
  border-radius: var(--radius-full);
  transition: width var(--transition-smooth);
}

.progress-text {
  font-size: var(--text-sm);
  color: var(--color-secondary);
  text-align: center;
  margin: 0;
}

/* 需求提示 */
.requirements-hint {
  width: 100%;
  max-width: 400px;
  padding: var(--spacing-lg);
  background: var(--color-warning-light);
  border: 1px solid var(--color-warning);
  border-radius: var(--radius-lg);
}

.requirements-hint p {
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  color: var(--color-warning-dark);
  margin: 0 0 var(--spacing-md) 0;
}

.requirements-hint ul {
  margin: 0;
  padding-left: var(--spacing-xl);
  color: var(--color-warning-dark);
}

.requirements-hint li {
  font-size: var(--text-sm);
  margin-bottom: var(--spacing-xs);
}

/* 预览区域 */
.preview-section {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-2xl);
}

.preview-card {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  overflow: hidden;
  transition: all var(--transition-smooth);
}

.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-xl);
  background: var(--color-gray-50);
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.preview-header h3 {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0;
}

.preview-status {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-md);
  background: var(--color-success-light);
  border: 1px solid var(--color-success);
  border-radius: var(--radius-full);
  color: var(--color-success);
  font-size: var(--text-xs);
  font-weight: var(--font-medium);
}

.status-dot {
  width: 6px;
  height: 6px;
  background: var(--color-success);
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.preview-content {
  padding: var(--spacing-2xl);
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.empty-preview {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-lg);
  color: var(--color-secondary);
}

.empty-icon {
  width: 64px;
  height: 64px;
  opacity: 0.5;
}

.empty-preview p {
  font-size: var(--text-base);
  margin: 0;
}

.image-preview-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-lg);
}

.preview-label {
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  color: var(--color-secondary);
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-md);
}

/* 提示卡片 */
.tips-card {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  overflow: hidden;
}

.tips-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl);
  background: var(--color-gray-50);
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.tips-icon {
  width: 20px;
  height: 20px;
  color: var(--color-primary);
}

.tips-header h4 {
  font-size: var(--text-base);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0;
}

.tips-list {
  padding: var(--spacing-xl);
  margin: 0;
  list-style: none;
}

.tips-list li {
  font-size: var(--text-sm);
  color: var(--color-gray-700);
  line-height: var(--leading-relaxed);
  margin-bottom: var(--spacing-md);
  padding-left: var(--spacing-xl);
  position: relative;
}

.tips-list li:last-child {
  margin-bottom: 0;
}

.tips-list li::before {
  content: '•';
  position: absolute;
  left: 0;
  color: var(--color-primary);
  font-weight: var(--font-bold);
}

/* 快速操作 */
.quick-actions {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-xl);
  padding: var(--spacing-xl);
}

.quick-actions h3 {
  font-size: var(--text-lg);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-lg) 0;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.action-btn {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-md) var(--spacing-lg);
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.06);
  border-radius: var(--radius-lg);
  color: var(--color-gray-700);
  text-decoration: none;
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  transition: all var(--transition-smooth);
}

.action-btn:hover {
  border-color: var(--color-primary);
  color: var(--color-primary);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.action-btn.secondary {
  background: var(--color-gray-50);
}

.action-btn svg {
  width: 16px;
  height: 16px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .content-grid {
    grid-template-columns: 1fr;
    gap: var(--spacing-3xl);
  }

  .upload-grid {
    grid-template-columns: 1fr;
  }

  .status-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .video-config-view {
    padding: var(--spacing-2xl);
  }

  .steps-wrapper {
    flex-direction: column;
    gap: var(--spacing-lg);
  }

  .step-connector {
    display: none;
  }

  .step-item {
    flex-direction: row;
    text-align: left;
  }

  .step-number {
    margin-bottom: 0;
    margin-right: var(--spacing-md);
  }

  .generate-btn {
    font-size: var(--text-base);
  }
}
</style>
