<template>
  <div class="modern-video-generator">
    <!-- 步骤指示器 -->
    <div class="steps-indicator">
      <div class="steps-container">
        <div
          v-for="(step, index) in steps"
          :key="index"
          :class="[
            'step-item',
            {
              active: currentStep > index,
              current: currentStep === index + 1,
            },
          ]"
        >
          <div class="step-connector" v-if="index < steps.length - 1"></div>
          <div class="step-circle">
            <svg v-if="currentStep > index" class="check-icon" viewBox="0 0 24 24" fill="none">
              <path
                d="M20 6L9 17L4 12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <span v-else class="step-number">{{ index + 1 }}</span>
          </div>
          <div class="step-info">
            <h4 class="step-title">{{ step.title }}</h4>
            <p class="step-description">{{ step.description }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 配置状态检查 -->
    <div class="config-status">
      <h3>配置检查</h3>
      <div class="status-items">
        <div class="status-item" :class="{ configured: digitalHumanStore.isPersonConfigured }">
          <div class="status-icon">
            <svg v-if="digitalHumanStore.isPersonConfigured" viewBox="0 0 24 24" fill="none">
              <path
                d="M20 6L9 17L4 12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <svg v-else viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" />
              <path
                d="M12 8V12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12 16H12.01"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </div>
          <div class="status-info">
            <h4>数字人形象</h4>
            <p>
              {{
                digitalHumanStore.isPersonConfigured ? '已配置' : '未配置 - 请前往数字人页面设置'
              }}
            </p>
          </div>
        </div>

        <div class="status-item" :class="{ configured: digitalHumanStore.isVoiceConfigured }">
          <div class="status-icon">
            <svg v-if="digitalHumanStore.isVoiceConfigured" viewBox="0 0 24 24" fill="none">
              <path
                d="M20 6L9 17L4 12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <svg v-else viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" />
              <path
                d="M12 8V12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12 16H12.01"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </div>
          <div class="status-info">
            <h4>语音模型</h4>
            <p>
              {{ digitalHumanStore.isVoiceConfigured ? '已配置' : '未配置 - 请前往数字人页面设置' }}
            </p>
          </div>
        </div>

        <!-- 调试信息 -->
        <div class="status-item" :class="{ configured: !!imageFile }">
          <div class="status-icon">
            <span>{{ !!imageFile ? '✓' : '✗' }}</span>
          </div>
          <div class="status-info">
            <h4>图片文件</h4>
            <p>{{ !!imageFile ? '已上传' : '未上传' }}</p>
          </div>
        </div>

        <div class="status-item" :class="{ configured: !!pptFile }">
          <div class="status-icon">
            <span>{{ !!pptFile ? '✓' : '✗' }}</span>
          </div>
          <div class="status-info">
            <h4>PPT文件</h4>
            <p>{{ !!pptFile ? '已上传' : '未上传' }}</p>
          </div>
        </div>

        <div class="status-item" :class="{ configured: pptRemakes.trim().length > 0 }">
          <div class="status-icon">
            <span>{{ pptRemakes.trim().length > 0 ? '✓' : '✗' }}</span>
          </div>
          <div class="status-info">
            <h4>PPT备注内容</h4>
            <p>{{ pptRemakes.trim().length > 0 ? '已解析' : '未解析' }}</p>
          </div>
        </div>
      </div>

      <!-- PPT备注内容显示 -->
      <div class="ppt-notes-input" style="margin-top: 16px">
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #1a202c">
          PPT备注内容 {{ pptRemakes.trim().length > 0 ? '✅ 已自动解析' : '⏳ 等待上传PPT' }}
        </h4>
        <textarea
          v-model="pptRemakes"
          placeholder="PPT备注内容将在这里显示，您可以手动编辑..."
          style="
            width: 100%;
            height: 120px;
            padding: 8px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            resize: vertical;
            font-size: 12px;
            font-family: monospace;
          "
        ></textarea>
        <div
          style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 4px;
          "
        >
          <p style="margin: 0; font-size: 12px; color: #64748b">
            💡 上传PPT后会自动解析备注内容，您也可以手动编辑JSON格式的备注
          </p>
          <button
            v-if="pptRemakes.trim().length > 0"
            @click="formatPPTRemakes"
            style="
              padding: 4px 8px;
              font-size: 11px;
              background: #667eea;
              color: white;
              border: none;
              border-radius: 4px;
              cursor: pointer;
            "
          >
            格式化JSON
          </button>
        </div>
      </div>

      <div
        style="
          margin-top: 16px;
          padding: 12px;
          background: #f0f9ff;
          border-radius: 8px;
          border: 1px solid #bae6fd;
        "
      >
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #0284c7">调试信息</h4>
        <p style="margin: 4px 0; font-size: 12px; color: #666">
          canGenerate: {{ canGenerate ? '✓ 可以生成' : '✗ 无法生成' }}
        </p>
        <p style="margin: 4px 0; font-size: 12px; color: #666">
          图片: {{ !!imageFile }} | PPT: {{ !!pptFile }} | 备注:
          {{ pptRemakes.trim().length > 0 }} | 数字人: {{ digitalHumanStore.isPersonConfigured }} |
          语音: {{ digitalHumanStore.isVoiceConfigured }}
        </p>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="content-layout">
      <!-- 左侧：上传和配置区域 -->
      <div class="upload-section">
        <div class="upload-cards">
          <!-- 照片上传卡片 -->
          <div class="upload-card">
            <div class="card-header">
              <div class="card-icon">
                <svg viewBox="0 0 24 24" fill="none">
                  <rect
                    x="3"
                    y="3"
                    width="18"
                    height="18"
                    rx="2"
                    stroke="currentColor"
                    stroke-width="2"
                  />
                  <circle cx="12" cy="10" r="3" stroke="currentColor" stroke-width="2" />
                  <path
                    d="M12 14C7.59 14 4 17.59 4 22H20C20 17.59 16.41 14 12 14Z"
                    stroke="currentColor"
                    stroke-width="2"
                  />
                </svg>
              </div>
              <div class="card-title">
                <h3>数字人照片</h3>
                <p>上传清晰的正面人脸照片</p>
              </div>
            </div>

            <div
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

              <div
                v-if="!imagePreview"
                class="upload-placeholder"
                @click="$refs.imageInput.click()"
              >
                <div class="upload-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M21 15V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M17 8L12 3L7 8"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M12 3V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
                <p>点击或拖拽上传照片</p>
                <span>支持 JPG、PNG 格式，建议正面清晰照片</span>
              </div>

              <div v-else class="image-preview-container">
                <img :src="imagePreview" alt="预览" class="preview-image" />
                <button class="remove-btn" @click="removeImage">
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
            <!-- 图像预处理：显示一个清晰的列表选择 -->
            <div v-if="imagePreview" class="img-preprocess-panel">
              <div class="pre-row" style="justify-content:space-between;">
                <div style="display:flex; align-items:center; gap:8px;">
                  <label class="pre-label">卡通化</label>
                  <label class="switch">
                    <input type="checkbox" v-model="cartoonizeEnabled" />
                    <span class="slider"></span>
                  </label>
                </div>
                <div v-if="cartoonizeEnabled">
                  <button class="style-btn" type="button" @click="styleListVisible = !styleListVisible">
                    {{ styleLabelMap[cartoonBackendStyle] }} ▼
                  </button>
                </div>
              </div>
              <transition name="fade">
                <div v-if="cartoonizeEnabled && styleListVisible" class="style-list-panel">
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
              <div class="preview-note" v-if="cartoonizeEnabled">
                上传后将调用后端 AnimeGAN 进行卡通化，无前端预处理。
              </div>
            </div>
          </div>

          <!-- PPT上传卡片 -->
          <div class="upload-card">
            <div class="card-header">
              <div class="card-icon ppt">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M14 2V8H20"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M9 15L15 15M9 11L15 11"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </div>
              <div class="card-title">
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
                <div class="upload-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M21 15V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M17 8L12 3L7 8"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M12 3V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
                <p>点击或拖拽上传PPT文件</p>
                <span>支持 PPT、PPTX 格式，确保每页都有批注内容</span>
              </div>

              <div v-else class="file-preview-container">
                <div class="file-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M14 2V8H20"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
                <div class="file-info">
                  <span class="file-name">{{ pptFile.name }}</span>
                  <span class="file-size">{{ formatFileSize(pptFile.size) }}</span>
                </div>
                <button class="remove-btn" @click="removePPT">
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
          </div>

          <!-- 视频上传卡片 -->
          <div class="upload-card">
            <div class="card-header">
              <div class="card-icon video">
                <svg viewBox="0 0 24 24" fill="none">
                  <path
                    d="M22 8H2C0.89543 8 0 7.10457 8 8V16C0 17.1046 0.89543 18 2 18H22C23.1046 18 24 17.1046 24 16V10C24 8.89543 23.1046 8 22 8H8"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M7 12L10 15L17 8"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
              </div>
              <div class="card-title">
                <h3>上传视频</h3>
                <p>上传已生成的视频文件 (可选)</p>
              </div>
            </div>

            <div
              class="upload-area"
              :class="{ 'has-file': videoFile, 'is-dragover': dragOverVideo }"
              @drop="handleVideoDrop"
              @dragover.prevent="dragOverVideo = true"
              @dragleave="dragOverVideo = false"
            >
              <input
                ref="videoInput"
                type="file"
                accept="video/*"
                @change="handleVideoSelect"
                style="display: none"
              />

              <div v-if="!videoFile" class="upload-placeholder" @click="$refs.videoInput.click()">
                <div class="upload-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M21 15V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M17 8L12 3L7 8"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M12 3V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
                <p>点击或拖拽上传视频文件</p>
                <span>支持 MP4、AVI、MOV 等格式</span>
              </div>

              <div v-else class="file-preview-container">
                <div class="file-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M22 8H2C0.89543 8 0 7.10457 8 8V16C0 17.1046 0.89543 18 2 18H22C23.1046 18 24 17.1046 24 16V10C24 8.89543 23.1046 8 22 8H8"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M7 12L10 15L17 8"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                </div>
                <div class="file-info">
                  <span class="file-name">{{ videoFile.name }}</span>
                  <span class="file-size">{{ formatFileSize(videoFile.size) }}</span>
                </div>
                <button class="remove-btn" @click="removeVideo">
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
          </div>
        </div>

        <!-- 生成按钮 -->
        <div class="action-section">
          <button
            class="generate-btn"
            :class="{
              'can-generate': canGenerate,
              'is-generating': isGenerating,
            }"
            :disabled="!canGenerate || isGenerating"
            @click="generateVideo"
          >
            <div class="btn-content">
              <svg v-if="!isGenerating" class="btn-icon" viewBox="0 0 24 24" fill="none">
                <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
              </svg>
              <div v-else class="loading-spinner">
                <svg class="spinner" viewBox="0 0 24 24">
                  <circle
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    stroke-width="2"
                    fill="none"
                    stroke-dasharray="31.416"
                    stroke-dashoffset="31.416"
                  >
                    <animate
                      attributeName="stroke-dasharray"
                      dur="2s"
                      values="0 31.416;15.708 15.708;0 31.416"
                      repeatCount="indefinite"
                    />
                    <animateTransform
                      attributeName="transform"
                      type="rotate"
                      dur="2s"
                      values="0 12 12;360 12 12"
                      repeatCount="indefinite"
                    />
                  </circle>
                </svg>
              </div>
              <span class="btn-text">{{ isGenerating ? '生成中...' : '开始生成视频' }}</span>
            </div>
          </button>

          <div v-if="isGenerating" class="progress-info">
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
            </div>
            <span class="progress-text">{{ currentStepText }}</span>
          </div>
        </div>
      </div>

      <!-- 右侧：预览和结果区域 -->
      <div class="preview-section">
        <div class="preview-card">
          <div class="card-header">
            <h3>预览区域</h3>
            <div class="status-badge" :class="{ online: true }">
              <div class="status-dot"></div>
              <span>就绪</span>
            </div>
          </div>

          <div class="preview-content">
            <div v-if="!videoGenerated && !imagePreview" class="empty-preview">
              <div class="empty-icon">
                <svg viewBox="0 0 24 24" fill="none">
                  <rect
                    x="2"
                    y="2"
                    width="20"
                    height="20"
                    rx="2.18"
                    stroke="currentColor"
                    stroke-width="2"
                  />
                  <path
                    d="M7 2V20M17 2V20M2 12H20M2 7H7M17 7H22M2 17H7M17 17H22"
                    stroke="currentColor"
                    stroke-width="2"
                  />
                </svg>
              </div>
              <p>上传照片后将显示预览</p>
            </div>

            <div v-else-if="imagePreview && !videoGenerated" class="image-preview-wrapper">
                          <div class="preview-label">原始照片</div>
                          <div :class="['anime-frame', animeFrameEnabled ? animeFrameStyle : '']">
                            <img :src="imagePreview" alt="预览" class="preview-img" />
                          </div>
            </div>

            <div v-else-if="videoGenerated" class="video-preview-wrapper">
              <div class="preview-label">生成结果</div>
              <div class="video-placeholder">
                <div class="video-icon">
                  <svg viewBox="0 0 24 24" fill="none">
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                  </svg>
                </div>
                <p>视频生成完成</p>
                <button class="download-btn" @click="downloadVideo">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path
                      d="M21 15V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V15"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M7 10L12 15L17 10"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M12 15V3"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </svg>
                  <span>下载视频</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 提示信息 -->
        <div class="tips-card">
          <div class="tips-header">
            <svg class="tips-icon" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" />
              <path
                d="M12 16V12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12 8H12.01"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
            <h4>使用提示</h4>
          </div>
          <ul class="tips-list">
            <li>确保照片为正面人脸，光线充足</li>
            <li>PPT文件每页都需要有批注内容</li>
            <li>生成过程可能需要几分钟，请耐心等待</li>
            <li>建议使用Chrome或Edge浏览器获得最佳体验</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <!-- 风格选择弹窗 -->
  <div v-if="showStylePicker" class="style-picker-overlay" @click="showStylePicker = false">
    <div class="style-picker" @click.stop>
      <div class="style-picker-header">选择 AnimeGAN 风格 (v2 / v3)</div>
      <div class="cartoonize-toggle-row">
        <label class="cartoonize-toggle-label">启用卡通化</label>
        <label class="switch">
          <input type="checkbox" v-model="cartoonizeEnabled" />
          <span class="slider"></span>
        </label>
      </div>
      <div class="style-picker-list">
        <div
          v-for="key in ['hayao','shinkai','paprika','celeba','animeganv3_paprika','animeganv3_hayao','animeganv3_shinkai']"
          :key="key"
          class="style-item"
          :class="{ active: cartoonBackendStyle === (key as any) }"
          @click="cartoonBackendStyle = key as any; showStylePicker = false"
        >
          <span>{{ styleLabelMap[key] }}</span>
          <span v-if="cartoonBackendStyle === key" style="color:#667eea">✓</span>
        </div>
      </div>
      <div class="frame-picker-header">二次元形象框</div>
      <div class="frame-toggle-row">
        <label class="frame-toggle-label">启用形象框</label>
        <label class="switch">
          <input type="checkbox" v-model="animeFrameEnabled" />
          <span class="slider"></span>
        </label>
      </div>
      <div v-if="animeFrameEnabled" class="frame-style-list">
        <div
          v-for="f in ['panel','glow','film']"
          :key="f"
          class="frame-style-item"
          :class="{ active: animeFrameStyle === f }"
          @click="animeFrameStyle = f as any"
        >
          <span>{{ frameStyleLabelMap[f] }}</span>
          <span v-if="animeFrameStyle === f" style="color:#667eea">✓</span>
        </div>
      </div>
    </div>
  </div>
  <!-- 是否卡通化 弹窗 -->
  <div v-if="showCartoonizePrompt" class="cartoonize-prompt-overlay" @click="showCartoonizePrompt = false">
    <div class="cartoonize-prompt" @click.stop>
      <div class="prompt-title">是否对图片进行卡通化？</div>
      <div class="prompt-actions">
        <button class="btn-secondary" @click="showCartoonizePrompt = false; cartoonizeEnabled = false">不启用</button>
        <button class="btn-primary" @click="showCartoonizePrompt = false; cartoonizeEnabled = true; showStylePicker = true">启用并选择风格</button>
      </div>
    </div>
  </div>
</template>
<script setup lang="ts">
import { ref, computed, nextTick } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import { digitalHumanApi } from '@/services/api'

const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

// Template refs
const imageInput = ref<HTMLInputElement>()
const pptInput = ref<HTMLInputElement>()
const videoInput = ref<HTMLInputElement>()

// 当前步骤 (1~4)
const currentStep = ref(1)
// 拖拽高亮状态
const dragOverImage = ref(false)
const dragOverPPT = ref(false)
const dragOverVideo = ref(false)
const imageFile = ref<File | null>(null)
const imagePreview = ref('')
const pptFile = ref<File | null>(null)
const videoFile = ref<File | null>(null)
const pptRemakes = ref('')
const isGenerating = ref(false)
const videoGenerated = ref(false)
const lastError = ref<string | null>(null)

// 步骤配置
const steps = [
  {
    title: '上传照片',
    description: '上传清晰的正面人脸照片',
  },
  {
    title: '上传PPT',
    description: '上传包含批注的PPT文件',
  },
  {
    title: '配置参数',
    description: '设置数字人表现参数',
  },
  {
    title: '生成视频',
    description: 'AI自动生成数字人视频',
  },
]

// 当前步骤文本
const currentStepText = computed(() => {
  const texts = ['准备中...', '上传图片中', '上传PPT中', '配置参数中', '生成视频中']
  return texts[currentStep.value] || texts[0]
})

const config = computed({
  get: () => digitalHumanStore.config,
  set: (value) => digitalHumanStore.setConfig(value),
})

// 图像预处理绑定（使用全局设置，两个页面保持一致）
const cartoonizeEnabled = computed<boolean>({
  get: () => digitalHumanStore.cartoonizeEnabled,
  set: (v) => digitalHumanStore.setCartoonizeEnabled(v),
})
// 后端风格（与形象管理页面共享全局设置）
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
const styleLabelMap: Record<string,string> = {
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
// Anime frame overlay bindings
const animeFrameEnabled = computed<boolean>({
  get: () => digitalHumanStore.animeFrameEnabled,
  set: (v) => digitalHumanStore.setAnimeFrameEnabled(v),
})
const animeFrameStyle = computed<'panel'|'glow'|'film'>({
  get: () => digitalHumanStore.animeFrameStyle,
  set: (v) => digitalHumanStore.setAnimeFrameStyle(v),
})
const frameStyleLabelMap: Record<string,string> = {
  panel: '面板描边',
  glow: '霓虹光效',
  film: '赛璐璐边框',
}
const showStylePicker = ref(false)
const showCartoonizePrompt = ref(false)
const styleListVisible = ref(true)
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

const canGenerate = computed(
  () =>
    !!(
      imageFile.value &&
      pptFile.value &&
      pptRemakes.value.trim() &&
      digitalHumanStore.isPersonConfigured &&
      digitalHumanStore.isVoiceConfigured
    ),
)

// 添加视频生成计算属性
const canGenerateWithVideo = computed(
  () =>
    !!(
      videoFile.value &&
      digitalHumanStore.isPersonConfigured &&
      digitalHumanStore.isVoiceConfigured
    ),
)

// 进度百分比（用于顶部进度条填充）
const progressPercent = computed(() => ((currentStep.value - 1) / 3) * 100)

const handleImageSelect = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    imageFile.value = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.value = e.target?.result as string
    }
    reader.readAsDataURL(file)
    digitalHumanStore.setUploadedFile('image', file)
    currentStep.value = 2
    // DOM 更新后再打开弹窗，避免渲染时机问题
    await nextTick()
    // 上传图片后直接弹出是否卡通化
    showCartoonizePrompt.value = true
  }
}

const handleImageDrop = async (event: DragEvent) => {
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
    currentStep.value = 2
    await nextTick()
    showCartoonizePrompt.value = true
  }
}

const removeImage = () => {
  imageFile.value = null
  imagePreview.value = ''
  digitalHumanStore.setUploadedFile('image', null)
  currentStep.value = 1
}

const removePPT = () => {
  pptFile.value = null
  digitalHumanStore.setUploadedFile('ppt', null)
  if (!imageFile.value && !videoFile.value) {
    currentStep.value = 1
  }
}

const handleVideoSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    videoFile.value = file
    digitalHumanStore.setUploadedFile('video', file)
    currentStep.value = 3
  }
}

const handleVideoDrop = (event: DragEvent) => {
  dragOverVideo.value = false
  const file = event.dataTransfer?.files[0]
  if (file && file.type.startsWith('video/')) {
    videoFile.value = file
    digitalHumanStore.setUploadedFile('video', file)
    currentStep.value = 3
  }
}

const removeVideo = () => {
  videoFile.value = null
  digitalHumanStore.setUploadedFile('video', null)
  if (!imageFile.value && !pptFile.value) {
    currentStep.value = 1
  }
}

const handlePPTSelect = async (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    pptFile.value = file
    digitalHumanStore.setUploadedFile('ppt', file)

    // 解析PPT备注内容
    await parsePPTRemakes(file)

    currentStep.value = 3
  }
}

const handlePPTDrop = async (event: DragEvent) => {
  dragOverPPT.value = false
  const file = event.dataTransfer?.files[0]
  if (file && (file.name.endsWith('.ppt') || file.name.endsWith('.pptx'))) {
    pptFile.value = file
    digitalHumanStore.setUploadedFile('ppt', file)

    // 解析PPT备注内容
    await parsePPTRemakes(file)

    currentStep.value = 3
  }
}

const generateVideo = async () => {
  if (!canGenerate.value || !authStore.currentUser) return

  // 检查配置状态 - 类似C#项目的检查逻辑
  if (!digitalHumanStore.isPersonConfigured) {
    alert('请先在数字人页面配置数字人形象')
    return
  }

  if (!digitalHumanStore.isVoiceConfigured) {
    alert('请先在数字人页面配置语音模型')
    return
  }

  isGenerating.value = true
  videoGenerated.value = false

  try {
    // Step 1: Upload image
    if (imageFile.value) {
      await digitalHumanStore.uploadImage(authStore.currentUser, imageFile.value)
    }

    // Step 2: Send PPT remakes
    let pptRemakesToSend = pptRemakes.value
    // 如果是字符串且能解析为对象，则转为对象
    try {
      if (typeof pptRemakesToSend === 'string') {
        const parsed = JSON.parse(pptRemakesToSend)
        pptRemakesToSend = parsed
      }
    } catch (e) {
      // 保持原样，后端会做兜底
      console.warn('PPT备注不是合法JSON，原样发送', e)
    }
    await digitalHumanStore.sendPPTRemakes(authStore.currentUser, pptRemakesToSend)

    // Step 3: Send configuration
    await digitalHumanStore.sendConfig(authStore.currentUser)

    // Step 4: Generate video
    const success = await digitalHumanStore.generateDigitalHuman(authStore.currentUser)
    if (success) {
      videoGenerated.value = true
      currentStep.value = 4
    }
  } catch (error) {
    const msg = error instanceof Error ? error.message : '未知错误'
    console.error('[视频生成失败]', msg, error)
    lastError.value = msg
  } finally {
    isGenerating.value = false
  }
}

// PPT备注解析函数
const parsePPTRemakes = async (file: File) => {
  if (!authStore.currentUser) {
    alert('请先登录')
    return
  }

  try {
    console.log('开始解析PPT备注...')

    // 使用后端接口解析PPT
    const result = await digitalHumanApi.uploadPPTParseRemakes(
      { User: authStore.currentUser },
      file
    )

    if (result.success && result.data) {
      // 仅在输入框为空时自动填充解析结果，否则保留用户手动输入内容
      if (!pptRemakes.value.trim()) {
        pptRemakes.value = JSON.stringify(result.data.remakes, null, 2)
        console.log('PPT备注解析完成:', pptRemakes.value)
        alert(`PPT备注解析完成！\n\n成功解析到 ${result.data.parsed_count} 页备注内容`)
      } else {
        alert('PPT备注已填写，未自动覆盖。可手动修改或清空后重新上传PPT自动解析。')
      }
    } else {
      // 如果解析失败，提供一个基本的示例内容
      console.error('PPT解析失败:', result.error)
      if (!pptRemakes.value.trim()) {
        const fallbackRemakes = {
          'Slide 1': '这是一个测试PPT的演讲内容',
          'Slide 2': '系统会自动根据备注生成语音',
        }
        pptRemakes.value = JSON.stringify(fallbackRemakes, null, 2)
        alert(`PPT自动解析遇到问题：${result.error}\n已使用示例内容，您也可以在下方文本框中手动修改备注内容。`)
      } else {
        alert('PPT备注已填写，未自动覆盖。可手动修改或清空后重新上传PPT自动解析。')
      }
    }
  } catch (error) {
    console.error('PPT解析失败:', error)

    // 如果解析失败，提供一个基本的示例内容
    const fallbackRemakes = {
      'Slide 1': '这是一个测试PPT的演讲内容',
      'Slide 2': '系统会自动根据备注生成语音',
    }

    pptRemakes.value = JSON.stringify(fallbackRemakes, null, 2)
    alert('PPT自动解析遇到问题，已使用示例内容。\n您也可以在下方文本框中手动修改备注内容。')
  }
}

// 格式化PPT备注JSON
const formatPPTRemakes = () => {
  try {
    const parsed = JSON.parse(pptRemakes.value)
    pptRemakes.value = JSON.stringify(parsed, null, 2)
    alert('JSON格式化完成！')
  } catch (error) {
    alert('JSON格式错误，请检查格式')
  }
}

const downloadVideo = async () => {
  if (!authStore.currentUser) return

  try {
    const blob = await digitalHumanStore.downloadVideo(authStore.currentUser)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `数字人视频_${new Date().toISOString().split('T')[0]}.mp4`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('下载失败:', error)
    alert('视频下载失败')
  }
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<style scoped>
/* 现代视频生成器样式 */
.modern-video-generator {
  padding: 32px;
  background: #fafbfc;
  min-height: 500px;
}

/* 步骤指示器 */
.steps-indicator {
  margin-bottom: 40px;
}

.steps-container {
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
  background: #e2e8f0;
  z-index: -1;
}

.step-item.active ~ .step-connector {
  background: #667eea;
}

.step-circle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 12px;
  border: 2px solid transparent;
  transition: all 0.3s ease;
}

.step-item.active .step-circle,
.step-item.current .step-circle {
  background: #667eea;
  border-color: #667eea;
  color: white;
}

.step-number {
  font-weight: 600;
  font-size: 14px;
  color: #64748b;
}

.step-item.active .step-number,
.step-item.current .step-number {
  color: white;
}

.check-icon {
  width: 20px;
  height: 20px;
  color: white;
}

.step-info {
  text-align: center;
  max-width: 120px;
}

.step-title {
  font-size: 14px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 4px 0;
  line-height: 1.3;
}

.step-description {
  font-size: 12px;
  color: #64748b;
  margin: 0;
  line-height: 1.4;
}

/* 配置状态检查 */
.config-status {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 32px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  border: 1px solid #e2e8f0;
}

.config-status h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1a202c;
}

.status-items {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  transition: all 0.3s ease;
}

.status-item.configured {
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.status-icon {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.status-item:not(.configured) .status-icon {
  background: #fef3c7;
  color: #f59e0b;
}

.status-item.configured .status-icon {
  background: #dcfce7;
  color: #22c55e;
}

.status-icon svg {
  width: 20px;
  height: 20px;
}

.status-info h4 {
  margin: 0 0 4px 0;
  font-size: 14px;
  font-weight: 600;
  color: #1a202c;
}

.status-info p {
  margin: 0;
  font-size: 12px;
  color: #64748b;
  line-height: 1.4;
}

/* 主要内容布局 */
.content-layout {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 32px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 上传区域 */
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.upload-cards {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.upload-card {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 20px;
  border: 1px solid rgba(226, 232, 240, 0.8);
  box-shadow:
    0 4px 20px rgba(0, 0, 0, 0.06),
    0 0 0 1px rgba(255, 255, 255, 0.5) inset;
  overflow: hidden;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(10px);
  position: relative;
}

.upload-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent,
    rgba(102, 126, 234, 0.4),
    rgba(255, 107, 157, 0.4),
    transparent
  );
  opacity: 0;
  transition: opacity 0.3s ease;
}

.upload-card:hover {
  box-shadow:
    0 8px 30px rgba(0, 0, 0, 0.12),
    0 0 0 1px rgba(102, 126, 234, 0.1) inset;
  border-color: rgba(102, 126, 234, 0.3);
  transform: translateY(-2px);
}

.upload-card:hover::before {
  opacity: 1;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.card-icon {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.card-icon.ppt {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
}

.card-icon svg {
  width: 20px;
  height: 20px;
}

.card-title h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 4px 0;
}

.card-title p {
  font-size: 13px;
  color: #64748b;
  margin: 0;
}

/* 上传区域 */
.upload-area {
  padding: 20px;
  border: 2px dashed #e2e8f0;
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
}

.upload-area:hover {
  border-color: #667eea;
  background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
}

.upload-area.is-dragover {
  border-color: #667eea;
  background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
  transform: translateY(-3px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
  animation: dragPulse 1.5s ease-in-out infinite;
}

@keyframes dragPulse {
  0%, 100% {
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.25);
  }
  50% {
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.35);
  }
}

.upload-area.has-image,
.upload-area.has-file {
  border-color: #667eea;
  border-style: solid;
  background: #f8faff;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.upload-icon {
  width: 48px;
  height: 48px;
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

/* 图片预览 */
.image-preview-container {
  position: relative;
  display: inline-block;
}

.preview-image {
  max-width: 100%;
  max-height: 200px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
/* 预处理面板样式 */
.img-preprocess-panel {
  margin-top: 12px;
  padding: 14px 16px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.pre-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.pre-label {
  font-size: 13px;
  font-weight: 600;
  color: #334155;
  flex: 0 0 60px;
}
.mini-label {
  font-size: 12px;
  color: #475569;
  flex: 0 0 56px;
}
.mini-value {
  font-size: 12px;
  color: #64748b;
  min-width: 42px;
  text-align: right;
}
.img-preprocess-panel input[type='range'] {
  flex: 1;
}
.preview-note {
  font-size: 11px;
  color: #64748b;
  margin-top: 2px;
}
/* 风格按钮 */
.style-btn {
  padding: 6px 10px;
  background:#fff;
  border:1px solid #e2e8f0;
  border-radius:6px;
  font-size:12px;
  cursor:pointer;
  color:#334155;
}
.style-btn:hover { background:#f1f5f9; }

/* Inline style list panel */
.style-list-panel { padding: 8px 8px 10px; background:#fff; border:1px solid #e5e7eb; border-radius:10px; }
.style-list-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; }
.style-list-item { display:flex; align-items:center; justify-content:space-between; padding:8px 10px; border:1px solid #e5e7eb; border-radius:8px; cursor:pointer; font-size:12px; color:#374151; }
.style-list-item:hover { background:#f9fafb; }
.style-list-item.active { background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }
.style-name { }
.style-check { color:#667eea; font-weight:600; }
.style-hint { margin-top:6px; font-size:11px; color:#64748b; }

.fade-enter-active,.fade-leave-active { transition: opacity .2s ease; }
.fade-enter-from,.fade-leave-to { opacity: 0; }

.style-picker-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.45); display:flex; align-items:center; justify-content:center; z-index:1200; }
.style-picker { background:#fff; border-radius:12px; width:300px; box-shadow:0 20px 40px rgba(0,0,0,0.25); overflow:hidden; }
.style-picker-header { padding:12px 16px; border-bottom:1px solid #e2e8f0; font-weight:600; font-size:14px; }
.style-picker-list { padding:4px 8px 12px; }
.style-item { display:flex; justify-content:space-between; align-items:center; padding:8px 10px; border-radius:8px; cursor:pointer; font-size:13px; }
.style-item:hover { background:#f8fafc; }
.style-item.active { background:#eef2ff; border:1px solid #c7d2fe; }
/* Cartoonize toggle row in picker */
.cartoonize-toggle-row { display:flex; align-items:center; justify-content:space-between; padding:8px 12px; border-bottom:1px solid #e2e8f0; }
.cartoonize-toggle-label { font-size:12px; color:#374151; }
/* Cartoonize prompt modal */
.cartoonize-prompt-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.5); display:flex; align-items:center; justify-content:center; z-index:1300; }
.cartoonize-prompt { background:#fff; width:320px; border-radius:12px; box-shadow:0 20px 40px rgba(0,0,0,0.25); padding:16px; }
.prompt-title { font-size:14px; font-weight:600; color:#111827; margin-bottom:12px; }
.prompt-actions { display:flex; gap:10px; justify-content:flex-end; }
.btn-secondary { padding:8px 12px; border:1px solid #e5e7eb; background:#fff; border-radius:8px; font-size:13px; cursor:pointer; }
.btn-primary { padding:8px 12px; border:1px solid #6366f1; background:#6366f1; color:#fff; border-radius:8px; font-size:13px; cursor:pointer; }
/* 开关复用 */
.switch {
  position: relative;
  display: inline-block;
  width: 42px;
  height: 22px;
}
.switch input { opacity:0; width:0; height:0; }
.switch .slider {
  position: absolute;
  inset:0;
  background:#cbd5e1;
  border-radius:22px;
  cursor:pointer;
  transition:.3s;
}
.switch .slider:before {
  content:'';
  position:absolute;
  width:16px; height:16px;
  left:3px; top:3px;
  background:#fff;
  border-radius:50%;
  box-shadow:0 2px 4px rgba(0,0,0,.15);
  transition:.3s;
}
.switch input:checked + .slider { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); }
.switch input:checked + .slider:before { transform: translateX(20px); }

/* 文件预览 */
.file-preview-container {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
}

.file-icon {
  width: 40px;
  height: 40px;
  background: #f59e0b;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.file-icon svg {
  width: 20px;
  height: 20px;
}

.file-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  color: #1a202c;
  margin-bottom: 2px;
}

.file-size {
  font-size: 12px;
  color: #64748b;
}

/* 删除按钮 */
.remove-btn {
  width: 24px;
  height: 24px;
  border: none;
  background: #ef4444;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
}

.remove-btn:hover {
  background: #dc2626;
  transform: scale(1.1);
}

.remove-btn svg {
  width: 14px;
  height: 14px;
}

/* 操作区域 */
.action-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.generate-btn {
  width: 100%;
  padding: 16px 24px;
  border: none;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  min-height: 56px;
}

.generate-btn:not(.can-generate) {
  background: #e2e8f0;
  color: #94a3b8;
  cursor: not-allowed;
}

.generate-btn.can-generate {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.generate-btn.can-generate:hover {
  transform: translateY(-3px) scale(1.02);
  box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}

.generate-btn.can-generate:active {
  transform: translateY(-1px) scale(1.01);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
  transition: all 0.1s ease;
}

.generate-btn.is-generating {
  background: #94a3b8;
  color: white;
  cursor: not-allowed;
}

.btn-content {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-icon {
  width: 20px;
  height: 20px;
}

.loading-spinner .spinner {
  width: 20px;
  height: 20px;
  color: white;
}

/* 进度信息 */
.progress-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: #e2e8f0;
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 13px;
  color: #64748b;
  text-align: center;
}

/* 预览区域 */
.preview-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.preview-card {
  background: white;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.preview-card .card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.preview-card .card-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 20px;
  color: #166534;
  font-size: 12px;
  font-weight: 500;
}

.status-badge.online {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #166534;
}

.status-dot {
  width: 6px;
  height: 6px;
  background: #22c55e;
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
  padding: 20px;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.empty-preview {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: #94a3b8;
}

.empty-icon {
  width: 64px;
  height: 64px;
  opacity: 0.5;
}

.empty-preview p {
  font-size: 14px;
  margin: 0;
}

.image-preview-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.preview-label {
  font-size: 14px;
  font-weight: 500;
  color: #64748b;
}

.preview-img {
  max-width: 100%;
  max-height: 200px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
/* Anime frame overlay styles */
.anime-frame {
  padding: 6px;
  border-radius: 10px;
  transition: box-shadow .25s ease, border-color .25s ease;
}
.anime-frame.panel {
  border: 2px solid #6366f1;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.15) inset, 0 6px 14px rgba(99,102,241,0.2);
}
.anime-frame.glow {
  border: 1px solid rgba(147,197,253,0.6);
  box-shadow: 0 0 18px rgba(59,130,246,0.45), 0 0 42px rgba(147,197,253,0.25);
  background: radial-gradient(circle at 30% 30%, rgba(99,102,241,0.12), transparent 70%);
}
.anime-frame.film {
  border: 2px dashed #94a3b8;
  background: repeating-linear-gradient(
    -45deg,
    rgba(148,163,184,0.1) 0 6px,
    rgba(255,255,255,0.0) 6px 12px
  );
}

.video-preview-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  text-align: center;
}

.video-icon {
  width: 48px;
  height: 48px;
  color: #667eea;
}

.video-placeholder p {
  font-size: 16px;
  font-weight: 500;
  color: #1a202c;
  margin: 0;
}

.download-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.download-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.download-btn svg {
  width: 16px;
  height: 16px;
}

/* 提示卡片 */
.tips-card {
  background: white;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.tips-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
}

.tips-icon {
  width: 20px;
  height: 20px;
  color: #3b82f6;
}

.tips-header h4 {
  font-size: 15px;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
}

.tips-list {
  padding: 16px;
  margin: 0;
  list-style: none;
}

.tips-list li {
  font-size: 13px;
  color: #475569;
  line-height: 1.6;
  margin-bottom: 8px;
  padding-left: 20px;
  position: relative;
}

.tips-list li:before {
  content: '•';
  position: absolute;
  left: 0;
  color: #3b82f6;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .content-layout {
    grid-template-columns: 1fr;
    gap: 24px;
  }

  .modern-video-generator {
    padding: 24px;
  }

  .steps-container {
    padding: 0 20px;
  }

  .step-info {
    max-width: 100px;
  }

  .step-title {
    font-size: 13px;
  }

  .step-description {
    font-size: 11px;
  }
}

@media (max-width: 768px) {
  .modern-video-generator {
    padding: 16px;
  }

  .steps-container {
    gap: 16px;
  }

  .step-circle {
    width: 32px;
    height: 32px;
  }

  .step-number {
    font-size: 12px;
  }

  .step-title {
    font-size: 12px;
  }

  .step-description {
    font-size: 10px;
  }

  .card-header {
    padding: 16px;
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .upload-area {
    padding: 16px;
  }

  .generate-btn {
    font-size: 15px;
    padding: 14px 20px;
  }
}
.progress-bar-wrapper {
  position: relative;
  margin: 0 auto 36px;
  max-width: 840px;
  padding: 12px 12px 28px;
}
.progress-track {
  height: 6px;
  background: #e5e7ef;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}
.progress-fill {
  height: 100%;
  background: var(--gradient-primary);
  width: 0;
  transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.3) inset;
}
.progress-nodes {
  display: flex;
  justify-content: space-between;
  position: absolute;
  left: 0;
  right: 0;
  top: 0;
  pointer-events: none;
}
.progress-node {
  text-align: center;
  width: 25%;
  position: relative;
}
.progress-node .dot {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: block;
  margin: 0 auto 4px;
  background: #cbd5e1;
  transition:
    background 0.4s,
    transform 0.4s;
}
.progress-node.active .dot {
  background: var(--color-primary);
  transform: scale(1.15);
}
.progress-node .num {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-weight: 500;
}
.progress-node.active .num {
  color: var(--color-primary);
}
.upload-area {
  transition:
    background 0.35s,
    border-color 0.35s,
    transform 0.35s;
}
.upload-area.is-dragover {
  border-color: var(--color-primary);
  background: rgba(99, 102, 241, 0.08);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}
.form-section {
  transition:
    box-shadow 0.35s,
    transform 0.35s;
}
.form-section:hover {
  box-shadow: var(--shadow-sm);
}
.download-section {
  position: relative;
  overflow: hidden;
}
.download-section::before {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.6), transparent 70%);
  opacity: 0.6;
}
.generate-button {
  position: relative;
}
.generate-button:disabled::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: repeating-linear-gradient(
    45deg,
    rgba(255, 255, 255, 0.18) 0 12px,
    transparent 12px 24px
  );
  animation: sheen 1.2s linear infinite;
  mix-blend-mode: overlay;
}
@keyframes sheen {
  0% {
    background-position: 0 0;
  }
  100% {
    background-position: 240px 0;
  }
}
@media (max-width: 640px) {
  .workflow-steps {
    flex-wrap: wrap;
    gap: 20px;
  }
  .progress-bar-wrapper {
    padding-bottom: 38px;
  }
  .progress-node .num {
    display: none;
  }
}
.section-header h2 {
  color: var(--color-text);
}
.section-header p {
  color: var(--color-text-secondary);
}
/* === End Enhanced Theme === */
.video-generator {
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

.workflow-steps {
  display: flex;
  justify-content: space-between;
  margin-bottom: 40px;
  position: relative;
}

.workflow-steps::before {
  content: '';
  position: absolute;
  top: 25px;
  left: 50px;
  right: 50px;
  height: 2px;
  background: #e0e0e0;
  z-index: 0;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
  z-index: 1;
}

.step-number {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin-bottom: 10px;
  transition: all 0.3s;
}

.step.active .step-number {
  background: #667eea;
  color: white;
}

.step-content {
  text-align: center;
}

.step-content h3 {
  font-size: 16px;
  margin-bottom: 5px;
}

.step-content p {
  font-size: 14px;
  color: #666;
}

.form-section {
  margin-bottom: 30px;
  padding: 25px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.form-section h3 {
  margin-bottom: 15px;
  color: #333;
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

.file-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.file-icon {
  font-size: 36px;
}

.file-details {
  text-align: left;
}

.file-name {
  font-weight: 600;
  display: block;
}

.file-size {
  font-size: 14px;
  color: #666;
}

.remove-btn {
  position: absolute;
  top: -10px;
  right: -10px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  cursor: pointer;
  font-size: 16px;
}

textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-family: inherit;
  resize: vertical;
}

.hint {
  font-size: 14px;
  color: #666;
  display: block;
  margin-top: 5px;
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
}

.toggle-switch {
  display: flex;
  align-items: center;
  gap: 10px;
}

.range-input {
  display: flex;
  align-items: center;
  gap: 10px;
}

.action-section {
  text-align: center;
  padding: 30px;
}

.generate-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 15px 40px;
  border-radius: 8px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.generate-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.generate-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.generation-info {
  color: #666;
  margin-top: 10px;
}

.download-section {
  text-align: center;
  padding: 30px;
  background: #e8f5e8;
  border-radius: 12px;
}

.download-button {
  background: #27ae60;
  color: white;
  border: none;
  padding: 12px 30px;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
}

.download-button:hover {
  background: #219a52;
}
.error-box {
  margin-top: 24px;
  padding: 16px 18px;
  border-radius: 10px;
  background: #fff5f5;
  border: 1px solid #fecaca;
  color: #b91c1c;
  font-size: 14px;
  line-height: 1.5;
}
.error-box strong {
  font-weight: 600;
}
.error-hint {
  margin-top: 4px;
  color: #9b1c1c;
  font-size: 12px;
}
.task-indicator {
  margin-top: 18px;
  font-size: 14px;
  color: #4a5568;
  display: inline-block;
  background: #edf2f7;
  padding: 8px 14px;
  border-radius: 20px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06) inset;
}
</style>
