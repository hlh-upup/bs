<template>
  <div class="person-manager">
    <!-- 顶部标题区域 -->
    <div class="manager-header">
      <div class="header-content">
        <div class="title-section">
          <h2>数字人形象管理</h2>
          <p>管理和切换您的数字人形象配置</p>
        </div>
        <button class="add-person-btn" @click="showCreateDialog = true">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="M12 5v14M5 12h14"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
          <span>新建形象</span>
        </button>
      </div>
    </div>

    <!-- 形象卡片列表 -->
    <div class="persons-grid" v-if="personProfiles.length > 0">
      <div
        v-for="profile in personProfiles"
        :key="profile.id"
        :class="['person-card', { active: profile.id === activeProfileId }]"
        @click="selectProfile(profile.id)"
      >
        <div class="card-avatar">
          <div class="avatar-container" :class="{ 'has-image': profile.avatarUrl }">
            <img v-if="profile.avatarUrl" :src="profile.avatarUrl" alt="形象预览" />
            <div v-else class="avatar-placeholder">
              <svg viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="8" r="3" stroke="currentColor" stroke-width="2" />
                <path
                  d="M12 14C8.13 14 5 17.13 5 21H19C19 17.13 15.87 14 12 14Z"
                  stroke="currentColor"
                  stroke-width="2"
                />
              </svg>
            </div>
          </div>
          <div v-if="profile.id === activeProfileId" class="active-badge">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M20 6L9 17L4 12"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </div>
        </div>

        <div class="card-content">
          <h3 class="profile-name">{{ profile.name }}</h3>
          <p class="profile-description">{{ profile.description || '暂无描述' }}</p>

          <div class="profile-meta">
            <div class="meta-item">
              <span class="meta-label">声音模型:</span>
              <span class="meta-value">{{ getVoiceModelName(profile.voiceModel) }}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">创建时间:</span>
              <span class="meta-value">{{ formatDate(profile.createdAt) }}</span>
            </div>
          </div>
        </div>

        <div class="card-actions">
          <button class="action-btn edit-btn" @click.stop="editProfile(profile)">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M11 4H4C3.46957 4 2.96086 4.21071 2.58579 4.58579C2.21071 4.96086 2 5.46957 2 6V20C2 20.5304 2.21071 21.0391 2.58579 21.4142C2.96086 21.7893 3.46957 22 4 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V11"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M18.5 2.5C18.8978 2.10217 19.4474 1.87868 20.0303 1.87868C20.6132 1.87868 21.1628 2.10217 21.5607 2.5C21.9585 2.89782 22.182 3.44738 22.182 4.0303C22.182 4.61321 21.9585 5.16277 21.5607 5.56065L12.5607 14.5607C12.2804 14.841 11.9189 14.9651 11.5803 14.9053L9 14.4142L9.49475 16.9239C9.55458 17.2625 9.43045 17.6241 9.15014 17.9043L7.15014 19.9043C6.77014 20.2843 6.49991 20.1146 6.58579 19.5858C6.67166 19.057 6.84239 18.7868 7.22239 18.4068L9.22239 16.4068C9.5027 16.1265 9.86425 16.0024 10.2028 16.0622L12.7929 16.5533C13.1315 16.6131 13.4931 16.7384 13.7734 16.4581L18.5 11.7315V2.5Z"
                stroke="currentColor"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </button>
          <button class="action-btn delete-btn" @click.stop="deleteProfile(profile.id)">
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M3 6H5H21M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z"
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

    <!-- 空状态 -->
    <div v-else class="empty-state">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="8" r="3" stroke="currentColor" stroke-width="2" />
          <path
            d="M12 14C8.13 14 5 17.13 5 21H19C19 17.13 15.87 14 12 14Z"
            stroke="currentColor"
            stroke-width="2"
          />
        </svg>
      </div>
      <h3>还没有创建数字人形象</h3>
      <p>创建您的第一个数字人形象，开始体验个性化数字人服务</p>
      <button class="create-first-btn" @click="showCreateDialog = true">
        <svg viewBox="0 0 24 24" fill="none">
          <path
            d="M12 5v14M5 12h14"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <span>创建第一个形象</span>
      </button>
    </div>

    <!-- 创建/编辑对话框 -->
    <div v-if="showCreateDialog || showEditDialog" class="dialog-overlay" @click="closeDialog">
      <div class="dialog" @click.stop>
        <div class="dialog-header">
          <h3>{{ showEditDialog ? '编辑形象' : '创建新形象' }}</h3>
          <button class="close-btn" @click="closeDialog">
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

        <div class="dialog-content">
          <div class="form-section">
            <label class="form-label">形象名称 *</label>
            <input
              v-model="formData.name"
              type="text"
              class="form-input"
              placeholder="请输入形象名称"
              maxlength="20"
            />
          </div>

          <div class="form-section">
            <label class="form-label">形象描述</label>
            <textarea
              v-model="formData.description"
              class="form-textarea"
              placeholder="请输入形象描述（可选）"
              rows="3"
              maxlength="200"
            ></textarea>
          </div>

          <div class="form-section">
            <label class="form-label">形象照片 *</label>
            <div class="image-upload">
              <div
                class="upload-area"
                :class="{ 'has-image': formData.imagePreview }"
                @click="$refs.imageInput.click()"
              >
                <input
                  ref="imageInput"
                  type="file"
                  accept="image/*"
                  @change="handleImageSelect"
                  style="display: none"
                />

                <div v-if="!formData.imagePreview" class="upload-placeholder">
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
                  <p>点击或拖拽上传照片</p>
                  <span>支持 JPG、PNG 格式</span>
                </div>

                <div v-else class="image-preview">
                  <img :src="formData.imagePreview" alt="预览" />
                  <div class="image-overlay">
                    <button class="remove-image-btn" @click.stop="removeImage">
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

              <!-- 图像预处理（AnimeGANv2 后端：风格 + 预览） -->
              <div class="pixelate-panel" v-if="formData.imagePreview">
                <div class="pixel-row">
                  <label class="form-label">启用卡通化（AnimeGANv2 后端）</label>
                  <label class="switch">
                    <input type="checkbox" v-model="cartoonizeEnabled" />
                    <span class="slider"></span>
                  </label>
                </div>

                <div class="pixel-row" v-if="cartoonizeEnabled">
                  <div class="slider-wrapper" style="align-items:center; gap:8px;">
                    <span class="slider-value" style="min-width:72px; text-align:left">风格</span>
                    <button class="style-btn" type="button" @click.stop="showStylePicker = true">
                      {{ styleLabelMap[cartoonBackendStyle] }}
                    </button>
                    <button class="preview-btn" type="button" @click.stop="generateCartoonPreview" :disabled="previewLoading || !formData.imagePreview">
                      {{ previewLoading ? '生成中…' : '生成预览' }}
                    </button>
                    <button v-if="previewImg" class="preview-reset-btn" type="button" @click.stop="previewImg = ''">清除预览</button>
                  </div>
                </div>

                <div class="preview-row" v-if="previewImg || previewError">
                  <div class="preview-box" :class="{ loading: previewLoading }">
                    <template v-if="previewImg">
                      <img :src="previewImg" alt="卡通化预览" />
                    </template>
                    <div class="loading-mask" v-if="previewLoading">后端处理中…</div>
                    <div v-if="!previewLoading && previewError" class="preview-error">{{ previewError }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="form-section">
            <label class="form-label">声音模型</label>
              <select v-model="formData.voiceModel" class="form-select">
                <option value="man">男生</option>
                <option value="girl">女生</option>
              </select>
          </div>

          <div class="form-section" v-if="formData.voiceModel === 'custom'">
            <label class="form-label">自定义声音文件</label>
            <input type="file" accept="audio/*" @change="handleVoiceSelect" class="form-input" />
            <p class="form-hint">上传 WAV 格式的参考音频文件</p>
          </div>
        </div>

        <div class="dialog-footer">
          <button class="cancel-btn" @click="closeDialog">取消</button>
          <button class="submit-btn" @click="saveProfile" :disabled="!isFormValid">
            {{ showEditDialog ? '保存修改' : '创建形象' }}
          </button>
        </div>
      </div>
      <!-- 风格选择弹窗 -->
      <div v-if="showStylePicker" class="style-picker-overlay" @click="showStylePicker = false">
        <div class="style-picker" @click.stop>
          <div class="style-picker-header">选择 AnimeGAN 风格 (v2 / v3)</div>
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
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import { digitalHumanApi } from '@/services/api'

interface PersonProfile {
  id: string
  name: string
  description: string
  avatarUrl: string
  voiceModel: string
  voiceFile?: File
  createdAt: string
  updatedAt: string
}

const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

// 响应式数据
const personProfiles = ref<PersonProfile[]>([])
const activeProfileId = ref<string>('')
const showCreateDialog = ref(false)
const showEditDialog = ref(false)
const editingProfile = ref<PersonProfile | null>(null)

// 表单数据
const formData = ref({
  name: '',
  description: '',
  imagePreview: '',
  voiceModel: 'default',
  voiceFile: null as File | null,
})

// 计算属性
const isFormValid = computed(() => {
  return formData.value.name.trim() !== '' && formData.value.imagePreview !== ''
})

// 生命周期
onMounted(() => {
  loadProfiles()
})

// 方法
const loadProfiles = async () => {
  if (!authStore.currentUser) return

  try {
    // 从localStorage加载用户配置
    const storageKey = `personProfiles_${authStore.currentUser}`
    const savedProfiles = localStorage.getItem(storageKey)

    if (savedProfiles) {
      personProfiles.value = JSON.parse(savedProfiles)

      // 加载当前激活的配置
      const activeKey = `activeProfile_${authStore.currentUser}`
      const activeProfile = localStorage.getItem(activeKey)
      if (activeProfile) {
        activeProfileId.value = JSON.parse(activeProfile).id
      }
    } else {
      // 创建默认配置
      createDefaultProfile()
    }
  } catch (error) {
    console.error('加载用户配置失败:', error)
    createDefaultProfile()
  }
}

const createDefaultProfile = () => {
  const defaultProfile: PersonProfile = {
    id: 'default',
    name: '默认形象',
    description: '系统默认数字人形象',
    avatarUrl: '',
    voiceModel: 'default',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }

  personProfiles.value = [defaultProfile]
  activeProfileId.value = defaultProfile.id
  saveProfilesToStorage()
}

const saveProfilesToStorage = () => {
  if (!authStore.currentUser) return

  try {
    const storageKey = `personProfiles_${authStore.currentUser}`
    localStorage.setItem(storageKey, JSON.stringify(personProfiles.value))

    const activeKey = `activeProfile_${authStore.currentUser}`
    const activeProfile = personProfiles.value.find((p) => p.id === activeProfileId.value)
    if (activeProfile) {
      localStorage.setItem(activeKey, JSON.stringify(activeProfile))
    }
  } catch (error) {
    console.error('保存用户配置失败:', error)
  }
}

const selectProfile = (profileId: string) => {
  activeProfileId.value = profileId
  const profile = personProfiles.value.find((p) => p.id === profileId)
  if (profile) {
    // 更新全局store中的配置
    // 这里需要调用API更新实际的后端配置
    updateGlobalConfig(profile)

    // 标记配置完成（用于生成页的前置校验）
    if (profile.avatarUrl) digitalHumanStore.setPersonConfigured(true)
    if (profile.voiceModel) digitalHumanStore.setVoiceConfigured(true)
  }
  saveProfilesToStorage()
}

const updateGlobalConfig = async (profile: PersonProfile) => {
  try {
    // voiceModel 映射为 index
  let index = 0;
  if (profile.voiceModel === 'girl') index = 1;
  // 同步设置 config.modelIndex，确保 Send_Config 发出正确的 model_index
  digitalHumanStore.config.modelIndex = String(index);
  // 调用 selectVITSModel 接口同步后端权重，补全 User 和 Index 字段
  const { selectVITSModel } = await import('@/services/api');
  await selectVITSModel({ User: authStore.currentUser, Index: String(index) });
  console.log('已切换后端权重:', profile.voiceModel);
  } catch (error) {
    console.error('更新配置失败:', error);
  }
}

const editProfile = (profile: PersonProfile) => {
  editingProfile.value = profile
  formData.value = {
    name: profile.name,
    description: profile.description,
    imagePreview: profile.avatarUrl,
    voiceModel: profile.voiceModel,
    voiceFile: null,
  }
  showEditDialog.value = true
}

const deleteProfile = (profileId: string) => {
  if (profileId === 'default') {
    alert('默认形象不能删除')
    return
  }

  if (confirm('确定要删除这个数字人形象吗？')) {
    personProfiles.value = personProfiles.value.filter((p) => p.id !== profileId)

    if (activeProfileId.value === profileId) {
      activeProfileId.value = personProfiles.value[0]?.id || 'default'
    }

    saveProfilesToStorage()
  }
}

const handleImageSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader()
    reader.onload = (e) => {
      formData.value.imagePreview = e.target?.result as string
    }
    reader.readAsDataURL(file)
  }
}

const handleVoiceSelect = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    formData.value.voiceFile = file
  }
}

const removeImage = () => {
  formData.value.imagePreview = ''
  if (showEditDialog.value && editingProfile.value) {
    // 编辑模式下，移除图片但不清空name
    editingProfile.value.avatarUrl = ''
  }
}

const saveProfile = async () => {
  if (!authStore.currentUser) return

  try {
    const profileData: PersonProfile = {
      id:
        showEditDialog.value && editingProfile.value
          ? editingProfile.value.id
          : Date.now().toString(),
      name: formData.value.name.trim(),
      description: formData.value.description.trim(),
      avatarUrl: formData.value.imagePreview,
      voiceModel: formData.value.voiceModel,
      createdAt:
        showEditDialog.value && editingProfile.value
          ? editingProfile.value.createdAt
          : new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }

    if (showEditDialog.value && editingProfile.value) {
      // 编辑现有配置
      const index = personProfiles.value.findIndex((p) => p.id === editingProfile.value!.id)
      if (index !== -1) {
        personProfiles.value[index] = profileData
      }
    } else {
      // 创建新配置
      personProfiles.value.push(profileData)
      // 自动选择新创建的配置
      activeProfileId.value = profileData.id
    }

    saveProfilesToStorage()
    closeDialog()
    alert(showEditDialog.value ? '形象更新成功！' : '形象创建成功！')
  } catch (error) {
    console.error('保存形象失败:', error)
    alert('保存失败，请重试')
  }
}

const closeDialog = () => {
  showCreateDialog.value = false
  showEditDialog.value = false
  editingProfile.value = null

  // 重置表单
  formData.value = {
    name: '',
    description: '',
    imagePreview: '',
    voiceModel: 'default',
    voiceFile: null,
  }
}

const getVoiceModelName = (model: string) => {
  const models: { [key: string]: string } = {
    man: '男生',
    girl: '女生',
  }
  return models[model] || model
}

const formatDate = (dateString: string) => {
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN')
}

// 卡通化参数（后端 AnimeGANv2 开关 + 风格）
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

// 风格选择弹窗
const showStylePicker = ref(false)
const styleLabelMap: Record<string, string> = {
  hayao: '宫崎骏 v2',
  shinkai: '新海诚 v2',
  paprika: '今敏 v2',
  celeba: '人像 v2',
  animeganv3_paprika: '今敏 Paprika v3',
  paprika_v3: '今敏 Paprika v3',
  animeganv3_hayao: '宫崎骏 v3',
  hayao_v3: '宫崎骏 v3',
  animeganv3_shinkai: '新海诚 v3',
  shinkai_v3: '新海诚 v3',
}

// 后端预览状态
const previewImg = ref<string>('')
const previewLoading = ref(false)
const previewError = ref<string>('')

const generateCartoonPreview = async () => {
  previewError.value = ''
  if (!formData.value.imagePreview) {
    previewError.value = '请先上传图片'
    return
  }
  if (!authStore.currentUser) {
    previewError.value = '请先登录后再预览'
    return
  }
  try {
    previewLoading.value = true
    const raw = (formData.value.imagePreview.split(',')[1]) || ''
    const res = await digitalHumanApi.cartoonizeImage({
      User: authStore.currentUser,
      Img: raw,
      Mode: 'animegan_v2',
      Style: cartoonBackendStyle.value,
      Params: { max_side: 1200 },
    })
    if (res.success && res.img) {
      previewImg.value = 'data:image/png;base64,' + res.img
      // 可选：控制台打印调试信息
      console.info('[Cartoon Preview] ok', { debug: res.debug, style: res.style_used })
    } else {
      previewError.value = res.error || '后端返回失败'
    }
  } catch (e:any) {
    previewError.value = e?.message || '请求异常'
  } finally {
    previewLoading.value = false
  }
}
</script>

<style scoped>
.person-manager {
  padding: 32px;
  min-height: 600px;
}

.manager-header {
  margin-bottom: 32px;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  /* 去掉页面级最大宽度限制，跟随右侧内容区满幅铺开 */
  max-width: none;
  width: 100%;
  margin: 0;
}

.title-section h2 {
  font-size: 28px;
  font-weight: 700;
  color: #1a202c;
  margin: 0 0 8px 0;
}

.title-section p {
  font-size: 16px;
  color: #64748b;
  margin: 0;
}

.add-person-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.add-person-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.add-person-btn svg {
  width: 20px;
  height: 20px;
}

.persons-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
  /* 移除居中与最大宽度限制，使用外层容器控制留白 */
  max-width: none;
  width: 100%;
  margin: 0;
}

.person-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  border: 2px solid #e2e8f0;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.person-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.person-card.active {
  border-color: #667eea;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
}

.card-avatar {
  position: relative;
  margin-bottom: 16px;
}

.avatar-container {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: #f1f5f9;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border: 3px solid #e2e8f0;
  transition: all 0.3s ease;
}

.person-card.active .avatar-container {
  border-color: #667eea;
  box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
}

.avatar-container.has-image {
  background: transparent;
}

.avatar-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.avatar-placeholder {
  color: #94a3b8;
}

.avatar-placeholder svg {
  width: 32px;
  height: 32px;
}

.active-badge {
  position: absolute;
  bottom: 0;
  right: 0;
  width: 24px;
  height: 24px;
  background: #22c55e;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid white;
  color: white;
}

.active-badge svg {
  width: 14px;
  height: 14px;
}

.card-content {
  margin-bottom: 16px;
}

.profile-name {
  font-size: 18px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 4px 0;
}

.profile-description {
  font-size: 14px;
  color: #64748b;
  margin: 0 0 12px 0;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.profile-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.meta-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.meta-label {
  color: #64748b;
  font-weight: 500;
}

.meta-value {
  color: #1a202c;
  font-weight: 600;
}

.card-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.action-btn {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-btn svg {
  width: 16px;
  height: 16px;
}

.edit-btn {
  background: #f1f5f9;
  color: #475569;
}

.edit-btn:hover {
  background: #e2e8f0;
  color: #334155;
}

.delete-btn {
  background: #fef2f2;
  color: #dc2626;
}

.delete-btn:hover {
  background: #fecaca;
  color: #b91c1c;
}

.empty-state {
  text-align: center;
  padding: 64px 32px;
  max-width: 400px;
  margin: 0 auto;
}

.empty-icon {
  width: 80px;
  height: 80px;
  background: #f1f5f9;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 24px;
  color: #94a3b8;
}

.empty-icon svg {
  width: 40px;
  height: 40px;
}

.empty-state h3 {
  font-size: 20px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 8px 0;
}

.empty-state p {
  font-size: 16px;
  color: #64748b;
  margin: 0 0 32px 0;
  line-height: 1.5;
}

.create-first-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  margin: 0 auto;
}

.create-first-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.create-first-btn svg {
  width: 20px;
  height: 20px;
}

/* 对话框样式 */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
}

.dialog {
  background: white;
  border-radius: 16px;
  width: 90%;
  max-width: 500px;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 24px 0;
}

.dialog-header h3 {
  font-size: 20px;
  font-weight: 600;
  color: #1a202c;
  margin: 0;
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: #f1f5f9;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.close-btn:hover {
  background: #e2e8f0;
}

.close-btn svg {
  width: 16px;
  height: 16px;
  color: #64748b;
}

.dialog-content {
  padding: 24px;
}

.form-section {
  margin-bottom: 24px;
}

.form-label {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: #1a202c;
  margin-bottom: 8px;
}

.form-input,
.form-textarea,
.form-select {
  width: 100%;
  padding: 12px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s ease;
  background: white;
}

.form-input:focus,
.form-textarea:focus,
.form-select:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-textarea {
  resize: vertical;
  min-height: 80px;
}

.form-hint {
  font-size: 12px;
  color: #64748b;
  margin-top: 4px;
}

.image-upload {
  position: relative;
}

.upload-area {
  width: 100%;
  border: 2px dashed #e2e8f0;
  border-radius: 8px;
  padding: 32px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-area:hover {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.02);
}

.upload-area.has-image {
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

.image-preview {
  position: relative;
  display: inline-block;
}

.image-preview img {
  max-width: 200px;
  max-height: 200px;
  border-radius: 8px;
  object-fit: cover;
}

.image-overlay {
  position: absolute;
  top: -8px;
  right: -8px;
}

.remove-image-btn {
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

.remove-image-btn:hover {
  background: #dc2626;
}

.remove-image-btn svg {
  width: 12px;
  height: 12px;
}

/* 像素化面板 */
.pixelate-panel {
  margin-top: 16px;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #f8fafc;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.pixel-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.slider-wrapper {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 12px;
}

.slider-wrapper input[type='range'] {
  flex: 1;
}

.slider-value {
  font-size: 12px;
  color: #475569;
  min-width: 48px;
  text-align: right;
}

.preview-row {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.preview-box {
  position: relative;
  width: 200px;
  height: 200px;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  background: #fff;
}

.preview-box.loading {
  opacity: 0.7;
}

.preview-box img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.loading-mask {
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  color: #334155;
  backdrop-filter: blur(2px);
}

/* 风格选择按钮和弹窗 */
.style-btn {
  padding: 8px 12px;
  border: 1px solid #e2e8f0;
  background: #fff;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  color: #334155;
}

.preview-btn {
  padding: 8px 10px;
  border: 1px solid #c7d2fe;
  background: #eef2ff;
  color: #3730a3;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
}
.preview-btn:disabled { opacity: .6; cursor: not-allowed; }

.preview-reset-btn {
  padding: 6px 8px;
  border: 1px solid #e2e8f0;
  background: #fff;
  color: #475569;
  border-radius: 8px;
  cursor: pointer;
  font-size: 12px;
}

.preview-error { font-size: 12px; color: #b91c1c; padding: 6px; }

.style-picker-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
}

.style-picker {
  background: #fff;
  border-radius: 12px;
  width: 320px;
  box-shadow: 0 20px 40px rgba(0,0,0,0.2);
  overflow: hidden;
}

.style-picker-header {
  padding: 12px 16px;
  border-bottom: 1px solid #e2e8f0;
  font-weight: 600;
  color: #1a202c;
}

.style-picker-list {
  padding: 8px;
}

.style-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
}
.style-item:hover { background: #f8fafc; }
.style-item.active { background: #eef2ff; border: 1px solid #c7d2fe; }

/* 自定义开关样式 */
.switch {
  position: relative;
  display: inline-block;
  width: 46px;
  height: 24px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.switch .slider {
  position: absolute;
  cursor: pointer;
  inset: 0;
  background: #cbd5e1;
  transition: 0.3s;
  border-radius: 24px;
}

.switch .slider:before {
  position: absolute;
  content: '';
  height: 18px;
  width: 18px;
  left: 3px;
  top: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
  box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}

.switch input:checked + .slider {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.switch input:checked + .slider:before {
  transform: translateX(22px);
}

.dialog-footer {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  padding: 0 24px 24px;
}

.cancel-btn,
.submit-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.cancel-btn {
  background: #f1f5f9;
  color: #475569;
}

.cancel-btn:hover {
  background: #e2e8f0;
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .person-manager {
    padding: 16px;
  }

  .header-content {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }

  .persons-grid {
    grid-template-columns: 1fr;
  }

  .dialog {
    width: 95%;
    margin: 16px;
  }
}
</style>
