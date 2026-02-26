<template>
  <!-- 嵌入到 Dashboard 右侧内容区的纯内容页：不再包含自己的 AppHeader/主容器 -->
  <div class="video-list-view">
    <div class="main-content">
      <div class="video-container page-layout">
        <div class="page-main">
          <!-- 状态检查和操作引导 -->
          <div class="status-section">
            <h2>视频中心</h2>

            <div class="generation-status" :class="{ configured: canGenerateVideo }">
              <div class="status-indicator">
                <span class="status-icon">{{ canGenerateVideo ? '✅' : '⚠️' }}</span>
                <span class="status-text">生成准备状态</span>
              </div>
              <p class="status-description">
                {{ canGenerateVideo ? '所有配置已完成，可以开始生成视频' : '请先完成必要配置' }}
              </p>
              <button
                v-if="canGenerateVideo"
                @click="openActionPanel('generate')"
                class="generate-btn"
              >
                🎬 生成新视频
              </button>
            </div>
          </div>

          <!-- 已生成的视频列表 -->
          <div class="videos-section">
            <div class="section-header">
              <h3>我的视频</h3>
              <div class="action-buttons">
                <button @click="refreshVideoList" :disabled="isLoading" class="refresh-btn">
                  🔄 刷新
                </button>
                <button @click="openVideoManager" class="manager-btn">📁 文件管理</button>
              </div>
            </div>

            <!-- 加载骨架屏 -->
            <div v-if="isLoading" class="skeleton-grid">
              <div class="skeleton-card" v-for="n in 6" :key="n">
                <div class="skeleton-thumb shimmer"></div>
                <div class="skeleton-info">
                  <div class="skeleton-line shimmer" style="width: 80%"></div>
                  <div class="skeleton-meta">
                    <div class="skeleton-line shimmer" style="width: 50%"></div>
                    <div class="skeleton-line shimmer" style="width: 30%"></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- 视频网格 -->
            <div class="video-grid" v-else-if="videoList.length > 0">
              <div v-for="video in videoList" :key="video.id" class="video-item">
                <div class="video-thumbnail">
                  <div class="thumbnail-placeholder" v-if="!video.thumbnail">📹</div>
                  <img v-else :src="video.thumbnail" :alt="video.name" class="thumbnail-image" />
                  <div class="video-duration">{{ formatDuration(video.duration) }}</div>
                </div>

                <div class="video-info">
                  <div class="video-name" :title="video.name">{{ video.name }}</div>
                  <div class="video-meta">
                    <span class="create-time">{{ formatDate(video.createTime) }}</span>
                    <span class="file-size">{{ formatFileSize(video.size) }}</span>
                  </div>
                  <div class="video-actions">
                    <button
                      @click="playVideo(video)"
                      class="icon-btn play-btn"
                      title="播放视频"
                      aria-label="播放视频"
                    >
                      ▶️
                    </button>
                    <button
                      @click="downloadVideo(video)"
                      class="icon-btn download-btn"
                      title="下载视频"
                      aria-label="下载视频"
                    >
                      ⬇️
                    </button>
                    <button
                      @click="shareVideo(video)"
                      class="icon-btn share-btn"
                      title="分享视频"
                      aria-label="分享视频"
                    >
                      🔗
                    </button>
                    <button
                      @click="deleteVideo(video)"
                      class="icon-btn delete-btn"
                      title="删除视频"
                      aria-label="删除视频"
                      :disabled="video.isDeleting"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- 空状态 -->
            <div v-else class="empty-state">
              <div class="empty-illustration">📹</div>
              <h4>还没有生成的视频</h4>
              <p>生成你的第一个数字人视频，开始创作吧。</p>
              <button @click="openActionPanel('generate')" class="start-generate-btn">
                开始生成第一个视频
              </button>
            </div>
          </div>

          <!-- 视频播放器 -->
          <div v-if="selectedVideo" class="video-player-overlay" @click="closePlayer">
            <div class="video-player-container" @click.stop>
              <video
                ref="videoPlayerRef"
                :src="API_BASE_URL + selectedVideo.url"
                controls
                autoplay
                @ended="onVideoEnded"
              ></video>
              <div class="player-controls">
                <button class="close-player-btn" @click="closePlayer">✕</button>
                <div class="video-info-player">
                  <span>{{ selectedVideo.name }}</span>
                  <span>{{ formatDuration(selectedVideo.duration) }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 右侧操作面板 -->
        <aside class="page-aside" :class="{ open: isActionPanelOpen }">
          <div class="aside-header">
            <h4>操作面板</h4>
            <button class="aside-close" @click="isActionPanelOpen = false">✕</button>
          </div>

          <div class="aside-section">
            <div class="aside-status">
              <div class="aside-status-item" :class="{ ok: digitalHumanStore.isPersonConfigured }">
                <span class="dot"></span>
                <div>
                  <div class="title">数字人形象</div>
                  <div class="desc">
                    {{ digitalHumanStore.isPersonConfigured ? '已配置' : '未配置' }}
                  </div>
                </div>
                <button class="mini-btn" @click="router.push('/dashboard/person-manager')">
                  配置
                </button>
              </div>
              <div class="aside-status-item" :class="{ ok: digitalHumanStore.isVoiceConfigured }">
                <span class="dot"></span>
                <div>
                  <div class="title">语音模型</div>
                  <div class="desc">
                    {{ digitalHumanStore.isVoiceConfigured ? '已配置' : '未配置' }}
                  </div>
                </div>
                <button class="mini-btn" @click="router.push('/dashboard/advanced-config')">
                  配置
                </button>
              </div>
              <div class="aside-status-item" :class="{ ok: !!uploadedPPTName }">
                <span class="dot"></span>
                <div>
                  <div class="title">PPT备注</div>
                  <div class="desc">{{ uploadedPPTName || '未上传' }}</div>
                </div>
                <button class="mini-btn" @click="triggerPPTUpload">上传</button>
                <input
                  ref="pptInputRef"
                  type="file"
                  accept=".ppt,.pptx"
                  style="display: none"
                  @change="onPPTSelected"
                />
              </div>
            </div>
          </div>

          <div class="aside-section">
            <button
              class="primary-action"
              :disabled="!canGenerateVideo || !uploadedPPTName"
              @click="router.push('/dashboard/video-config')"
            >
              🎬 打开视频配置
            </button>
            <p class="aside-tip">需完成形象与语音配置并上传PPT后可开始生成。</p>
          </div>
        </aside>

        <button class="toggle-aside-btn" @click="isActionPanelOpen = !isActionPanelOpen">
          {{ isActionPanelOpen ? '→' : '操作' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import { digitalHumanApi } from '@/services/api'

const router = useRouter()
const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

// 类型定义
type VideoItem = {
  id: string
  name: string
  url: string
  duration: number
  size: number
  createTime: Date
  thumbnail?: string
  isDeleting?: boolean
}

// 响应式数据
const videoList = ref<VideoItem[]>([])

const selectedVideo = ref<VideoItem | null>(null)
const isLoading = ref(false)
const videoPlayerRef = ref<HTMLVideoElement>()
const isActionPanelOpen = ref(false)
const pptInputRef = ref<HTMLInputElement>()
const uploadedPPTName = ref<string>('')

// 计算属性
const canGenerateVideo = computed(() => {
  return (
    digitalHumanStore.isPersonConfigured &&
    digitalHumanStore.isVoiceConfigured &&
    digitalHumanStore.uploadedFiles.ppt !== null
  )
})

// 方法实现
const refreshVideoList = async () => {
  if (!authStore.currentUser) return

  isLoading.value = true
  try {
    // 获取后端视频列表
    const videos = await getBackendVideoList()
    videoList.value = videos

    // 生成本地预览缩略图
    for (const video of videos) {
      video.thumbnail = await generateVideoThumbnail(video)
    }

    // 保存到本地存储
    saveVideoListToStorage(videos)
  } catch (error) {
    console.error('获取视频列表失败:', error)
    alert('获取视频列表失败，请重试')
  } finally {
    isLoading.value = false
  }
}

const getBackendVideoList = async (): Promise<VideoItem[]> => {
  if (!authStore.currentUser) return []

  try {
    // 调用真实的后端API获取视频列表
    const response = await digitalHumanApi.getVideoList({ User: authStore.currentUser })

    // 转换后端数据格式为前端需要的格式
    return response.map(
      (video: any): VideoItem => ({
        id: video.id,
        name: video.name,
        url: video.url,
        duration: video.duration,
        size: video.size,
        createTime: new Date(video.createTime * 1000), // 转换时间戳为Date对象
      }),
    )
  } catch (error) {
    console.error('获取视频列表失败:', error)
    return []
  }
}

const getVideoStream = async (videoId: string) => {
  // 模拟从后端获取视频流
  // 实际实现需要调用后端API
  return `/api/video/stream/${videoId}`
}

const generateVideoThumbnail = async (video: VideoItem): Promise<string> => {
  // 使用Canvas生成视频缩略图
  return new Promise((resolve) => {
    const videoElement = document.createElement('video')
  videoElement.src = API_BASE_URL + video.url
    videoElement.crossOrigin = 'anonymous'
    videoElement.muted = true

    videoElement.addEventListener('loadeddata', () => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')

      canvas.width = 320
      canvas.height = 180

      // 跳转到第5秒作为缩略图
      videoElement.currentTime = Math.min(5, video.duration)

      setTimeout(() => {
        if (!ctx) {
          resolve('')
          return
        }
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height)

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              resolve('')
              return
            }
            const url = URL.createObjectURL(blob)
            resolve(url)
          },
          'image/jpeg',
          0.8,
        )
      }, 100)
    })

    videoElement.addEventListener('error', () => {
      resolve('') // 失败时返回空字符串
    })
  })
}

const playVideo = (video: VideoItem) => {
  selectedVideo.value = video
  // 等待DOM更新
  setTimeout(() => {
    videoPlayerRef.value?.play()
  }, 100)
}

import { API_BASE_URL } from '../services/api'

const downloadVideo = async (video: VideoItem) => {
  try {
    // 修正：下载时走后端端口
    const response = await fetch(API_BASE_URL + video.url)
    const blob = await response.blob()

    // 创建下载链接
    const downloadUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = video.name
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)

    URL.revokeObjectURL(downloadUrl)
  } catch (error) {
    console.error('下载失败:', error)
    alert('下载失败，请重试')
  }
}

const shareVideo = (video: VideoItem) => {
  // 分享功能
  if (navigator.share) {
    navigator
      .share({
        title: video.name,
        text: `查看我生成的数字人视频：${video.name}`,
        url: video.url,
      })
      .catch(() => {
        // 分享失败时的处理
        copyToClipboard(video.url)
      })
  } else {
    copyToClipboard(video.url)
  }
}

const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text)
    alert('视频链接已复制到剪贴板')
  } catch {
    console.error('复制到剪贴板失败')
  }
}

// 侧边面板逻辑
const openActionPanel = (tab?: string) => {
  isActionPanelOpen.value = true
  // 可根据 tab 做更多事，例如滚动到按钮
  setTimeout(() => {
    document.querySelector('.page-aside')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, 0)
}

const triggerPPTUpload = () => {
  pptInputRef.value?.click()
}

const onPPTSelected = async (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  uploadedPPTName.value = file.name
  // 可在此直接调用后端解析接口，或保存到 store
  try {
    if (authStore.currentUser) {
      await digitalHumanApi.uploadPPTParseRemakes({ User: authStore.currentUser }, file)
    }
  } catch (err) {
    console.warn('PPT上传/解析失败', err)
  }
}

const deleteVideo = async (video: VideoItem) => {
  if (!confirm(`确定要删除视频 "${video.name}" 吗？`)) {
    return
  }

  video.isDeleting = true

  try {
    // 调用真实的后端删除API
    await digitalHumanApi.deleteVideo({
      User: authStore.currentUser!,
      VideoId: video.id,
    })

    // 从列表中移除
    const index = videoList.value.findIndex((v) => v.id === video.id)
    if (index > -1) {
      videoList.value.splice(index, 1)
    }

    // 从本地存储中移除
    const videos = getVideosFromStorage()
    const updatedVideos = videos.filter((v: VideoItem) => v.id !== video.id)
    saveVideoListToStorage(updatedVideos)

    alert('视频已删除')
  } catch (error) {
    console.error('删除失败:', error)
    alert('删除失败，请重试')
  } finally {
    video.isDeleting = false
  }
}

const saveVideoListToStorage = (videos: typeof videoList.value) => {
  const videoData = {
    videos: videos.map((v) => ({
      id: v.id,
      name: v.name,
      url: v.url,
      duration: v.duration,
      size: v.size,
      createTime: v.createTime,
      thumbnail: v.thumbnail,
    })),
    lastUpdated: new Date().toISOString(),
  }
  localStorage.setItem('video_list', JSON.stringify(videoData))
}

const getVideosFromStorage = () => {
  const data = localStorage.getItem('video_list')
  if (data) {
    const parsed = JSON.parse(data)
    return parsed.videos || []
  }
  return []
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatDuration = (seconds: number) => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  } else if (minutes > 0) {
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  } else {
    return `0:${secs.toString().padStart(2, '0')}`
  }
}

const formatDate = (date: Date) => {
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

const closePlayer = () => {
  selectedVideo.value = null
}

const onVideoEnded = () => {
  // 视频播放结束后的处理
  console.log('视频播放结束')
}

const openVideoManager = () => {
  // 打开视频管理功能
  alert('视频管理功能开发中...')
}

// 生命周期
const onKeyDown = (e: KeyboardEvent) => {
  if (e.key === 'Escape' && selectedVideo.value) {
    closePlayer()
  }
}

onMounted(() => {
  loadVideoListFromStorage()
  window.addEventListener('keydown', onKeyDown)
  // 默认桌面端展开侧边栏
  if (window.innerWidth >= 1024) {
    isActionPanelOpen.value = true
  }
})

const loadVideoListFromStorage = () => {
  const videos = getVideosFromStorage()
  // 将存储中的时间字符串还原为 Date 对象，保证格式化函数可用
  videoList.value = videos.map((v: any) => ({
    ...v,
    createTime: v.createTime ? new Date(v.createTime) : new Date(),
  }))
}

onUnmounted(() => {
  if (videoPlayerRef.value) {
    videoPlayerRef.value.pause()
  }
  window.removeEventListener('keydown', onKeyDown)
})
</script>

<style scoped>
.video-list-view {
  /* 使用父级 main-content 的背景，不再叠加全屏渐变，避免视觉割裂 */
}

/* 两栏布局 */
.page-layout {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 20px;
  align-items: start;
}

.page-main {
  min-width: 0;
}

.page-aside {
  position: sticky;
  top: 88px;
  align-self: start;
  background: white;
  border: 1px solid rgba(2, 6, 23, 0.06);
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
  padding: 16px;
  height: fit-content;
}

.aside-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.aside-header h4 {
  margin: 0;
  font-size: 16px;
  color: #0f172a;
}
.aside-close {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 16px;
}

.aside-section {
  margin-top: 12px;
}
.aside-status {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.aside-status-item {
  display: grid;
  grid-template-columns: 14px 1fr auto;
  align-items: center;
  gap: 10px;
  padding: 10px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
}
.aside-status-item .dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #f59e0b;
}
.aside-status-item.ok {
  background: #f0fdf4;
  border-color: #bbf7d0;
}
.aside-status-item.ok .dot {
  background: #22c55e;
}
.aside-status-item .title {
  font-size: 14px;
  font-weight: 600;
  color: #0f172a;
}
.aside-status-item .desc {
  font-size: 12px;
  color: #64748b;
}
.mini-btn {
  padding: 6px 10px;
  border-radius: 8px;
  border: none;
  background: #e2e8f0;
  color: #0f172a;
  cursor: pointer;
}
.mini-btn:hover {
  background: #cbd5e1;
}

.primary-action {
  width: 100%;
  padding: 12px 14px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%);
  color: white;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 12px 24px rgba(99, 102, 241, 0.28);
}
.primary-action:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
.aside-tip {
  margin: 8px 0 0;
  font-size: 12px;
  color: #64748b;
  text-align: center;
}

/* 移动端：侧栏折叠为抽屉 */
@media (max-width: 1023px) {
  .page-layout {
    grid-template-columns: 1fr;
  }
  .page-aside {
    position: fixed;
    right: 16px;
    bottom: 16px;
    top: auto;
    left: 16px;
    transform: translateY(120%);
    transition: transform 0.25s ease;
    z-index: 1100;
  }
  .page-aside.open {
    transform: translateY(0);
  }
  .toggle-aside-btn {
    position: fixed;
    right: 16px;
    bottom: 16px;
    z-index: 1200;
    border: none;
    background: #0ea5e9;
    color: white;
    border-radius: 999px;
    padding: 10px 14px;
    box-shadow: 0 10px 20px rgba(14, 165, 233, 0.3);
  }
}

.main-content {
  padding-top: 0;
}

.video-container {
  /* 交给外层 content-area 控制整体留白 */
  padding: 0;
}

.status-section {
  background: white;
  border-radius: 16px;
  padding: 28px;
  margin-bottom: 28px;
  border: 1px solid rgba(2, 6, 23, 0.06);
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
}

.status-section h2 {
  margin: 0 0 20px 0;
  color: #333;
  text-align: center;
}

.generation-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 20px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px dashed rgba(2, 6, 23, 0.06);
}

.generation-status.configured {
  background: #f0fff4;
  border-color: rgba(34, 197, 94, 0.35);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 22px;
}

.status-text {
  font-size: 15px;
  font-weight: 600;
  color: #0f172a;
}

.status-description {
  margin: 0 auto 0 0;
  color: #475569;
  font-size: 14px;
}

.generate-btn {
  background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.2px;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease,
    opacity 0.2s ease;
  box-shadow: 0 10px 20px rgba(22, 163, 74, 0.25);
}

.generate-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 22px rgba(22, 163, 74, 0.35);
}

.videos-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(2, 6, 23, 0.06);
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 25px;
}

.section-header h3 {
  margin: 0;
  color: #0f172a;
  font-size: clamp(18px, 2.2vw, 22px);
  letter-spacing: 0.2px;
}

.action-buttons {
  display: flex;
  gap: 10px;
}

.refresh-btn,
.manager-btn {
  background: #4f46e5;
  color: white;
  border: none;
  padding: 8px 14px;
  border-radius: 10px;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease,
    background 0.2s ease;
  box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
}

.refresh-btn:hover,
.manager-btn:hover {
  background: #4338ca;
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(67, 56, 202, 0.3);
}

.video-grid,
.skeleton-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
}

.video-item {
  background: #ffffff;
  border-radius: 14px;
  border: 1px solid rgba(2, 6, 23, 0.06);
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.06);
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease;
  overflow: hidden;
}

.video-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 30px rgba(2, 6, 23, 0.08);
}

.video-thumbnail {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
  border-radius: 10px;
  overflow: hidden;
}

.thumbnail-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 42px;
  color: #94a3b8;
}

.thumbnail-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scale(1.01);
}

.video-duration {
  position: absolute;
  bottom: 8px;
  right: 8px;
  background: rgba(15, 23, 42, 0.82);
  color: white;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 700;
  backdrop-filter: blur(4px);
}

.video-info {
  padding: 14px 16px 16px;
}

.video-name {
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 8px;
  line-height: 1.35;
  max-height: 2.7em;
  overflow: hidden;
}

.video-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #475569;
}

.create-time {
  white-space: nowrap;
}

.video-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.icon-btn,
.play-btn,
.download-btn,
.share-btn,
.delete-btn {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  font-size: 16px;
  transition:
    transform 0.15s ease,
    box-shadow 0.15s ease,
    opacity 0.15s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.12);
}

.play-btn {
  background: #22c55e;
  color: white;
}
.download-btn {
  background: #0ea5e9;
  color: white;
}
.share-btn {
  background: #64748b;
  color: white;
}
.delete-btn {
  background: #ef4444;
  color: white;
}

.play-btn:hover,
.download-btn:hover,
.share-btn:hover,
.delete-btn:hover {
  transform: translateY(-1px) scale(1.02);
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.18);
}

.delete-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 60px 20px;
  color: #475569;
}

.skeleton-card {
  background: #fff;
  border: 1px solid rgba(2, 6, 23, 0.06);
  border-radius: 14px;
  padding: 12px;
}

.skeleton-thumb {
  height: 160px;
  border-radius: 10px;
  background: #e5e7eb;
  margin-bottom: 12px;
}

.skeleton-info {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.skeleton-meta {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
.skeleton-line {
  height: 12px;
  background: #e5e7eb;
  border-radius: 6px;
}

.shimmer {
  position: relative;
  overflow: hidden;
}
.shimmer::after {
  content: '';
  position: absolute;
  top: 0;
  left: -150px;
  height: 100%;
  width: 150px;
  background: linear-gradient(
    90deg,
    rgba(255, 255, 255, 0),
    rgba(255, 255, 255, 0.6),
    rgba(255, 255, 255, 0)
  );
  animation: shimmer 1.2s infinite;
}
@keyframes shimmer {
  0% {
    left: -150px;
  }
  100% {
    left: 100%;
  }
}

.empty-illustration {
  font-size: 56px;
  margin-bottom: 14px;
  color: #94a3b8;
}

.start-generate-btn {
  background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%);
  color: white;
  border: none;
  padding: 12px 20px;
  border-radius: 12px;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
  transition:
    transform 0.2s ease,
    box-shadow 0.2s ease,
    opacity 0.2s ease;
  box-shadow: 0 12px 24px rgba(99, 102, 241, 0.28);
}

.start-generate-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 28px rgba(99, 102, 241, 0.32);
}

.video-player-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(2, 6, 23, 0.92);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(6px);
}

.video-player-container {
  position: relative;
  background: #0b1220;
  border-radius: 12px;
  max-width: min(1080px, 92%);
  max-height: 88vh;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
  overflow: hidden;
}

video {
  width: 100%;
  height: 100%;
}

.player-controls {
  position: absolute;
  bottom: 10px;
  left: 10px;
  right: 10px;
  background: rgba(15, 23, 42, 0.8);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 14px;
  border-radius: 14px;
}

.close-player-btn {
  background: rgba(255, 255, 255, 0.95);
  color: #0f172a;
  border: none;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  font-size: 16px;
  cursor: pointer;
}

.video-info-player {
  color: white;
  font-size: 13px;
  font-weight: 600;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}
</style>
