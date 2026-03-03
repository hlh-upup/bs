<template>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-brand">
          <div class="brand-logo">智</div>
          <div class="brand-text">
            <h1 class="brand-name">智课堂</h1>
            <p class="brand-tagline">AI数字人平台</p>
          </div>
        </div>
      </div>

      <nav class="sidebar-nav">
        <div class="nav-section">
          <h3 class="nav-section-title">核心功能</h3>
          <router-link
            to="/dashboard"
            class="nav-item"
            :class="{ active: $route.name === 'dashboard-home' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="14" y="14" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" />
              </svg>
            </div>
            <span class="nav-text">工作台</span>
          </router-link>

          <router-link
            to="/dashboard/person-manager"
            class="nav-item"
            :class="{ active: $route.name === 'person-manager' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
            <span class="nav-text">形象管理</span>
          </router-link>

          <router-link
            to="/dashboard/video-config"
            class="nav-item"
            :class="{ active: $route.name === 'video-config' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="23 7 16 12 23 17 23 7" />
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
              </svg>
            </div>
            <span class="nav-text">视频生成</span>
            <span class="nav-badge">新</span>
          </router-link>
        </div>

        <div class="nav-section">
          <h3 class="nav-section-title">高级功能</h3>
          <router-link
            to="/dashboard/voice-trainer"
            class="nav-item"
            :class="{ active: $route.name === 'voice-trainer' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1v22M5 12h14" />
              </svg>
            </div>
            <span class="nav-text">语音训练</span>
          </router-link>
          <router-link
            to="/dashboard/advanced-config"
            class="nav-item"
            :class="{ active: $route.name === 'advanced-config' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3" />
                <path
                  d="M12 1v6m0 6v6m4.22-13.22l4.24 4.24M1.54 1.54l4.24 4.24M20.46 20.46l-4.24-4.24M1.54 20.46l4.24-4.24"
                />
              </svg>
            </div>
            <span class="nav-text">高级配置</span>
          </router-link>

          <router-link
            to="/dashboard/video-list"
            class="nav-item"
            :class="{ active: $route.name === 'video-list' }"
          >
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path
                  d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"
                />
              </svg>
            </div>
            <span class="nav-text">视频列表</span>
          </router-link>
        </div>

        <div class="nav-section">
          <h3 class="nav-section-title">系统</h3>
          <a href="#" class="nav-item" @click.prevent="handleLogout">
            <div class="nav-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                <polyline points="16 17 21 12 16 7" />
                <line x1="21" y1="12" x2="9" y2="12" />
              </svg>
            </div>
            <span class="nav-text">退出登录</span>
          </a>
        </div>
      </nav>

      <div class="sidebar-footer">
        <div class="user-profile">
          <div class="user-avatar">{{ authStore.currentUser?.charAt(0).toUpperCase() || 'U' }}</div>
          <div class="user-info">
            <p class="user-name">{{ authStore.currentUser || '未登录' }}</p>
            <p class="user-role">管理员</p>
          </div>
          <div class="user-actions">
            <button class="action-btn" title="设置">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3" />
                <path
                  d="M12 1v6m0 6v6m4.22-13.22l4.24 4.24M1.54 1.54l4.24 4.24M20.46 20.46l-4.24-4.24M1.54 20.46l4.24-4.24"
                />
              </svg>
            </button>
            <button class="action-btn" title="通知">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                <path d="M13.73 21a2 2 0 0 1-3.46 0" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </aside>

    <!-- 主内容区域 -->
    <main class="main-content">
      <!-- 顶部工具栏 -->
      <header class="top-toolbar">
        <div class="toolbar-left">
          <div>
            <h1 class="page-title">{{ getPageTitle() }}</h1>
            <p class="page-subtitle">{{ getPageSubtitle() }}</p>
          </div>
        </div>
        <div class="toolbar-right">
          <div class="toolbar-actions">
            <button class="toolbar-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
              搜索
            </button>
            <button class="toolbar-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
              帮助
            </button>
          </div>
        </div>
      </header>

      <!-- 内容区域 -->
      <div class="content-area">
        <div class="content-container">
          <Transition name="page" mode="out-in">
            <router-view />
          </Transition>
        </div>
      </div>
    </main>

    <!-- 全局处理模态框 -->
    <Teleport to="body">
      <ProcessingModal
        v-if="digitalHumanStore.isProcessing"
        :task="digitalHumanStore.currentTask || ''"
        :progress="digitalHumanStore.progress"
      />
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { useDigitalHumanStore } from '@/stores/digitalHuman'
import ProcessingModal from '@/components/ProcessingModal.vue'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()
const digitalHumanStore = useDigitalHumanStore()

const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}

// 获取页面标题
const getPageTitle = () => {
  const titles = {
    'dashboard-home': '工作台',
    'person-manager': '形象管理',
    'video-config': '视频生成',
    'advanced-config': '高级配置',
    'voice-trainer': '语音训练',
    'video-list': '视频列表',
  } as Record<string, string>
  return titles[route.name as keyof typeof titles] || '智课堂'
}

// 获取页面副标题
const getPageSubtitle = () => {
  const subtitles = {
    'dashboard-home': '管理和创建您的数字人内容',
    'person-manager': '配置和管理数字人形象',
    'video-config': '上传PPT生成数字人视频',
    'advanced-config': '精细调整视频参数',
    'voice-trainer': '上传训练音频并训练VITS模型',
    'video-list': '查看和管理已生成的视频',
  } as Record<string, string>
  return subtitles[route.name as keyof typeof subtitles] || ''
}
</script>

<style scoped>
.app-layout {
  display: flex;
  min-height: 100vh;
  background: var(--gradient-background);
}

.sidebar {
  width: 260px;
  min-width: 260px;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 20;
  border-right: 1px solid var(--color-gray-200);
  background: var(--color-white);
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: var(--spacing-2xl);
  border-bottom: 1px solid var(--color-gray-200);
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.brand-logo {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-lg);
  background: var(--gradient-primary);
  color: var(--color-white);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-bold);
}

.brand-name {
  margin: 0;
  font-size: var(--text-xl);
  color: var(--color-gray-900);
}

.brand-tagline {
  margin: 2px 0 0 0;
  font-size: var(--text-xs);
  color: var(--color-secondary);
}

.sidebar-nav {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-xl) var(--spacing-md);
}

.nav-section {
  margin-bottom: var(--spacing-2xl);
}

.nav-section-title {
  margin: 0 0 var(--spacing-sm) var(--spacing-md);
  font-size: var(--text-xs);
  color: var(--color-secondary);
}

.nav-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: 10px var(--spacing-md);
  border-radius: var(--radius-lg);
  color: var(--color-gray-700);
  text-decoration: none;
  margin-bottom: var(--spacing-xs);
  transition: all var(--transition-fast);
}

.nav-item:hover {
  background: var(--color-gray-100);
  color: var(--color-primary);
}

.nav-item.active {
  background: var(--gradient-primary);
  color: var(--color-white);
  box-shadow: var(--shadow-primary);
}

.nav-icon {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nav-text {
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
}

.nav-badge {
  margin-left: auto;
  padding: 2px 6px;
  font-size: 10px;
  border-radius: var(--radius-full);
  background: rgba(99, 102, 241, 0.15);
  color: var(--color-primary-dark);
}

.nav-item.active .nav-badge {
  background: rgba(255, 255, 255, 0.2);
  color: var(--color-white);
}

.sidebar-footer {
  border-top: 1px solid var(--color-gray-200);
  padding: var(--spacing-lg);
}

.user-profile {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.user-avatar {
  width: 36px;
  height: 36px;
  border-radius: var(--radius-full);
  background: var(--gradient-accent);
  color: var(--color-white);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-semibold);
}

.user-name {
  margin: 0;
  font-size: var(--text-sm);
  color: var(--color-gray-900);
}

.user-role {
  margin: 2px 0 0 0;
  font-size: var(--text-xs);
  color: var(--color-secondary);
}

.user-actions {
  margin-left: auto;
  display: flex;
  gap: var(--spacing-xs);
}

.action-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--color-secondary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-btn:hover {
  background: var(--color-gray-100);
  color: var(--color-primary);
}

.main-content {
  margin-left: 260px;
  width: calc(100% - 260px);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.top-toolbar {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 var(--spacing-3xl);
  border-bottom: 1px solid var(--color-gray-200);
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  position: sticky;
  top: 0;
  z-index: 10;
}

.page-title {
  margin: 0;
  font-size: var(--text-2xl);
  color: var(--color-gray-900);
}

.page-subtitle {
  margin: 4px 0 0 0;
  font-size: var(--text-sm);
  color: var(--color-secondary);
}

.toolbar-actions {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.toolbar-btn {
  border: 1px solid var(--color-gray-300);
  background: var(--color-white);
  color: var(--color-gray-700);
  border-radius: var(--radius-md);
  padding: 8px 12px;
  font-size: var(--text-sm);
  cursor: pointer;
  transition: all var(--transition-fast);
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.toolbar-btn:hover {
  border-color: var(--color-primary-light);
  color: var(--color-primary);
  background: var(--color-gray-50);
}

.content-area {
  flex: 1;
  padding: var(--spacing-2xl);
  overflow-y: auto;
}

.content-container {
  width: 100%;
  max-width: 100%;
}

@media (max-width: 1024px) {
  .sidebar {
    width: 220px;
    min-width: 220px;
  }

  .main-content {
    margin-left: 220px;
    width: calc(100% - 220px);
  }
}

@media (max-width: 820px) {
  .sidebar {
    display: none;
  }

  .main-content {
    margin-left: 0;
    width: 100%;
  }

  .top-toolbar {
    padding: 0 var(--spacing-lg);
  }

  .toolbar-btn {
    padding: 6px 10px;
    font-size: var(--text-xs);
  }

  .content-area {
    padding: var(--spacing-lg);
  }

  .content-container {
    padding: var(--spacing-md);
  }
}
</style>
