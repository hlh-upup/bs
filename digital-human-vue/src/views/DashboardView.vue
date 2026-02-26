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
/* 欢迎页面样式 */
.welcome-dashboard {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 72px);
  padding: var(--spacing-4xl);
}

.welcome-card {
  background: rgba(255, 255, 255, 0.95);
  border-radius: var(--radius-3xl);
  box-shadow: var(--shadow-xl);
  backdrop-filter: blur(16px);
  padding: var(--spacing-6xl);
  max-width: 900px;
  width: 100%;
  border: 1px solid rgba(0, 0, 0, 0.06);
}

.welcome-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-3xl);
  margin-bottom: var(--spacing-6xl);
  padding-bottom: var(--spacing-4xl);
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.welcome-icon {
  width: 80px;
  height: 80px;
  background: var(--gradient-primary);
  border-radius: var(--radius-2xl);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: var(--shadow-primary);
  flex-shrink: 0;
}

.welcome-icon svg {
  width: 40px;
  height: 40px;
}

.welcome-header h2 {
  font-size: var(--text-4xl);
  font-weight: var(--font-bold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-sm) 0;
  line-height: var(--leading-tight);
}

.welcome-header p {
  font-size: var(--text-lg);
  color: var(--color-secondary);
  margin: 0;
  font-weight: var(--font-medium);
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: var(--spacing-3xl);
}

.action-card {
  background: rgba(255, 255, 255, 0.8);
  border: 1px solid rgba(0, 0, 0, 0.04);
  border-radius: var(--radius-xl);
  padding: var(--spacing-3xl);
  text-align: center;
  transition: all var(--transition-smooth);
  position: relative;
  overflow: hidden;
}

.action-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--gradient-primary);
  opacity: 0;
  transition: opacity var(--transition-normal);
}

.action-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-xl);
  border-color: var(--color-primary-lightest);
}

.action-card:hover::before {
  opacity: 1;
}

.action-icon {
  width: 60px;
  height: 60px;
  border-radius: var(--radius-xl);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto var(--spacing-xl);
  color: white;
  box-shadow: var(--shadow-md);
  transition: all var(--transition-smooth);
}

.action-icon.primary {
  background: var(--gradient-primary);
}

.action-icon.accent {
  background: var(--gradient-accent);
}

.action-icon.success {
  background: var(--gradient-success);
}

.action-card:hover .action-icon {
  transform: scale(1.1);
  box-shadow: var(--shadow-lg);
}

.action-icon svg {
  width: 28px;
  height: 28px;
}

.action-card h3 {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  color: var(--color-gray-900);
  margin: 0 0 var(--spacing-md) 0;
}

.action-card p {
  font-size: var(--text-base);
  color: var(--color-secondary);
  margin: 0 0 var(--spacing-2xl) 0;
  line-height: var(--leading-relaxed);
}

.action-card .action-btn {
  display: inline-flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-md) var(--spacing-xl);
  border-radius: var(--radius-lg);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
  text-decoration: none;
  transition: all var(--transition-smooth);
  border: 1px solid transparent;
}

.action-card .action-btn:not(.primary):not(.accent) {
  background: rgba(255, 255, 255, 0.8);
  color: var(--color-gray-700);
  border-color: rgba(0, 0, 0, 0.08);
}

.action-card .action-btn.primary {
  background: var(--gradient-primary);
  color: white;
  box-shadow: var(--shadow-primary);
}

.action-card .action-btn.accent {
  background: var(--gradient-accent);
  color: white;
  box-shadow: var(--shadow-accent);
}

.action-card .action-btn:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-lg);
}

.action-card .action-btn.primary:hover {
  box-shadow: var(--shadow-primary-hover);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .welcome-dashboard {
    padding: var(--spacing-2xl);
  }

  .welcome-card {
    padding: var(--spacing-4xl);
  }

  .welcome-header {
    flex-direction: column;
    text-align: center;
    gap: var(--spacing-xl);
    margin-bottom: var(--spacing-4xl);
  }

  .welcome-icon {
    width: 60px;
    height: 60px;
  }

  .welcome-icon svg {
    width: 30px;
    height: 30px;
  }

  .welcome-header h2 {
    font-size: var(--text-2xl);
  }

  .welcome-header p {
    font-size: var(--text-base);
  }

  .quick-actions {
    grid-template-columns: 1fr;
    gap: var(--spacing-2xl);
  }

  .action-card {
    padding: var(--spacing-2xl);
  }

  .action-icon {
    width: 50px;
    height: 50px;
  }

  .action-icon svg {
    width: 24px;
    height: 24px;
  }

  .action-card h3 {
    font-size: var(--text-lg);
  }

  .action-card p {
    font-size: var(--text-sm);
  }
}
</style>
