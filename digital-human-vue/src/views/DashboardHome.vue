<template>
  <div class="welcome-dashboard">
    <div class="welcome-card">
      <div class="welcome-header">
        <div class="welcome-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" />
          </svg>
        </div>
        <div>
          <h2>欢迎回来！</h2>
          <p>开始创建您的数字人视频内容</p>
        </div>
      </div>

      <div class="quick-actions">
        <div class="action-card">
          <div class="action-icon primary">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="23 7 16 12 23 17 23 7" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
          </div>
          <h3>创建视频</h3>
          <p>上传PPT快速生成数字人视频</p>
          <router-link to="/dashboard/video-config" class="action-btn primary"
            >开始创建</router-link
          >
        </div>

        <div class="action-card">
          <div class="action-icon accent">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <h3>管理形象</h3>
          <p>配置和管理数字人形象</p>
          <router-link to="/dashboard/person-manager" class="action-btn accent"
            >管理形象</router-link
          >
        </div>

        <div class="action-card">
          <div class="action-icon" style="background: var(--gradient-accent)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 1v22M5 12h14" />
            </svg>
          </div>
          <h3>语音训练</h3>
          <p>上传训练音频并训练VITS模型</p>
          <router-link to="/dashboard/voice-trainer" class="action-btn accent"
            >进入训练</router-link
          >
        </div>

        <div class="action-card">
          <div class="action-icon success">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path
                d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"
              />
            </svg>
          </div>
          <h3>视频列表</h3>
          <p>查看和管理已生成的视频</p>
          <router-link to="/dashboard/video-list" class="action-btn">查看列表</router-link>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 纯展示页，无额外逻辑
</script>

<style scoped>
.welcome-dashboard {
  padding: var(--spacing-xl);
}

.welcome-card {
  background: var(--color-white);
  border: 1px solid var(--color-gray-200);
  border-radius: var(--radius-xl);
  padding: var(--spacing-3xl);
}

.welcome-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-xl);
  margin-bottom: var(--spacing-3xl);
}

.welcome-icon {
  width: 56px;
  height: 56px;
  border-radius: var(--radius-lg);
  background: var(--gradient-primary);
  color: var(--color-white);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.welcome-icon svg {
  width: 26px;
  height: 26px;
}

.welcome-header h2 {
  margin: 0;
  color: var(--color-gray-900);
  font-size: var(--text-3xl);
}

.welcome-header p {
  margin: 4px 0 0 0;
  color: var(--color-secondary);
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--spacing-xl);
}

.action-card {
  border: 1px solid var(--color-gray-200);
  border-radius: var(--radius-lg);
  padding: var(--spacing-xl);
  background: var(--color-white);
  transition: all var(--transition-fast);
}

.action-card:hover {
  border-color: var(--color-primary-light);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.action-icon {
  width: 44px;
  height: 44px;
  border-radius: var(--radius-md);
  color: var(--color-white);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: var(--spacing-md);
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

.action-icon svg {
  width: 22px;
  height: 22px;
}

.action-card h3 {
  margin: 0;
  font-size: var(--text-xl);
  color: var(--color-gray-900);
}

.action-card p {
  margin: 8px 0 var(--spacing-lg) 0;
  color: var(--color-secondary);
  font-size: var(--text-sm);
}

.action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-gray-300);
  text-decoration: none;
  color: var(--color-gray-700);
  font-size: var(--text-sm);
  font-weight: var(--font-medium);
}

.action-btn.primary,
.action-btn.accent {
  border-color: transparent;
  color: var(--color-white);
}

.action-btn.primary {
  background: var(--gradient-primary);
}

.action-btn.accent {
  background: var(--gradient-accent);
}

@media (max-width: 960px) {
  .quick-actions {
    grid-template-columns: 1fr;
  }

  .welcome-header h2 {
    font-size: var(--text-2xl);
  }
}
</style>
