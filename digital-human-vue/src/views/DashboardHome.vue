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
/* 复用 DashboardView 中的欢迎样式类名，避免重复样式 */
</style>
