<template>
  <header class="app-header">
    <div class="header-content">
      <div class="logo">
        <div class="logo-icon">🎓</div>
        <div class="logo-text">
          <h1>传媒体数字人</h1>
          <span class="subtitle">智能授课小助手</span>
        </div>
      </div>

      <div class="user-info">
        <div class="welcome-message">
          <span class="greeting">你好！</span>
          <span class="username">{{ authStore.currentUser }}</span>
        </div>
        <button @click="handleLogout" class="logout-btn">
          👋 退出
        </button>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { useRouter } from 'vue-router'

const authStore = useAuthStore()
const router = useRouter()

const handleLogout = () => {
  authStore.logout()
  router.push('/login')
}
</script>

<style scoped>
.app-header {
  background: linear-gradient(135deg, #000000 0%, #111111 100%);
  border-bottom: 1px solid #222222;
  padding: 0 clamp(20px, 4vw, 60px);
  height: clamp(60px, 8vw, 80px);
  display: flex;
  align-items: center;
  box-shadow: 0 2px 20px rgba(0, 0, 0, 0.5);
}

@media (max-width: 768px) {
  .app-header {
    padding: 0 20px;
    height: 60px;
  }
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.logo {
  display: flex;
  align-items: center;
  gap: 15px;
}

.logo-icon {
  font-size: 40px;
  color: #ffffff;
  filter: drop-shadow(0 2px 4px rgba(255, 255, 255, 0.1));
}

.logo-text h1 {
  font-size: clamp(20px, 4vw, 28px);
  color: #ffffff;
  margin: 0;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.subtitle {
  font-size: clamp(12px, 2.5vw, 14px);
  color: #888888;
  font-weight: 400;
  display: block;
  margin-top: 2px;
  letter-spacing: 0.05em;
}

.user-info {
  display: flex;
  align-items: center;
  gap: clamp(15px, 3vw, 30px);
}

@media (max-width: 768px) {
  .user-info {
    gap: 15px;
  }
}

.welcome-message {
  display: flex;
  align-items: center;
  gap: 10px;
}

.greeting {
  font-size: clamp(14px, 3vw, 16px);
  color: #cccccc;
  font-weight: 500;
}

.username {
  font-size: clamp(14px, 3vw, 16px);
  color: #ffffff;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.1);
  padding: clamp(6px, 2vw, 8px) clamp(10px, 3vw, 16px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

@media (max-width: 480px) {
  .greeting {
    display: none;
  }
}

.logout-btn {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: clamp(8px, 2vw, 10px) clamp(12px, 3vw, 20px);
  border-radius: 25px;
  cursor: pointer;
  font-size: clamp(12px, 2.5vw, 14px);
  font-weight: 500;
  color: #cccccc;
  transition: all 0.3s ease;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.logout-btn:hover {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.5);
  color: #ffffff;
  transform: translateY(-1px);
}
</style>