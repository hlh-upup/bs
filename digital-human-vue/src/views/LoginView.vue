<template>
  <div class="login-page">
    <div class="login-shell">
      <section class="login-hero">
        <div class="hero-badge">数字人平台</div>
        <h1>传媒体·智课堂</h1>
        <p>面向教学场景的一站式数字人生成系统</p>
        <ul class="hero-points">
          <li>视频生成与质量评测一体化</li>
          <li>语音训练与形象管理统一工作台</li>
          <li>支持后续平滑接入 BS 真模型</li>
        </ul>
      </section>

      <section class="login-panel">
        <div class="login-header">
          <div class="logo-section">
            <div class="logo-icon">智</div>
            <div class="logo-text">
              <h2>账号登录</h2>
              <p>请输入账号信息进入系统</p>
            </div>
          </div>
        </div>

        <form @submit.prevent="handleLogin" class="login-form">
          <div class="form-group">
            <label for="username">用户名</label>
            <input
              id="username"
              v-model="username"
              type="text"
              placeholder="请输入用户名"
              required
              :disabled="authStore.loading"
              class="input-field"
            />
          </div>

          <div class="form-group">
            <label for="password">密码</label>
            <input
              id="password"
              v-model="password"
              type="password"
              placeholder="请输入密码"
              required
              :disabled="authStore.loading"
              class="input-field"
            />
          </div>

          <div v-if="authStore.error" class="error-message">
            {{ authStore.error }}
          </div>

          <button type="submit" :disabled="authStore.loading" class="login-button">
            <span v-if="authStore.loading">登录中...</span>
            <span v-else>登录系统</span>
          </button>
        </form>

        <div class="login-footer">
          <button
            type="button"
            class="demo-fill-btn"
            :disabled="authStore.loading"
            @click="fillDemoAccount"
          >
            一键填充
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')

const fillDemoAccount = () => {
  username.value = 'Test'
  password.value = '123000'
}

const handleLogin = async () => {
  const success = await authStore.login(username.value, password.value)
  if (success) {
    const redirect = (route.query.redirect as string) || '/dashboard'
    router.push(redirect)
  }
}
</script>

<style scoped>

.login-page {
  min-height: 100vh;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-4xl);
  background: var(--gradient-background);
}

.login-shell {
  width: min(1060px, 100%);
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  border-radius: var(--radius-3xl);
  overflow: hidden;
  border: 1px solid var(--color-gray-200);
  background: var(--color-white);
  box-shadow: var(--shadow-xl);
}

.login-hero {
  padding: var(--spacing-6xl);
  background: var(--gradient-primary);
  color: var(--color-white);
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: var(--spacing-xl);
}

.hero-badge {
  width: fit-content;
  padding: var(--spacing-xs) var(--spacing-md);
  border-radius: var(--radius-full);
  background: rgba(255, 255, 255, 0.18);
  font-size: var(--text-xs);
}

.login-hero h1 {
  margin: 0;
  font-size: var(--text-5xl);
  line-height: var(--leading-tight);
}

.login-hero p {
  margin: 0;
  font-size: var(--text-base);
  opacity: 0.95;
}

.hero-points {
  margin: 0;
  padding-left: 18px;
  display: grid;
  gap: var(--spacing-sm);
  font-size: var(--text-sm);
}

.login-panel {
  padding: var(--spacing-5xl);
  display: flex;
  flex-direction: column;
  justify-content: center;
  background: var(--color-white);
}

.login-header {
  margin-bottom: var(--spacing-3xl);
}

.logo-section {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.logo-icon {
  width: 44px;
  height: 44px;
  border-radius: var(--radius-lg);
  background: var(--gradient-primary);
  color: var(--color-white);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: var(--font-bold);
}

.logo-text h2 {
  margin: 0;
  font-size: var(--text-2xl);
  color: var(--color-gray-900);
}

.logo-text p {
  margin: 2px 0 0 0;
  color: var(--color-secondary);
  font-size: var(--text-sm);
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xl);
}

.login-footer {
  margin-top: var(--spacing-3xl);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.form-group label {
  display: block;
  margin-bottom: var(--spacing-sm);
  font-weight: var(--font-medium);
  color: var(--color-gray-700);
  font-size: var(--text-sm);
}

.input-field {
  width: 100%;
  padding: var(--spacing-md) var(--spacing-lg);
  border: 1px solid var(--color-gray-300);
  border-radius: var(--radius-lg);
  background: var(--color-white);
  color: var(--color-gray-900);
  font-size: var(--text-base);
  transition: all var(--transition-smooth);
}

.input-field:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12);
}

.input-field:hover:not(:focus) {
  border-color: var(--color-gray-400);
}

.input-field:disabled {
  background-color: var(--color-gray-100);
  cursor: not-allowed;
}

.error-message {
  color: var(--color-error);
  font-size: var(--text-sm);
  background: var(--color-error-light);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: var(--radius-md);
  padding: var(--spacing-sm) var(--spacing-md);
}

.login-button {
  background: var(--gradient-primary);
  color: var(--color-white);
  border: none;
  width: 100%;
  padding: var(--spacing-md) var(--spacing-lg);
  border-radius: var(--radius-lg);
  font-size: var(--text-base);
  font-weight: var(--font-semibold);
  cursor: pointer;
  transition: all var(--transition-smooth);
  box-shadow: var(--shadow-primary);
}

.login-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: var(--shadow-primary-hover);
}

.login-button:active:not(:disabled) {
  transform: translateY(0);
}

.login-button:disabled {
  opacity: 0.65;
  cursor: not-allowed;
  box-shadow: none;
}

.demo-fill-btn {
  border: 1px solid var(--color-gray-300);
  background: var(--color-white);
  color: var(--color-gray-700);
  padding: var(--spacing-sm) var(--spacing-md);
  border-radius: var(--radius-full);
  font-size: var(--text-xs);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.demo-fill-btn:hover:not(:disabled) {
  border-color: var(--color-primary-light);
  color: var(--color-primary);
}

.demo-fill-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 980px) {
  .login-page {
    padding: var(--spacing-2xl);
  }

  .login-shell {
    grid-template-columns: 1fr;
  }

  .login-hero {
    padding: var(--spacing-4xl);
  }

  .login-hero h1 {
    font-size: var(--text-4xl);
  }

  .login-panel {
    padding: var(--spacing-4xl);
  }
}

@media (max-width: 640px) {
  .login-hero {
    padding: var(--spacing-3xl);
  }

  .login-panel {
    padding: var(--spacing-3xl);
  }

  .login-footer {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>