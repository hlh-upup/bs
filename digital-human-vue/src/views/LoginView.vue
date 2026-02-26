<template>
  <div class="login-container">
    <div class="login-card">
      <div class="login-header">
        <div class="logo-section">
          <div class="logo-icon">🎬</div>
          <div class="logo-text">
            <h1>传媒体·智课堂</h1>
            <p>数字人智能授课系统</p>
          </div>
        </div>
        <div class="welcome-decoration">
          <span class="star">✨</span>
          <span class="heart">💖</span>
          <span class="star">⭐</span>
        </div>
      </div>

      <form @submit.prevent="handleLogin" class="login-form">
        <div class="form-group">
          <label for="username">👤 用户名</label>
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
          <label for="password">🔒 密码</label>
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
          ⚠️ {{ authStore.error }}
        </div>

        <button type="submit" :disabled="authStore.loading" class="login-button">
          <span v-if="authStore.loading">🚀 登录中...</span>
          <span v-else>🎯 开始学习之旅</span>
        </button>
      </form>

      <div class="login-footer">
        <div class="test-account">
          <span class="label">🎓 测试账号:</span>
          <span class="account">Test / 123000</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')

const handleLogin = async () => {
  const success = await authStore.login(username.value, password.value)
  if (success) {
    router.push('/dashboard')
  }
}
</script>

<style scoped>
.login-container {
  width: 100vw;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background:
    radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(14, 165, 233, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 40% 60%, rgba(99, 102, 241, 0.06) 0%, transparent 50%),
    linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #334155 100%);
  margin: 0;
  padding: 0;
  position: fixed;
  top: 0;
  left: 0;
  overflow: hidden;
}

.login-container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    radial-gradient(circle at 25% 25%, rgba(255, 255, 255, 0.02) 0%, transparent 40%),
    radial-gradient(circle at 75% 75%, rgba(255, 255, 255, 0.01) 0%, transparent 40%);
  animation: floatGradient 20s ease-in-out infinite alternate;
  pointer-events: none;
}

@keyframes floatGradient {
  0% { transform: translate(0, 0) rotate(0deg); }
  50% { transform: translate(-20px, -20px) rotate(1deg); }
  100% { transform: translate(20px, -10px) rotate(-1deg); }
}

.login-card {
  background: rgba(15, 15, 15, 0.85);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: var(--radius-3xl);
  padding: 0;
  width: min(90vw, 500px);
  min-height: 600px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  backdrop-filter: blur(20px);
  box-shadow:
    var(--shadow-2xl),
    0 0 0 1px rgba(99, 102, 241, 0.1) inset,
    0 20px 80px rgba(99, 102, 241, 0.15);
  position: relative;
  overflow: hidden;
  animation: cardEntrance 0.6s ease-out;
}

.login-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent,
    rgba(99, 102, 241, 0.6),
    rgba(14, 165, 233, 0.6),
    transparent
  );
  animation: shimmer 3s ease-in-out infinite;
}

@keyframes cardEntrance {
  from {
    opacity: 0;
    transform: translateY(30px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  50% { transform: translateX(100%); }
  100% { transform: translateX(100%); }
}

.login-header {
  text-align: center;
  margin-bottom: clamp(30px, 5vh, 60px);
  max-width: min(480px, 90vw);
  padding: 0 clamp(20px, 3vw, 40px);
}

.logo-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-bottom: 20px;
}

.logo-icon {
  font-size: clamp(40px, 8vw, 50px);
  filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
}

.logo-text h1 {
  color: #ffffff;
  font-size: clamp(22px, 5vw, 26px);
  margin: 0;
  font-weight: 700;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.logo-text p {
  color: #cccccc;
  font-size: clamp(14px, 3vw, 16px);
  margin: 5px 0 0 0;
  font-weight: 500;
}

.welcome-decoration {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 15px;
}

.welcome-decoration span {
  font-size: clamp(16px, 4vw, 20px);
}

@media (max-width: 480px) {
  .welcome-decoration {
    display: none;
  }
}

.login-form {
  display: flex;
  flex-direction: column;
  gap: clamp(20px, 3vh, 30px);
  width: 100%;
  max-width: min(400px, 85vw);
  padding: 0 clamp(15px, 2vw, 30px);
}

.form-group label {
  font-weight: 600;
  color: #cccccc;
  margin-bottom: 8px;
  font-size: clamp(14px, 3vw, 16px);
}

.input-field {
  padding: clamp(12px, 3vw, 15px) clamp(15px, 4vw, 20px);
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 25px;
  font-size: clamp(14px, 3vw, 16px);
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.05);
  color: #ffffff;
  backdrop-filter: blur(5px);
}

.input-field:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.08);
}

.input-field:hover:not(:focus) {
  border-color: var(--color-primary-lighter);
  background: rgba(255, 255, 255, 0.06);
}

.input-field:disabled {
  background-color: rgba(245, 245, 245, 0.5);
  cursor: not-allowed;
}

.error-message {
  color: #e74c3c;
  text-align: center;
  font-size: 14px;
  font-weight: 600;
  background: rgba(231, 76, 60, 0.1);
  padding: 10px;
  border-radius: 15px;
  border: 1px solid rgba(231, 76, 60, 0.2);
}

.login-button {
  background: var(--gradient-primary);
  color: white;
  border: none;
  padding: clamp(14px, 3vw, 16px) clamp(25px, 5vw, 30px);
  border-radius: 30px;
  font-size: clamp(14px, 3vw, 16px);
  font-weight: 700;
  cursor: pointer;
  transition: all var(--transition-smooth);
  box-shadow: var(--shadow-primary);
  text-transform: uppercase;
  letter-spacing: 1px;
  min-height: 48px;
}

.login-button:hover:not(:disabled) {
  transform: translateY(-3px);
  box-shadow: var(--shadow-primary-hover);
  background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
}

.login-button:active:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4);
}

.login-button:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: scale(0.98);
  background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
}

.test-account {
  text-align: center;
  margin-top: 30px;
  padding: 15px;
  background: rgba(255, 107, 157, 0.1);
  border-radius: 20px;
  border: 1px solid rgba(255, 107, 157, 0.2);
}

.label {
  color: #2c3e50;
  font-weight: 600;
  margin-right: 8px;
}

.account {
  color: #ff6b9d;
  font-weight: 700;
  font-family: monospace;
  background: rgba(255, 255, 255, 0.5);
  padding: 4px 8px;
  border-radius: 8px;
}
</style>