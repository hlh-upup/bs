<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h3>处理中...</h3>
      </div>

      <div class="modal-body">
        <div class="loading-spinner"></div>
        <p class="task-name">{{ task || '正在处理' }}</p>
        <div class="progress-container" v-if="progress > 0">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: `${progress}%` }"></div>
          </div>
          <span class="progress-text">{{ progress }}%</span>
        </div>
      </div>

      <div class="modal-footer">
        <p class="hint">请耐心等待，处理时间较长...</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  task?: string
  progress: number
}>()

defineEmits<{
  close: []
}>()
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 12px;
  padding: 30px;
  max-width: 400px;
  width: 90%;
  text-align: center;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.modal-header h3 {
  margin: 0 0 20px 0;
  color: #333;
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.task-name {
  font-size: 18px;
  color: #333;
  margin-bottom: 20px;
}

.progress-container {
  margin: 20px 0;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
}

.progress-text {
  margin-top: 10px;
  font-weight: 600;
  color: #667eea;
}

.hint {
  color: #666;
  font-size: 14px;
  margin: 0;
}
</style>