<template>
  <!-- 嵌入到 Dashboard 的右侧内容区，去掉自身顶栏和外层背景，保持与其它模块对齐 -->
  <div class="voice-trainer-content">
    <VoiceTrainer />
  </div>
</template>

<script setup lang="ts">
import VoiceTrainer from '@/components/VoiceTrainer.vue'
</script>

<style scoped>
.voice-trainer-content {
  width: 100%;
}
</style>
