<template>
  <!-- 嵌入到 Dashboard 右侧内容区的纯内容页：去掉自身顶栏与外层背景，保持对齐 -->
  <div class="advanced-config-content">
    <AdvancedVideoConfig />
  </div>
</template>

<script setup lang="ts">
import AdvancedVideoConfig from '@/components/AdvancedVideoConfig.vue'
</script>

<style scoped>
.advanced-config-content {
  width: 100%;
}
</style>