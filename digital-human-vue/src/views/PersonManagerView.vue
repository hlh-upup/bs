<template>
  <!-- 嵌入到 Dashboard 右侧内容区的纯内容页：不再包含自己的 AppHeader/主容器 -->
  <div class="person-manager-content">
    <PersonManager />
  </div>
</template>

<script setup lang="ts">
import PersonManager from '@/components/PersonManager.vue'
</script>

<style scoped>
/* 让内容跟随父级 content-area 铺满，不额外添加背景或内边距 */
.person-manager-content {
  width: 100%;
}
</style>
