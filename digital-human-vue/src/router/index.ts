import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/dashboard' },
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/LoginView.vue'),
      meta: { requiresGuest: true },
    },
    {
      path: '/dashboard',
      component: () => import('../views/DashboardView.vue'),
      meta: { requiresAuth: true },
      children: [
        { path: '', name: 'dashboard-home', component: () => import('../views/DashboardHome.vue') },
        {
          path: 'person-manager',
          name: 'person-manager',
          component: () => import('../views/PersonManagerView.vue'),
        },
        {
          path: 'video-config',
          name: 'video-config',
          component: () => import('../views/VideoConfigView.vue'),
        },
        {
          path: 'advanced-config',
          name: 'advanced-config',
          component: () => import('../views/AdvancedConfigView.vue'),
        },
        {
          path: 'voice-trainer',
          name: 'voice-trainer',
          component: () => import('../views/VoiceTrainerView.vue'),
        },
        {
          path: 'video-list',
          name: 'video-list',
          component: () => import('../views/VideoListView.vue'),
        },
      ],
    },
  ],
})

// Navigation guards（更稳健：支持嵌套路由、并在首次进入时恢复本地登录状态）
router.beforeEach((to, from, next) => {
  const authStore = useAuthStore()
  if (!authStore.user) authStore.checkAuth()

  const requiresAuth = to.matched.some((record) => record.meta?.requiresAuth)
  const requiresGuest = to.matched.some((record) => record.meta?.requiresGuest)

  if (requiresAuth && !authStore.isLoggedIn) {
    next({ name: 'login' })
  } else if (requiresGuest && authStore.isLoggedIn) {
    next({ name: 'dashboard' })
  } else {
    next()
  }
})

export default router
