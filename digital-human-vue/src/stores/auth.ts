import { defineStore } from 'pinia'
import { digitalHumanApi } from '@/services/api'

export interface User {
  username: string
  isLoggedIn: boolean
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null as User | null,
    loading: false,
    error: null as string | null,
  }),

  getters: {
    isLoggedIn: (state) => state.user?.isLoggedIn || false,
    currentUser: (state) => state.user?.username || '',
  },

  actions: {
    async login(username: string, password: string) {
      this.loading = true
      this.error = null

      try {
        const success = await digitalHumanApi.login({ User: username, Password: password })
        if (success) {
          this.user = {
            username,
            isLoggedIn: true,
          }
          localStorage.setItem('user', JSON.stringify(this.user))
          return true
        } else {
          this.error = '登录失败，请检查用户名和密码'
          return false
        }
      } catch (error) {
        this.error = '服务器连接失败'
        return false
      } finally {
        this.loading = false
      }
    },

    logout() {
      this.user = null
      localStorage.removeItem('user')
    },

    checkAuth() {
      const savedUser = localStorage.getItem('user')
      if (savedUser) {
        this.user = JSON.parse(savedUser)
      }
    },
  },
})