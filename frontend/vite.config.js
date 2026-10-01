// Vite 配置：Vue 插件 + 把 /api 请求代理到后端
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      // 前端发出的 /api/xxx 请求，会被 Vite 转发到后端 8000
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})