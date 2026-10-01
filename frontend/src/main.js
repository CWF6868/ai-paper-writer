// 应用入口：创建 Vue 应用，挂载路由，装入全局样式
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './style.css'

createApp(App).use(router).mount('#app')