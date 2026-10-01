// 路由表 + 登录守卫
import { createRouter, createWebHashHistory } from 'vue-router'
import Login from '../views/Login.vue'
import PaperList from '../views/PaperList.vue'
import PaperDetail from '../views/PaperDetail.vue'

const routes = [
  { path: '/', redirect: '/papers' },   // 访问根路径 → 直接去论文列表
  { path: '/login', component: Login },
  { path: '/papers', component: PaperList },
  { path: '/papers/:id', component: PaperDetail },   // 论文详情 + 流式生成
]

// 用 hash 模式：地址形如 http://localhost:5173/#/papers
const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

// 全局守卫：每次页面跳转前都会执行
router.beforeEach((to) => {
  const token = localStorage.getItem('token')
  // 想去 /papers 等页但没登录 → 拦回登录页
  if (to.path !== '/login' && !token) return '/login'
  // 已经登录却还去登录页 → 直接送去论文列表
  if (to.path === '/login' && token) return '/papers'
})

export default router