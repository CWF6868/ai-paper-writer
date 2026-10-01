// 前端与后端通信的入口：axios 实例 + 拦截器 + 所有接口
import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// ---- 请求拦截器：自动带 token，并打印日志方便排查 ----
api.interceptors.request.use((config) => {
  const token = getToken()
  console.log('[拦截器] 请求地址 =', config.url, '| token =', token ? token.slice(0, 25) + '...' : '(空)')
  if (token) {
    config.headers = config.headers || {}
    config.headers['Authorization'] = 'Bearer ' + token
  }
  return config
})

// ---- 响应拦截器：受保护接口 401 时清 token 跳登录（排除登录/注册）----
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const status = err.response && err.response.status
    const url = (err.config && err.config.url) || ''
    console.log('[拦截器] 响应出错 status =', status, '| 地址 =', url)
    const isAuthUrl = url.includes('/auth/login') || url.includes('/auth/register')
    if (status === 401 && !isAuthUrl) {
      clearAuth()
      if (window.location.hash !== '#/login') window.location.hash = '#/login'
    }
    return Promise.reject(err)
  }
)

// ---- token 与用户名存取 ----
export function getToken() { return localStorage.getItem('token') }
export function setToken(t) { localStorage.setItem('token', t) }
export function setUserName(name) { localStorage.setItem('uname', name) }
export function getUserName() { return localStorage.getItem('uname') }
export function clearAuth() {
  localStorage.removeItem('token')
  localStorage.removeItem('uname')
}

// ---- 认证接口 ----
export function login(username, password) {
  const form = new URLSearchParams()
  form.append('username', username)
  form.append('password', password)
  return api.post('/auth/login', form, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  })
}
export function register(data) {
  return api.post('/auth/register', data)
}

// ---- 论文接口 ----
// 注意：后端列表/创建路由是 /api/papers/（带尾斜杠），必须保持一致，
// 否则会触发 307 跨源重定向把 Authorization 头剥掉，导致 401。
export function listPapers(params = {}) { return api.get('/papers/', { params }) }
export function createPaper(data) { return api.post('/papers/', data) }
export function getPaper(id) { return api.get('/papers/' + id) }
export function updatePaper(id, data) { return api.put('/papers/' + id, data) }
export function deletePaper(id) { return api.delete('/papers/' + id) }
export function exportPaper(id, format) { return api.get('/papers/' + id + '/export', { params: { format }, responseType: 'blob' }) }

// ---- 选题推荐：LLM 生成较慢，单独放大超时 ----
export function recommendTopic(data) { return api.post('/topics/recommend', data, { timeout: 120000 }) }

// ---- 流式生成（SSE）：用 fetch 而非 axios，因为要逐条读进度事件 ----
export async function streamGeneratePaper(id, onEvent) {
  const token = getToken() || ''
  const res = await fetch(`/api/papers/${id}/stream-generate`, {
    method: 'POST',
    headers: { Authorization: 'Bearer ' + token },
  })

  if (res.status === 401) {
    clearAuth()
    if (window.location.hash !== '#/login') window.location.hash = '#/login'
    throw new Error('登录已失效，请重新登录')
  }
  if (!res.ok || !res.body) {
    let detail = ''
    try { detail = (await res.json()).detail || '' } catch (e) { /* 忽略 */ }
    throw new Error(detail || '生成请求失败（' + res.status + '）')
  }

  // 逐块读取响应流，按 SSE 规范拆包：事件块以空行分隔，行以 data: 开头
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const rawEvent = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const line = rawEvent.trim()
      if (!line.startsWith('data:')) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      try { onEvent(JSON.parse(payload)) } catch (e) { /* 忽略无法解析的行 */ }
    }
  }
}