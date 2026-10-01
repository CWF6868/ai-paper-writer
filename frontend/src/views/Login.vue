<!-- 登录/注册页：登录成功后自动跳到论文列表 -->
<template>
  <div class="center">
    <h1>{{ isLogin ? '登录' : '注册' }}</h1>

    <template v-if="isLogin">
      <input v-model="username" placeholder="用户名" />
      <input v-model="password" type="password" placeholder="密码" />
      <button @click="handleLogin" :disabled="loading">
        {{ loading ? '登录中…' : '登录' }}
      </button>
    </template>

    <template v-else>
      <input v-model="regUsername" placeholder="用户名（至少 3 个字符）" />
      <input v-model="regEmail" type="email" placeholder="邮箱" />
      <input v-model="regPassword" type="password" placeholder="密码（至少 6 位）" />
      <input v-model="regConfirm" type="password" placeholder="确认密码" />
      <button @click="handleRegister" :disabled="loading">
        {{ loading ? '注册中…' : '注册' }}
      </button>
    </template>

    <p v-if="error" class="tip" style="color:#dc2626">{{ error }}</p>
    <p v-if="success" class="tip" style="color:#16a34a">{{ success }}</p>

    <p class="tip link" @click="toggleMode">
      {{ isLogin ? '没有账号？去注册' : '已有账号？去登录' }}
    </p>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { login, register, setToken, setUserName } from '../api'

const router = useRouter()
const isLogin = ref(true)
const username = ref('')
const password = ref('')
const regUsername = ref('')
const regEmail = ref('')
const regPassword = ref('')
const regConfirm = ref('')
const loading = ref(false)
const error = ref('')
const success = ref('')

function toggleMode() {
  isLogin.value = !isLogin.value
  error.value = ''
  success.value = ''
}

async function handleLogin() {
  error.value = ''
  success.value = ''
  loading.value = true
  try {
    const res = await login(username.value, password.value)
    setToken(res.data.access_token)
    setUserName(username.value)  // 先用输入框的值做显示名，后续可改为调 /me
    router.push('/papers')
  } catch (e) {
    error.value = e.response?.data?.detail || '登录失败，请检查用户名密码'
  } finally {
    loading.value = false
  }
}

async function handleRegister() {
  error.value = ''
  success.value = ''
  if (regPassword.value !== regConfirm.value) {
    error.value = '两次输入的密码不一致'
    return
  }
  loading.value = true
  try {
    await register({
      username: regUsername.value,
      email: regEmail.value,
      password: regPassword.value,
    })
    success.value = '注册成功，请登录'
    username.value = regUsername.value
    isLogin.value = true
    regPassword.value = ''
    regConfirm.value = ''
  } catch (e) {
    error.value = e.response?.data?.detail || '注册失败'
  } finally {
    loading.value = false
  }
}
</script>