<!-- 论文列表页：展示、新建、删除、退出、选题推荐 -->
<template>
  <div class="container">
    <header class="topbar">
      <h1 class="title">我的论文</h1>
      <div class="actions">
        <span class="uname">{{ uname }}</span>
        <button class="btn btn-sm" @click="logout">退出</button>
      </div>
    </header>

    <div class="toolbar list-toolbar">
      <button class="btn btn-primary" @click="openCreate">新建论文</button>
      <div class="filters">
        <input v-model="searchQuery" class="search-input" placeholder="搜索标题 / 方向 / 关键词" @keyup.enter="applySearch" />
        <button class="btn btn-sm" @click="applySearch">搜索</button>
        <select v-model="statusFilter" class="filter-select" @change="applyFilter">
          <option value="">全部状态</option>
          <option value="draft">草稿</option>
          <option value="outline">大纲</option>
          <option value="writing">撰写中</option>
          <option value="review">审核中</option>
          <option value="completed">已完成</option>
        </select>
      </div>
    </div>

    <p v-if="error" class="tip" style="color:#dc2626">{{ error }}</p>

    <!-- 新建弹窗 -->
    <div v-if="showCreate" class="mask" @click.self="showCreate = false">
      <div class="modal">
        <h2>新建论文</h2>
        <form @submit.prevent="handleCreate">
          <label>论文标题（必填，至少 5 字）</label>
          <input v-model="form.title" required />
          <div class="field-with-btn">
            <label>选题方向</label>
            <button type="button" class="btn btn-sm" @click="openRecommend">AI 推荐选题</button>
          </div>
          <input v-model="form.topic" />
          <label>关键词（逗号分隔）</label>
          <input v-model="form.keywords" />
          <label>摘要（可选）</label>
          <textarea v-model="form.abstract"></textarea>
          <div class="modal-btns">
            <button type="button" class="btn" @click="showCreate = false">取消</button>
            <button type="submit" class="btn btn-primary" :disabled="creating">创建</button>
          </div>
        </form>
      </div>
    </div>

    <!-- 选题推荐弹窗 -->
    <div v-if="showRecommend" class="mask" @click.self="showRecommend = false">
      <div class="modal modal-wide">
        <h2>AI 选题推荐</h2>
        <form @submit.prevent="handleRecommend">
          <label>研究方向（必填）</label>
          <input v-model="recommendForm.field" required />
          <label>关键词（逗号分隔）</label>
          <input v-model="recommendForm.keywords" />
          <label>特殊要求（可选）</label>
          <textarea v-model="recommendForm.requirements"></textarea>
          <div class="modal-btns">
            <button type="button" class="btn" @click="showRecommend = false">取消</button>
            <button type="submit" class="btn btn-primary" :disabled="recommending">
              {{ recommending ? '生成中…' : '生成推荐' }}
            </button>
          </div>
        </form>

        <p v-if="recommendError" class="tip error-tip">{{ recommendError }}</p>

        <div v-if="topicCards.length" class="topic-list">
          <div v-for="(t, i) in topicCards" :key="i" class="topic-card">
            <h3>{{ i + 1 }}. {{ t.title }}</h3>
            <div class="topic-body" v-html="renderMarkdown(t.body)"></div>
            <button class="btn btn-primary btn-sm" @click="useTopic(t)">选用这个选题</button>
          </div>
        </div>
        <div v-else-if="recommendResult" class="topic-list">
          <div class="content" v-html="renderMarkdown(recommendResult)"></div>
        </div>
      </div>
    </div>

    <!-- 编辑弹窗 -->
    <EditPaperModal v-if="editing" :paper="editing" @close="editing = null" @saved="onEdited" />

    <!-- 列表 -->
    <p v-if="loading" class="tip">加载中…</p>
    <p v-else-if="papers.length === 0" class="tip">{{ hasFilter ? '没有符合条件的论文' : '还没有论文，点「新建论文」开始第一篇。' }}</p>
    <div v-else class="grid">
      <div v-for="p in papers" :key="p.id" class="card">
        <div class="card-head">
          <span class="paper-title">{{ p.title }}</span>
          <span class="status">{{ statusLabel(p.status) }}</span>
        </div>
        <p v-if="p.topic" class="paper-topic">{{ p.topic }}</p>
        <p class="paper-meta">更新于 {{ p.updated_at.slice(0, 10) }}</p>
        <div class="card-actions">
          <button class="btn btn-sm" @click="goDetail(p)">查看 / 生成</button>
          <button class="btn btn-sm" @click="editing = p">编辑</button>
          <button class="btn btn-danger btn-sm" @click="remove(p)">删除</button>
        </div>
      </div>
    </div>

    <!-- 分页 -->
    <div v-if="total > pageSize" class="pagination">
      <button class="btn btn-sm" :disabled="page <= 1" @click="goPage(page - 1)">上一页</button>
      <span>第 {{ page }} / {{ totalPages }} 页 · 共 {{ total }} 篇</span>
      <button class="btn btn-sm" :disabled="page >= totalPages" @click="goPage(page + 1)">下一页</button>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { listPapers, createPaper, deletePaper, recommendTopic, clearAuth, getUserName } from '../api'
import { renderMarkdown } from '../utils/markdown'
import EditPaperModal from '../components/EditPaperModal.vue'

const router = useRouter()
const papers = ref([])
const loading = ref(false)
const error = ref('')
const showCreate = ref(false)
const creating = ref(false)
const uname = getUserName() || ''

const form = reactive({ title: '', topic: '', keywords: '', abstract: '' })

// 选题推荐相关状态
const showRecommend = ref(false)
const recommending = ref(false)
const recommendError = ref('')
const recommendResult = ref('')
const topicCards = ref([])
const recommendForm = reactive({ field: '', keywords: '', requirements: '' })
const editing = ref(null)

// 列表筛选 / 搜索 / 分页
const searchQuery = ref('')
const statusFilter = ref('')
const page = ref(1)
const pageSize = 20
const total = ref(0)

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize)))
const hasFilter = computed(() => !!(searchQuery.value.trim() || statusFilter.value))

const STATUS_LABEL = {
  draft: '草稿',
  outline: '大纲',
  writing: '撰写中',
  review: '审核中',
  completed: '已完成',
}
function statusLabel(s) { return STATUS_LABEL[s] || s }

async function load() {
  loading.value = true
  error.value = ''
  try {
    const params = { skip: (page.value - 1) * pageSize, limit: pageSize }
    if (searchQuery.value.trim()) params.q = searchQuery.value.trim()
    if (statusFilter.value) params.status_filter = statusFilter.value
    const res = await listPapers(params)
    papers.value = res.data.items
    total.value = res.data.total
  } catch (e) {
    error.value = e.response?.data?.detail || '加载失败'
  } finally {
    loading.value = false
  }
}

function applySearch() {
  page.value = 1
  load()
}

function applyFilter() {
  page.value = 1
  load()
}

function goPage(p) {
  page.value = p
  load()
}

function openCreate() {
  Object.assign(form, { title: '', topic: '', keywords: '', abstract: '' })
  showCreate.value = true
}

async function handleCreate() {
  creating.value = true
  error.value = ''
  try {
    await createPaper({ ...form })
    showCreate.value = false
    await load()
  } catch (e) {
    error.value = e.response?.data?.detail || '创建失败'
  } finally {
    creating.value = false
  }
}

async function remove(p) {
  if (!confirm(`确定删除《${p.title}》？删除后无法恢复。`)) return
  try {
    await deletePaper(p.id)
    await load()
  } catch (e) {
    error.value = e.response?.data?.detail || '删除失败'
  }
}

function goDetail(p) {
  router.push('/papers/' + p.id)
}

function logout() {
  clearAuth()
  router.push('/login')
}

// ---- 选题推荐 ----
function openRecommend() {
  recommendForm.field = form.topic || ''
  recommendForm.keywords = ''
  recommendForm.requirements = ''
  recommendError.value = ''
  recommendResult.value = ''
  topicCards.value = []
  showRecommend.value = true
}

// 把选题 Agent 返回的 markdown 拆成「标题 + 详情」卡片
function parseTopics(md) {
  const topics = []
  if (!md) return topics
  const lines = String(md).replace(/\r\n/g, '\n').split('\n')
  let current = null
  for (const line of lines) {
    const m = line.match(/^\s*###\s*选题\s*\d+\s*[：:]\s*(.+)$/)
    if (m) {
      if (current) topics.push(current)
      current = { title: m[1].trim(), body: '' }
    } else if (current) {
      current.body += line + '\n'
    }
  }
  if (current) topics.push(current)
  return topics
}

async function handleRecommend() {
  if (!recommendForm.field.trim()) {
    recommendError.value = '研究方向不能为空'
    return
  }
  recommending.value = true
  recommendError.value = ''
  try {
    const res = await recommendTopic({ ...recommendForm })
    recommendResult.value = res.data.topics || ''
    if (!recommendResult.value) {
      recommendError.value = 'AI 返回结果为空，请重试'
      topicCards.value = []
    } else {
      topicCards.value = parseTopics(recommendResult.value)
    }
  } catch (e) {
    recommendError.value = e.response?.data?.detail || '推荐失败'
  } finally {
    recommending.value = false
  }
}

function useTopic(t) {
  form.title = t.title
  form.topic = recommendForm.field.trim()
  showRecommend.value = false
}

async function onEdited() {
  editing.value = null
  await load()
}

onMounted(load)
</script>