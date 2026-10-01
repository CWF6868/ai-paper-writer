<!-- 论文详情页：展示大纲/正文/文献，按钮触发生成并实时显示进度 -->
<template>
  <div class="container">
    <header class="topbar">
      <div class="left">
        <button class="btn btn-sm" @click="goBack">← 返回</button>
        <h1 class="title">{{ paper ? paper.title : '加载中…' }}</h1>
      </div>
      <div class="actions">
        <span v-if="paper" class="status">{{ statusLabel(paper.status) }}</span>
        <button class="btn btn-sm" @click="logout">退出</button>
      </div>
    </header>

    <p v-if="loadError" class="tip" style="color:#dc2626">{{ loadError }}</p>
    <p v-if="loading" class="tip">加载中…</p>

    <template v-else-if="paper">
      <!-- 元信息 -->
      <section class="block meta">
        <p v-if="paper.topic"><b>选题方向：</b>{{ paper.topic }}</p>
        <p v-if="paper.keywords"><b>关键词：</b>{{ paper.keywords }}</p>
        <p v-if="paper.abstract"><b>摘要：</b>{{ paper.abstract }}</p>
        <p class="paper-meta">创建于 {{ paper.created_at.slice(0, 10) }} · 更新于 {{ paper.updated_at.slice(0, 10) }}</p>
      </section>

      <!-- 生成按钮 + 进度 -->
      <section class="block">
        <div class="toolbar" style="margin-bottom:0">
          <button class="btn btn-primary" @click="startGenerate" :disabled="generating">
            {{ generating ? '生成中…' : (paper.content ? '重新生成' : '开始生成') }}
          </button>
          <button class="btn" @click="showEdit = true">编辑</button>
          <button class="btn" :disabled="!paper.content" @click="exportMarkdown">导出 MD</button>
          <button class="btn" :disabled="!paper.content || exporting === 'docx'" @click="exportFile('docx')">导出 Word</button>
          <button class="btn" :disabled="!paper.content || exporting === 'pdf'" @click="exportFile('pdf')">导出 PDF</button>
        </div>
        <p v-if="exportError" class="tip error-tip" style="margin:10px 0 0">{{ exportError }}</p>

        <div v-if="generating" class="progress">
          <h3>生成进度</h3>
          <ul>
            <li v-for="s in steps" :key="s.key" :class="{ done: s.done, active: s.active }">
              {{ s.label }}
            </li>
          </ul>
          <p v-if="genError" class="tip error-tip">{{ genError }}</p>
        </div>
        <p v-if="genError && !generating" class="tip error-tip">{{ genError }}</p>
      </section>

      <!-- 质量评审结果 -->
      <section v-if="qualityScore != null && !generating" class="block review-block">
        <h2>质量评审</h2>
        <p class="score-line">综合评分：<b>{{ Number(qualityScore).toFixed(2) }}</b></p>
        <ul v-if="qualityDimensions.length" class="dim-list">
          <li v-for="(d, i) in qualityDimensions" :key="i">
            <b>{{ d.dimension }}</b>：<span class="dim-score">{{ Number(d.score).toFixed(2) }}</span>
            <span v-if="d.comment" class="dim-comment">— {{ d.comment }}</span>
          </li>
        </ul>
        <p v-if="qualityComment" class="review-comment">{{ qualityComment }}</p>
      </section>

      <!-- 大纲 -->
      <section v-if="outline" class="block">
        <h2>论文大纲</h2>
        <div v-for="(sec, i) in outline.sections" :key="i" class="outline-item">
          <h3>{{ sec.title }}</h3>
          <ul>
            <li v-for="(pt, j) in sec.points" :key="j">{{ pt }}</li>
          </ul>
        </div>
      </section>

      <!-- 正文（Markdown 渲染） -->
      <section v-if="contentHtml" class="block">
        <h2>论文正文</h2>
        <!-- 内容已经过 escapeHtml 转义，v-html 安全 -->
        <div class="content" v-html="contentHtml"></div>
      </section>

      <!-- 参考文献 -->
      <section v-if="paper.references && paper.references.length" class="block">
        <h2>参考文献</h2>
        <ol class="refs">
          <li v-for="r in paper.references" :key="r.id">
            {{ r.authors ? r.authors + '. ' : '' }}{{ r.title }}{{ r.year ? '（' + r.year + '）' : '' }}
          </li>
        </ol>
      </section>

      <EditPaperModal v-if="showEdit" :paper="paper" @close="showEdit = false" @saved="onEdited" />
    </template>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { getPaper, streamGeneratePaper, clearAuth, exportPaper } from '../api'
import { renderMarkdown } from '../utils/markdown'
import EditPaperModal from '../components/EditPaperModal.vue'

const route = useRoute()
const router = useRouter()

const paper = ref(null)
const loading = ref(true)
const loadError = ref('')
const generating = ref(false)
const genError = ref('')
const qualityScore = ref(null)
const qualityComment = ref('')
const qualityDimensions = ref([])
const currentStepKey = ref('')
const doneKeys = ref([])
const showEdit = ref(false)
const exporting = ref('')
const exportError = ref('')

// 工作流节点 → 中文步骤名（与后端 graph_builder 的节点名一一对应）
const STEP_DEFS = [
  { key: 'analyze_topic', label: '选题分析' },
  { key: 'generate_outline', label: '生成大纲' },
  { key: 'write_sections', label: '撰写章节' },
  { key: 'add_references', label: '添加文献' },
  { key: 'polish_paper', label: '全文润色' },
  { key: 'quality_check', label: '质量检查' },
]

const steps = computed(() => STEP_DEFS.map((s) => ({
  ...s,
  done: doneKeys.value.includes(s.key),
  active: currentStepKey.value === s.key,
})))

const STATUS_LABEL = {
  draft: '草稿',
  outline: '大纲',
  writing: '撰写中',
  review: '审核中',
  completed: '已完成',
}
function statusLabel(s) { return STATUS_LABEL[s] || s }

// 解析大纲（后端存的是 JSON 字符串）
const outline = computed(() => {
  const o = paper.value?.outline
  if (!o) return null
  try { return JSON.parse(o) } catch (e) { return null }
})

const contentHtml = computed(() => renderMarkdown(paper.value?.content || ''))

// 把后端的原始错误翻译成用户能看懂的话
function friendlyError(raw) {
  // 后端已给出面向用户的中文提示（如缺少 Key），直接原样展示，不再二次包装
  if (raw && /未配置|缺少|\.env|请在/.test(raw)) return raw
  const s = String(raw || '').toLowerCase()
  if (s.includes('insufficient') || s.includes('balance') || s.includes('quota') || s.includes('余额') || s.includes('欠费')) {
    return 'AI 服务余额不足或额度已用尽，请检查 DeepSeek / DashScope 账户'
  }
  if (s.includes('api key') || s.includes('authentication') || s.includes('unauthorized') || s.includes('invalid_api_key') || s.includes('incorrect api key')) {
    return 'API Key 无效或缺失，请检查后端 .env 配置'
  }
  if (s.includes('timeout') || s.includes('timed out')) {
    return '请求 AI 服务超时，请稍后重试'
  }
  if (s.includes('failed to fetch') || s.includes('networkerror') || s.includes('load failed')) {
    return '无法连接后端服务，请确认后端（8000 端口）已启动'
  }
  if (s.includes('connection') || s.includes('connect') || s.includes('resolve') || s.includes('refused')) {
    return '无法连接 AI 服务，请检查网络或后端是否正常运行'
  }
  if (s.includes('embedding')) {
    return '向量检索（Embedding）服务出错，请检查 DashScope 配置'
  }
  return '生成失败：' + (raw || '未知错误')
}

function parseDimensions(raw) {
  if (!raw) return []
  try { return JSON.parse(raw) } catch (e) { return [] }
}

async function loadPaper(silent) {
  if (!silent) loading.value = true
  loadError.value = ''
  try {
    const res = await getPaper(route.params.id)
    paper.value = res.data
    // 质量评审已持久化，刷新/重进页面也能从详情读回
    qualityScore.value = res.data.quality_score ?? null
    qualityComment.value = res.data.quality_comment || ''
    qualityDimensions.value = parseDimensions(res.data.quality_dimensions)
  } catch (e) {
    loadError.value = e.response?.data?.detail || '加载失败'
  } finally {
    loading.value = false
  }
}

function handleStreamEvent(evt) {
  if (!evt) return
  // 后端出错：{"error": "..."}
  if (evt.error) { genError.value = friendlyError(evt.error); return }
  // 结束事件：{"done": true, "quality_score": ..., "quality_comment": ..., "quality_dimensions": [...]}
  if (evt.done) {
    qualityScore.value = evt.quality_score
    qualityComment.value = evt.quality_comment || ''
    qualityDimensions.value = evt.quality_dimensions || []
    return
  }
  // 普通事件：LangGraph 每次推一个 {节点名: 更新数据}
  for (const nodeName of Object.keys(evt)) {
    currentStepKey.value = nodeName
    if (!doneKeys.value.includes(nodeName)) doneKeys.value.push(nodeName)
    const update = evt[nodeName]
    if (update && typeof update === 'object') {
      if ('quality_comment' in update) qualityComment.value = update.quality_comment || ''
      if ('quality_dimensions' in update) qualityDimensions.value = update.quality_dimensions || []
    }
  }
}

async function startGenerate() {
  if (generating.value) return
  generating.value = true
  genError.value = ''
  qualityScore.value = null
  qualityComment.value = ''
  qualityDimensions.value = []
  doneKeys.value = []
  currentStepKey.value = ''
  try {
    await streamGeneratePaper(paper.value.id, handleStreamEvent)
    // 出错时不要覆盖错误提示去刷新
    if (!genError.value) await loadPaper(true)
  } catch (e) {
    genError.value = friendlyError(e.message || '生成失败')
  } finally {
    generating.value = false
    currentStepKey.value = ''
  }
}

async function onEdited() {
  showEdit.value = false
  await loadPaper(true)
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function exportMarkdown() {
  const p = paper.value
  const lines = ['# ' + (p.title || ''), '']
  if (p.topic) lines.push('**选题方向**：' + p.topic)
  if (p.keywords) lines.push('**关键词**：' + p.keywords)
  if (p.abstract) lines.push('', '## 摘要', '', p.abstract)
  lines.push('', '---', '')
  if (p.content) lines.push(p.content)
  if (p.references && p.references.length) {
    lines.push('', '## 参考文献', '')
    p.references.forEach((r, i) => {
      lines.push(`${i + 1}. ${r.authors ? r.authors + '. ' : ''}${r.title}${r.year ? '（' + r.year + '）' : ''}`)
    })
  }
  downloadBlob(new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' }), (p.title || '论文').slice(0, 40) + '.md')
}

async function exportFile(format) {
  exporting.value = format
  exportError.value = ''
  try {
    const res = await exportPaper(paper.value.id, format)
    const ext = format === 'docx' ? 'docx' : 'pdf'
    downloadBlob(res.data, (paper.value.title || '论文').slice(0, 40) + '.' + ext)
  } catch (e) {
    exportError.value = '导出失败，请稍后重试'
  } finally {
    exporting.value = ''
  }
}

function goBack() { router.push('/papers') }
function logout() { clearAuth(); router.push('/login') }

onMounted(() => loadPaper())
</script>