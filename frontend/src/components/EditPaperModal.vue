<!-- 论文编辑弹窗：元信息 + 正文（正文仅在有 content 时显示） -->
<template>
  <div class="mask" @click.self="$emit('close')">
    <div class="modal modal-wide">
      <h2>编辑论文</h2>
      <form @submit.prevent="save">
        <label>论文标题（至少 5 字）</label>
        <input v-model="form.title" required />

        <label>选题方向</label>
        <input v-model="form.topic" />

        <label>关键词（逗号分隔）</label>
        <input v-model="form.keywords" />

        <label>摘要</label>
        <textarea v-model="form.abstract"></textarea>

        <template v-if="hasContent">
          <div class="field-with-btn">
            <label>正文（Markdown）</label>
            <button type="button" class="btn btn-sm" @click="preview = !preview">
              {{ preview ? '返回编辑' : '预览' }}
            </button>
          </div>
          <textarea v-if="!preview" v-model="form.content" class="content-editor"></textarea>
          <div v-else class="content preview-box" v-html="renderMarkdown(form.content)"></div>
        </template>

        <p v-if="error" class="tip error-tip">{{ error }}</p>

        <div class="modal-btns">
          <button type="button" class="btn" @click="$emit('close')">取消</button>
          <button type="submit" class="btn btn-primary" :disabled="saving">保存</button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch } from 'vue'
import { updatePaper } from '../api'
import { renderMarkdown } from '../utils/markdown'

const props = defineProps({
  paper: { type: Object, required: true },
})
const emit = defineEmits(['close', 'saved'])

const form = reactive({ title: '', topic: '', keywords: '', abstract: '', content: '' })
const saving = ref(false)
const error = ref('')
const preview = ref(false)

// 列表页传进来的是 PaperResponse（无 content），此时只编辑元信息
const hasContent = computed(() => props.paper.content != null)

function init() {
  form.title = props.paper.title || ''
  form.topic = props.paper.topic || ''
  form.keywords = props.paper.keywords || ''
  form.abstract = props.paper.abstract || ''
  form.content = props.paper.content || ''
  error.value = ''
  preview.value = false
}
watch(() => props.paper, init, { immediate: true })

async function save() {
  saving.value = true
  error.value = ''
  try {
    const payload = { ...form }
    if (!hasContent.value) delete payload.content
    await updatePaper(props.paper.id, payload)
    emit('saved')
  } catch (e) {
    error.value = e.response?.data?.detail || '保存失败'
  } finally {
    saving.value = false
  }
}
</script>