// 极简 Markdown → HTML（只处理标题/列表/加粗/行内代码/段落）
function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
function renderInline(s) {
  let t = escapeHtml(s)
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>')
  return t
}
export function renderMarkdown(md) {
  if (!md) return ''
  const lines = String(md).replace(/\r\n/g, '\n').split('\n')
  const out = []
  let listType = null    // 当前列表类型：'ul' 或 'ol'
  let paragraph = []     // 暂存普通段落行

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push('<p>' + paragraph.map(renderInline).join('<br>') + '</p>')
      paragraph = []
    }
  }
  const closeList = () => {
    if (listType) { out.push('</' + listType + '>'); listType = null }
  }

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '')

    // 标题 # ~ ######
    const h = line.match(/^(#{1,6})\s+(.*)$/)
    if (h) {
      flushParagraph(); closeList()
      const level = h[1].length
      out.push('<h' + level + '>' + renderInline(h[2].trim()) + '</h' + level + '>')
      continue
    }
    // 无序列表 - / * / +
    const ulItem = line.match(/^\s*[-*+]\s+(.*)$/)
    if (ulItem) {
      flushParagraph()
      if (listType !== 'ul') { closeList(); out.push('<ul>'); listType = 'ul' }
      out.push('<li>' + renderInline(ulItem[1]) + '</li>')
      continue
    }
    // 有序列表 1. / 1)
    const olItem = line.match(/^\s*\d+[.)]\s+(.*)$/)
    if (olItem) {
      flushParagraph()
      if (listType !== 'ol') { closeList(); out.push('<ol>'); listType = 'ol' }
      out.push('<li>' + renderInline(olItem[1]) + '</li>')
      continue
    }
    // 空行：结束当前段落与列表
    if (line.trim() === '') {
      flushParagraph(); closeList()
      continue
    }
    // 普通段落行
    closeList()
    paragraph.push(line)
  }

  flushParagraph()
  closeList()
  return out.join('')
}