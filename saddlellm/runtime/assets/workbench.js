/* Evidence-backed, read-only study browser. No model or training is started here. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const VIEWS = {
    data: ['数据说明', '先看原始记录，再对照模型实际输入和标准答案。'],
    checks: ['训练前检查', '查看实际检查范围、检查结果，以及当时执行的命令。'],
    experiments: ['实验方案', '对照数据、方法和训练成本，理解每个实验改了什么。'],
    training: ['训练过程', '按实验查看各训练阶段的指标、日志和问题记录。'],
    results: ['效果对比', '用同一批样本比较训练前后，找出改对和改错的具体例子。'],
    reports: ['报告与复现', '下载已有报告和证据文件，复制命令复现实验过程。']
  };
  const OUTCOMES = {all: '全部样本', improved: '改对了', regressed: '改错了', both_correct: '一直正确', both_wrong: '一直错误', unpaired: '未配对（缺少记录或同条件基线）'};
  const STATUS = {completed: ['已完成', 'success'], ok: ['通过', 'success'], passed: ['通过', 'success'], success: ['通过', 'success'], failed: ['失败', 'error'], error: ['失败', 'error'], running: ['记录为运行中', 'warning'], pending: ['等待中', 'neutral'], unknown: ['状态未知', 'neutral'], warning: ['有提醒', 'warning'], missing: ['缺少记录', 'warning'], missing_evidence: ['缺少证据', 'warning'], historical: ['历史记录', 'neutral']};
  const SPLITS = {train: '本次选中的训练样本', test: '本次选中的测试样本', validation: '本次验证样本记录', original_train: '原始文件 · 训练划分', original_test: '原始文件 · 测试划分'};
  let state = readURL(), studies = [], study = null, controller = null, renderSerial = 0, toastTimer;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function text(value, missing = '未记录') {
    if (value === undefined || value === null || value === '') return missing;
    return typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
  }
  function finite(value) { return typeof value === 'number' && Number.isFinite(value); }
  function metricValue(metrics, phase) { const value = metrics && metrics[phase]; return finite(value) ? value : value && value[metrics.primary_metric]; }
  function modelName() { return study.model && typeof study.model === 'object' ? study.model.name || '基础模型未记录' : study.model || '基础模型未记录'; }
  function percentage(value) { return finite(value) ? `${(value * 100).toFixed(2)}%` : '—'; }
  function delta(value) { return finite(value) ? `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} 个百分点` : '—'; }
  function numeral(value) { return finite(value) ? value !== 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toLocaleString('zh-CN', {maximumFractionDigits: 6}) : '—'; }
  function duration(value) { return finite(value) ? `${(value / 60).toFixed(1)} 分钟` : '未记录'; }
  function bytes(value) { return finite(value) ? value > 1048576 ? `${(value / 1048576).toFixed(1)} MB` : `${(value / 1024).toFixed(1)} KB` : ''; }
  function badge(status) { const [label, style] = STATUS[status] || [text(status, '未记录'), 'neutral']; return el('span', `badge ${style}`, label); }
  function array(value) { return Array.isArray(value) ? value : []; }
  function section(title, subtitle) { const node = el('section', 'card'); node.append(el('h2', '', title)); if (subtitle) node.append(el('p', 'muted small', subtitle)); return node; }
  function button(label, action, secondary = true) { const node = el('button', `button${secondary ? ' secondary' : ''}`, label); node.type = 'button'; node.addEventListener('click', action); return node; }
  function list(items, empty = '没有此项记录。') { const node = el('ul', 'kv-list'); if (!items.length) return el('p', 'muted small', empty); items.forEach(item => node.append(el('li', '', text(item)))); return node; }
  function definition(values) { const dl = el('dl', 'definition-list'); values.forEach(([name, value]) => { dl.append(el('dt', '', name), el('dd', '', text(value))); }); return dl; }
  function details(label, value) { const node = el('details'); node.append(el('summary', '', label), el('pre', 'code-block', text(value))); return node; }
  function table(headers, rows, className = '') {
    const wrap = el('div', 'table-wrap'), tableNode = el('table', className), thead = el('thead'), head = el('tr'), body = el('tbody');
    wrap.tabIndex = 0; wrap.setAttribute('aria-label', `${headers.join('、')}表格，可横向滚动`);
    headers.forEach(item => head.append(el('th', '', item))); thead.append(head);
    rows.forEach(items => { const row = el('tr'); items.forEach(item => { const td = el('td'); if (item instanceof Node) td.append(item); else td.textContent = text(item, '—'); row.append(td); }); body.append(row); });
    tableNode.append(thead, body); wrap.append(tableNode); return wrap;
  }
  function notice(message, warning = false) { const node = el('div', `notice${warning ? ' warning' : ''}`); node.append(el('p', '', message)); return node; }
  function showToast(message) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3000); }
  async function copy(value) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(value);
      else { const input = el('textarea'); input.value = value; input.style.position = 'fixed'; input.style.opacity = '0'; document.body.append(input); input.select(); const ok = document.execCommand('copy'); input.remove(); if (!ok) throw new Error('copy'); }
      showToast('已复制到剪贴板');
    } catch (_) { showToast('复制失败，请展开记录后手动选择文字复制。'); }
  }
  function downloadText(name, value) { const href = URL.createObjectURL(new Blob([value], {type: 'text/plain;charset=utf-8'})), anchor = el('a'); anchor.href = href; anchor.download = name; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(href), 1000); }
  function readURL() {
    const url = new URL(location.href), p = url.searchParams;
    return {view: VIEWS[p.get('view')] ? p.get('view') : 'data', study: p.get('study') || '', run: p.get('run') || '', task: ['A', 'B'].includes(p.get('task')) ? p.get('task') : 'A', outcome: OUTCOMES[p.get('outcome')] ? p.get('outcome') : 'all', query: (p.get('query') || '').slice(0, 500), split: SPLITS[p.get('split')] ? p.get('split') : 'train', offset: Math.floor(Math.max(0, Number(p.get('offset')) || 0))};
  }
  function saveURL(replace = false) { const url = new URL(location.href); Object.entries(state).forEach(([name, value]) => value === '' || (name === 'offset' && value === 0) ? url.searchParams.delete(name) : url.searchParams.set(name, String(value))); (replace ? history.replaceState : history.pushState).call(history, {}, '', url); }
  async function update(changes, replace = false) { const changedStudy = changes.study && changes.study !== state.study, changedView = changes.view && changes.view !== state.view, changedPage = Object.hasOwn(changes, 'offset') && changes.offset !== state.offset; state = {...state, ...changes}; if (changedStudy) { study = null; state.run = ''; state.offset = 0; } saveURL(replace); await render(); if (changedStudy || changedView) { window.scrollTo(0, 0); $('main').focus({preventScroll: true}); } else if (changedPage && Object.keys(changes).every(name => name === 'offset')) { const samples = $('samplesSection'); if (samples) samples.scrollIntoView({block: 'start'}); } }
  function baseURL() { return `/api/workbench/studies/${encodeURIComponent(state.study)}`; }
  async function get(path, signal) {
    const requestController = new AbortController(); let timedOut = false;
    const abortRequest = () => requestController.abort();
    if (signal) { if (signal.aborted) abortRequest(); else signal.addEventListener('abort', abortRequest, {once: true}); }
    const timer = setTimeout(() => { timedOut = true; requestController.abort(); }, 20000);
    try {
      const response = await fetch(path, {signal: requestController.signal, cache: 'no-store'});
      if (!response.ok) { let message = `读取失败（HTTP ${response.status}）`; try { const payload = await response.json(); if (typeof payload.detail === 'string') message += `：${payload.detail}`; } catch (_) {} throw new Error(message); }
      return await response.json();
    } catch (error) { if (timedOut) throw new Error('读取超过 20 秒，请重试或检查服务日志'); throw error; }
    finally { clearTimeout(timer); if (signal) signal.removeEventListener('abort', abortRequest); }
  }
  function stateBox(title, message, error = false, retry) { const box = el('div', `state-box${error ? ' error' : ''}`); if (!error && !retry) box.append(el('span', 'spinner')); box.append(el('h2', '', title), el('p', '', message)); if (retry) box.append(button('重新读取', retry, false)); return box; }
  function artifactLink(artifact, label = '下载') { const link = el('a', 'button secondary small', label); link.href = `${baseURL()}/artifacts/${encodeURIComponent(artifact.id)}`; link.setAttribute('download', ''); return link; }
  function artifacts(items) {
    if (!items.length) return el('p', 'empty-inline', '此处没有可下载的已存档文件。');
    const node = el('div', 'artifact-list');
    items.forEach(item => { const row = el('div', 'artifact-row'), info = el('div'); info.append(el('strong', '', item.name || item.id), el('p', '', [item.format, item.kind, bytes(item.bytes)].filter(Boolean).join(' · '))); if (item.description) info.append(el('p', '', item.description)); row.append(info, artifactLink(item)); node.append(row); });
    return node;
  }
  function commands(items) {
    const node = el('div', 'stack');
    if (!items.length) return el('p', 'empty-inline', '未发现命令记录。没有记录的步骤不会被补写为实际执行。');
    items.forEach(item => {
      const block = el('div', 'command-card'), head = el('div', 'command-head'), actions = el('div', 'inline-controls');
      const value = item.display || (Array.isArray(item.command) ? item.command.map(arg => /\s/.test(String(arg)) ? JSON.stringify(arg) : String(arg)).join(' ') : text(item.command, ''));
      head.append(el('strong', '', item.title || item.id || '命令记录')); actions.append(button('复制命令', () => copy(value)), button('下载命令', () => downloadText(`${item.id || 'saddlellm-command'}.txt`, `${value}\n`))); head.append(actions);
      block.append(head, el('div', 'command-meta', {actual: '实际执行命令', equivalent: '当前等价复现命令', 'recorded-step': '存档步骤'}[item.kind] || '存档命令'), el('pre', 'code-block', value || '此步骤没有可复制的命令。'));
      if (!value) actions.querySelectorAll('button').forEach(control => { control.disabled = true; });
      if (item.explanation) block.append(el('p', 'small muted', item.explanation));
      if (item.result) block.append(el('p', 'small', `执行结果：${text(item.result)}`));
      if (item.evidence_level || item.evidence) block.append(el('div', 'command-meta', `证据：${[item.evidence_level, text(item.evidence, '')].filter(Boolean).join(' · ')}`));
      node.append(block);
    }); return node;
  }
  function issues(items) {
    if (!items.length) return el('p', 'muted small', '没有已存档的问题记录。此状态不代表完成了额外检查。');
    const node = el('ul', 'issue-list'); items.forEach(item => { const row = el('li'); if (typeof item === 'string') row.append(el('p', 'small', item)); else { row.append(el('strong', '', item.title || item.name || item.id || '问题记录')); const fields = [[item.description_zh || item.description || item.message || item.detail, '记录'], [item.impact_zh || item.impact, '影响'], [item.resolution_zh || item.resolution, '处理'], [{fixed: '已修复', mitigated: '已缓解', open: '尚未解决', documented: '已记录'}[item.status] || item.status, '状态']]; fields.forEach(([value, label]) => { if (value) row.append(el('p', 'small muted', `${label}：${text(value)}`)); }); if (!fields.some(([value]) => value)) row.append(el('pre', 'code-block', text(item))); } node.append(row); }); return node;
  }
  function selectControl(label, id, options, value, onChange) { const field = el('div', 'field'), caption = el('label', '', label), select = el('select'); caption.htmlFor = id; select.id = id; options.forEach(([key, name]) => { const option = el('option', '', name); option.value = key; select.append(option); }); select.value = value; select.addEventListener('change', () => onChange(select.value)); field.append(caption, select); return field; }
  function searchControl(onSearch) { const form = el('form', 'search-form'), field = el('div', 'field'), label = el('label', '', '查找样本'), input = el('input'), submit = el('button', 'button', '查找'); input.id = 'sampleQuery'; input.type = 'search'; input.maxLength = 500; input.value = state.query; input.placeholder = '输入题目、标签或样本编号'; label.htmlFor = input.id; submit.type = 'submit'; field.append(label, input); form.append(field, submit); form.addEventListener('submit', event => { event.preventDefault(); onSearch(input.value.trim()); }); if (state.query) form.append(button('清除', () => onSearch(''))); return form; }
  function taskOptions() { return array(study.datasets).map(item => [item.id, `${item.id} · ${item.name || item.task}`]); }
  function runControl() { return selectControl('选择实验', 'runSelect', array(study.runs).map(run => [run.id, run.name || run.id]), state.run, run => update({run, offset: 0})); }
  function taskControl() { return selectControl('数据任务', 'taskSelect', taskOptions().length ? taskOptions() : [['A', '任务 A'], ['B', '任务 B']], state.task, task => update({task, offset: 0})); }
  function pagination(payload) { const node = el('div', 'pagination'), total = Number(payload.total) || 0, offset = Number(payload.offset) || state.offset, limit = Number(payload.limit) || 10, controls = el('div', 'inline-controls'); node.append(el('span', '', total ? `共 ${total.toLocaleString()} 条 · 当前 ${offset + 1}–${Math.min(offset + limit, total)} 条` : '共 0 条')); const prev = button('上一页', () => update({offset: Math.max(0, offset - limit)})), next = button('下一页', () => update({offset: offset + limit})); prev.disabled = offset <= 0; next.disabled = offset + limit >= total; controls.append(prev, next); node.append(controls); return node; }

  const SVG_NS = 'http://www.w3.org/2000/svg';
  function svgNode(tag, attrs = {}, content) { const node = document.createElementNS(SVG_NS, tag); Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, String(value))); if (content !== undefined) node.textContent = String(content); return node; }
  function barChart(rows, title, percent = true, paired = false) {
    const usable = rows.filter(row => finite(row.value) || finite(row.pre)); if (!usable.length) return el('p', 'empty-inline', '没有可绘制的实测数值。');
    const width = 700, left = 175, right = 590, rowHeight = paired ? 53 : 38, height = usable.length * rowHeight + 22, max = percent ? 1 : Math.max(...usable.map(row => row.value), 1), svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': title});
    svg.append(svgNode('title', {}, title), svgNode('desc', {}, usable.map(row => `${row.label}：${paired ? `训练前 ${percentage(row.pre)}，训练后 ` : ''}${percent ? percentage(row.value) : numeral(row.value)}`).join('；')));
    usable.forEach((row, index) => { const y = index * rowHeight + 22; svg.append(svgNode('text', {x: 0, y: y + 4, class: 'chart-label'}, row.label.length > 21 ? `${row.label.slice(0, 20)}…` : row.label)); if (paired && finite(row.pre)) { svg.append(svgNode('rect', {x: left, y: y - 11, width: Math.max(0, Math.min(1, row.pre / max)) * (right - left), height: 7, rx: 3, fill: '#a8b9b2'}), svgNode('text', {x: right + 9, y: y - 4}, percentage(row.pre))); } const cy = paired ? y + 9 : y - 5; if (finite(row.value)) svg.append(svgNode('rect', {x: left, y: cy, width: Math.max(0, Math.min(1, row.value / max)) * (right - left), height: 9, rx: 3, fill: '#226d5a'}), svgNode('text', {x: right + 9, y: cy + 8, class: 'chart-value'}, percent ? percentage(row.value) : numeral(row.value))); });
    const wrap = el('div', 'chart-scroll'); wrap.append(svg); return wrap;
  }
  function lineChart(points, keys, title) {
    const values = points.flatMap(point => keys.map(key => point[key])).filter(finite); if (!values.length) return el('p', 'empty-inline', '这个阶段没有存档曲线。');
    const width = 560, height = 230, left = 72, right = 537, top = 21, bottom = 185, maxStep = Math.max(...points.map(point => finite(point.step) ? point.step : 0), 1), low = Math.min(...values), high = Math.max(...values), range = Math.max(high - low, Math.abs(high) * .1, 1e-12), ratio = keys.every(key => ['accuracy', 'train_accuracy', 'validation_accuracy', 'rewards/accuracies', 'informative_group_rate'].includes(key)), min = ratio ? Math.max(0, low - range * .08) : low - range * .08, max = ratio ? Math.min(1, high + range * .08) : high + range * .08, svg = svgNode('svg', {viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': title});
    svg.append(svgNode('title', {}, title), svgNode('desc', {}, `训练记录中的 ${keys.join('、')}；完整数值在下方数据表。横轴为训练步数。`));
    for (let i = 0; i < 4; i++) { const y = top + (bottom - top) * i / 3, tick = max - (max - min) * i / 3, label = tick !== 0 && (Math.abs(tick) < .001 || Math.abs(tick) >= 10000) ? tick.toExponential(2) : tick.toPrecision(3); svg.append(svgNode('line', {x1: left, y1: y, x2: right, y2: y, stroke: '#e4ebf0'}), svgNode('text', {x: left - 8, y: y + 4, 'text-anchor': 'end'}, label)); }
    keys.forEach((key, index) => { const pointsText = points.filter(point => finite(point[key])).map(point => `${left + (finite(point.step) ? point.step : 0) / maxStep * (right - left)},${bottom - (point[key] - min) / (max - min) * (bottom - top)}`).join(' '); if (pointsText) { svg.append(svgNode('polyline', {points: pointsText, fill: 'none', stroke: index ? '#b9670a' : '#226d5a', 'stroke-width': 2.3, 'stroke-linejoin': 'round'})); points.filter(point => finite(point[key])).forEach(point => { svg.append(svgNode('circle', {cx: left + (finite(point.step) ? point.step : 0) / maxStep * (right - left), cy: bottom - (point[key] - min) / (max - min) * (bottom - top), r: 2.5, fill: index ? '#b9670a' : '#226d5a'})); }); } });
    svg.append(svgNode('text', {x: left, y: 210}, '0 step'), svgNode('text', {x: right, y: 210, 'text-anchor': 'end'}, `${maxStep} step`)); return svg;
  }
  function metricCards(run, task) { const metrics = run && run.metrics && run.metrics[task], grid = el('div', 'grid four'); [['训练前', percentage(metricValue(metrics, 'pre'))], ['训练后', percentage(metricValue(metrics, 'post'))], ['变化', delta(metricValue(metrics, 'delta'))], ['运行耗时（含评测）', duration(run && run.duration_s)]].forEach(([name, value]) => { const card = el('div', 'card'); card.append(el('div', 'metric-label', name), el('div', 'metric-value', value)); if (name !== '运行耗时（含评测）') card.append(el('div', 'metric-help', metrics && metrics.primary_label || '没有主评测指标记录')); grid.append(card); }); return grid; }

  async function dataView(signal) {
    const stack = el('div', 'stack'), datasets = array(study.datasets), grid = el('div', 'grid');
    if (study.data_context) stack.append(notice(study.data_context));
    datasets.forEach(dataset => { const card = section(`${dataset.id} · ${dataset.name}`, dataset.description); card.append(definition([['任务', dataset.task], ['原始格式', dataset.format], ['原始记录数', numeral(dataset.raw_records)], ['官方数据划分', dataset.official_splits], ['实验数据划分', dataset.derived_splits], ['本次选取', dataset.selected]])); if (dataset.field_usage) { const fields = el('details'); fields.append(el('summary', '', '字段怎样用于训练'), table(['字段', '含义 / 用途'], Object.entries(dataset.field_usage))); card.append(fields); } if (dataset.labels) { const labels = el('details'); labels.append(el('summary', '', '标签及其含义')); labels.append(typeof dataset.labels === 'object' ? table(['标签', '含义'], Object.entries(dataset.labels)) : el('p', 'small', dataset.labels)); card.append(labels); } if (dataset.files) card.append(details('数据文件与版本', dataset.files)); if (dataset.split_explanation) card.append(el('p', 'small muted', dataset.split_explanation)); grid.append(card); });
    if (datasets.length) stack.append(grid);
    stack.append(notice('通常，训练集用于学习、验证集用于选择方案、测试集用于验收效果。划分字段本身不代表已执行验证或选择 checkpoint；本轮实际使用情况以配置和训练记录为准。下方对照原始记录、存档输入和标准答案。'));
    if (study.validation_usage || study.data_usage_note) stack.append(notice(study.validation_usage || study.data_usage_note));
    const controls = section('查看真实样本'), form = el('div', 'control-group'); form.append(taskControl(), selectControl('数据范围', 'splitSelect', Object.entries(SPLITS), state.split, split => update({split, offset: 0})), searchControl(query => update({query, offset: 0}))); controls.append(form); stack.append(controls);
    const params = new URLSearchParams({source: state.task, split: state.split, offset: state.offset, limit: 10, query: state.query}), payload = await get(`${baseURL()}/data?${params}`, signal);
    const samples = section('原始记录 → 模型实际输入 → 标准答案', `${state.task} 数据 · ${SPLITS[state.split]} · 数据和输入的可用范围以存档为准。`);
    samples.id = 'samplesSection';
    if (array(payload.notes).length) samples.append(list(payload.notes));
    if (!array(payload.rows).length) samples.append(el('p', 'empty-inline', state.query ? '没有匹配的样本，清除搜索或切换数据划分后再试。' : '此数据划分没有可读取的样本记录。'));
    array(payload.rows).forEach(row => {
      const sample = el('article', 'sample-card'), heading = el('div', 'sample-row-label'); heading.append(el('strong', 'small', `样本 ${row.id}`), el('span', 'badge neutral', `${row.source} · ${row.split}`)); if (row.original_split) heading.append(el('span', 'small muted', `原始划分：${row.original_split}`)); sample.append(heading);
      const columns = el('div', 'sample-columns'); [['原始记录', row.raw, ''], ['模型实际输入', row.input, 'input-column'], ['标准答案 / 目标标签', row.target, '']].forEach(([name, value, cls]) => { const column = el('div', `sample-column ${cls}`); column.append(el('h3', '', name), el('div', `sample-value${name.startsWith('标准') ? ' sample-target' : ''}`, text(value, name === '模型实际输入' ? '没有存档的实际 prompt，不补写为真实输入。' : '未记录'))); columns.append(column); }); sample.append(columns); if (row.parsed_problem) sample.append(details('展开原始 problem 内容', row.parsed_problem)); if (array(row.notes).length) sample.append(list(row.notes)); samples.append(sample);
    }); samples.append(pagination(payload)); stack.append(samples);
    const selected = datasets.find(item => item.id === state.task); if (selected && selected.label_counts) { const counts = section('原始数据的标签分布', `这是完整原始数据的标签出现次数，和当前分页、选中样本数量不同。${String(selected.task).includes('多标签') ? '一条多标签记录可同时计入多个标签；字母含义以当前问题的候选主题为准。' : ''}`); const entries = Object.entries(selected.label_counts).filter(([, count]) => finite(count)); if (entries.length) counts.append(barChart(entries.map(([label, value]) => ({label, value})), '原始数据的标签出现次数', false), table(['标签', '出现次数'], entries)); else counts.append(details('标签分布原始记录', selected.label_counts)); stack.append(counts); }
    return stack;
  }
  async function checksView() {
    const stack = el('div', 'stack'); stack.append(notice('doctor 通过表示其明确列出的检查项通过。抽样检查、全量检查和训练时检查的范围不同，请以每条记录的范围为准。'));
    const checks = array(study.checks); if (!checks.length) stack.append(section('检查记录', '没有找到 doctor 存档。此页面不会把缺失记录视作通过。'));
    checks.forEach(check => { const card = section(check.name || '训练前检查'), header = el('div', 'card-heading'); header.append(el('strong', 'small', check.id), badge(check.status)); card.append(header, definition([['执行阶段 / 时间', check.timing], ['检查范围', check.scope], ['结果摘要', check.summary]])); const rows = array(check.checks).map(item => [item.name, typeof item.ok === 'boolean' ? badge(item.ok ? 'passed' : 'failed') : badge('unknown'), text(item.value, '—')]); if (rows.length) card.append(table(['检查项', '结果', '实测值 / 说明'], rows)); if (array(check.errors).length) card.append(el('h3', '', '失败项'), list(check.errors)); if (array(check.warnings).length) card.append(el('h3', '', '提醒'), list(check.warnings)); stack.append(card); });
    const commandSection = section('检查命令与结果'); commandSection.append(commands(array(study.commands).filter(item => /doctor|check|检查/i.test(`${item.id} ${item.title} ${item.display}`)))); stack.append(commandSection); return stack;
  }
  async function experimentsView() {
    const stack = el('div', 'stack'), config = section('本项目的实验设置'); config.append(definition([['基础模型', modelName()], ['实验项目', study.project], ['实验数量', array(study.runs).length], ['比较说明', study.comparison_note || '在同一项目内核对评测数据、抽样方式和训练预算后比较。']])); if (study.config) config.append(details('展开存档配置', study.config)); stack.append(config);
    const sectionNode = section('各方法的实际结果', '训练前、训练后和变化来自存档评测。数值越高是否越好，以指标定义为准。'), controls = el('div', 'control-group'); controls.append(taskControl()); sectionNode.append(controls);
    const runs = array(study.runs), rows = runs.map(run => { const name = el('div'); name.append(el('strong', 'row-name', run.name || run.id), el('span', 'row-meta', `${run.method || '未记录方法'} · ${run.source || '本地存档'}`)); const metrics = run.metrics && run.metrics[state.task], actions = el('div', 'inline-controls'); actions.append(button('训练记录', () => update({view: 'training', run: run.id, offset: 0})), button('逐例对比', () => update({view: 'results', run: run.id, offset: 0}))); return [name, badge(run.status), percentage(metricValue(metrics, 'pre')), percentage(metricValue(metrics, 'post')), delta(metricValue(metrics, 'delta')), duration(run.duration_s), actions]; });
    if (rows.length) sectionNode.append(table(['实验 / 方法', '状态', '训练前', '训练后', '变化', '运行耗时（含评测）', '查看'], rows, 'run-table')); else sectionNode.append(el('p', 'empty-inline', '没有可读取的实验记录。')); stack.append(sectionNode);
    const chart = section('训练前后对照', runs.find(run => run.metrics && run.metrics[state.task])?.metrics[state.task].primary_label || '当前任务没有主评测指标记录。'); chart.append(el('p', 'small muted', '灰绿色：训练前；深绿色：训练后。完整数值已列于上表。'), barChart(runs.map(run => ({label: run.name || run.id, value: metricValue(run.metrics && run.metrics[state.task], 'post'), pre: metricValue(run.metrics && run.metrics[state.task], 'pre')})), '各实验训练前后的评测指标', true, true)); stack.append(chart);
    if (array(study.caveats).length) { const limits = section('怎样理解这组对比'); limits.append(list(study.caveats)); stack.append(limits); } return stack;
  }
  async function trainingView(signal) {
    const stack = el('div', 'stack'), controls = section('选择实验记录'); controls.append(runControl()); stack.append(controls); if (!state.run) { stack.append(el('p', 'empty-inline', '此项目没有实验记录。')); return stack; }
    const run = await get(`${baseURL()}/runs/${encodeURIComponent(state.run)}`, signal), info = section(run.name || run.id), head = el('div', 'card-heading'); head.append(el('span', 'small muted', run.method || '方法未记录'), badge(run.status)); info.append(head, definition([['记录来源', run.source], ['随机种子', run.seed], ['运行耗时（含评测）', duration(run.duration_s)], ['峰值显存', finite(run.peak_cuda_allocated_gib) ? `${run.peak_cuda_allocated_gib.toFixed(2)} GiB` : '未记录']])); stack.append(info);
    stack.append(notice('每个方法的 loss 含义不同。曲线用于理解当前实验的训练过程；方法优劣请查看共同任务的评测指标。存档状态不代表服务器当前仍在运行。'));
    const stages = array(run.history); if (!stages.length) stack.append(section('训练曲线', '没有发现可读取的逐步训练指标。该结果不补画曲线。'));
    const chartGrid = el('div', 'run-detail-grid'); stages.forEach(stage => {
      const points = array(stage.points), keys = [...new Set(points.flatMap(point => Object.keys(point)))].filter(key => !['step', 'epoch', 'elapsed_s', 'timestamp', 'wall_time'].includes(key) && points.some(point => finite(point[key]))), preferred = keys.filter(key => ['loss', 'train_loss', 'eval_loss', 'validation_loss', 'eval_reward', 'reward', 'reward_mean', 'accepted_reward_mean', 'informative_group_rate', 'kl_divergence', 'learning_rate', 'lr', 'accuracy', 'train_accuracy', 'validation_accuracy', 'rewards/accuracies', 'rewards/margins'].includes(key));
      const chartKeys = preferred.length ? preferred : keys.slice(0, 4), lossKeys = chartKeys.filter(key => /(^|_)loss$/.test(key)), groups = []; if (lossKeys.length) groups.push(lossKeys); chartKeys.filter(key => !lossKeys.includes(key)).forEach(key => groups.push([key])); if (!groups.length) groups.push([]);
      const metricLabels = {loss: '逐步训练 Loss', train_loss: '全程平均训练 Loss', eval_loss: '验证 Loss', validation_loss: '验证 Loss', reward_mean: '采样回答的平均奖励', accepted_reward_mean: '参与更新回答的平均奖励', informative_group_rate: '有奖励差异的组比例', kl_divergence: '相对参考模型的 KL', learning_rate: '学习率', lr: '学习率', 'rewards/accuracies': '偏好配对的训练判别率', 'rewards/margins': '偏好奖励差值'};
      groups.forEach(group => { const card = section(`${stage.name || stage.stage || '训练'} · ${group.map(key => metricLabels[key] || key).join(' / ') || '指标记录'}`); card.classList.add('chart-card'); if (stage.shared) card.append(el('p', 'small muted', '本阶段复用同来源 SFT 的训练记录，没有重复训练 SFT。')); const legend = el('div', 'legend'); group.forEach((key, index) => { const item = el('span', 'legend-item'), swatch = el('span', `legend-swatch${index ? ' orange' : ''}`); item.append(swatch, el('span', '', key)); legend.append(item); }); card.append(legend, lineChart(points, group, `${stage.stage} 阶段 ${group.join(' / ')} 曲线`), el('p', 'chart-caption', stage.explanation || '记录的数值展示在下表，缺少的值留空。')); if (stage.source) card.append(el('p', 'chart-caption', `来源：${stage.source}`)); const recordDetails = el('details'); recordDetails.append(el('summary', '', `展开 ${points.length} 条真实数值`), table(['step', ...group], points.map(point => [numeral(point.step), ...group.map(key => numeral(point[key]))]))); card.append(recordDetails); chartGrid.append(card); });
    }); if (stages.length) stack.append(chartGrid);
    const commandSection = section('实际步骤与命令'); commandSection.append(commands(array(run.commands))); stack.append(commandSection);
    const issueSection = section('训练中遇到的问题'); issueSection.append(issues(array(run.issues))); stack.append(issueSection);
    const logs = run.logs || run.log_tail; if (logs) { const evidence = section('日志证据', '这里展示已存档的日志内容。'); evidence.append(el('pre', 'code-block', Array.isArray(logs) ? logs.join('\n') : text(logs))); stack.append(evidence); }
    if (run.source_file || run.training_summary) { const evidence = section('训练记录证据', '逐步曲线直接读取训练存档，没有从最终成绩推算过程。'); if (run.source_file) evidence.append(el('p', 'small muted', `运行记录：${run.source_file}`)); if (run.training_summary) evidence.append(details('展开阶段摘要与诊断', run.training_summary)); if (stages.length) evidence.append(button('下载原始训练指标', () => downloadText(`${run.id}-history.json`, JSON.stringify(stages, null, 2)))); stack.append(evidence); }
    const attached = array(run.artifacts); if (attached.length) { const evidence = section('此实验的证据文件'); evidence.append(artifacts(attached)); stack.append(evidence); } return stack;
  }
  async function resultsView(signal) {
    const stack = el('div', 'stack'), controls = section('选择对比范围'), form = el('div', 'control-group'); form.append(runControl(), taskControl(), selectControl('样本变化', 'outcomeSelect', Object.entries(OUTCOMES), state.outcome, outcome => update({outcome, offset: 0})), searchControl(query => update({query, offset: 0}))); controls.append(form); stack.append(controls); if (!state.run) { stack.append(el('p', 'empty-inline', '此项目没有实验结果。')); return stack; }
    const run = array(study.runs).find(item => item.id === state.run); stack.append(metricCards(run, state.task));
    const metrics = run && run.metrics && run.metrics[state.task];
    if (metrics && (metrics.paired || metrics.breakdown)) { const analysis = section('理解变化的幅度', '下面是存档的数据分析，置信区间仅描述这批评测样本的不确定性。'); const paired = metrics.paired || {}, clustered = metrics.organization_paired || {}; if (finite(paired.improved_cases) || finite(paired.regressed_cases)) analysis.append(definition([['改对 / 改错的样本', `${numeral(paired.improved_cases)} / ${numeral(paired.regressed_cases)}`], ['逐案例 95% 变化区间', finite(paired.ci95_low) && finite(paired.ci95_high) ? `${delta(paired.ci95_low)} 至 ${delta(paired.ci95_high)}` : '未记录'], ['按机构分组的 95% 变化区间', finite(clustered.ci95_low) && finite(clustered.ci95_high) ? `${delta(clustered.ci95_low)} 至 ${delta(clustered.ci95_high)}` : '未记录']])); if (finite(paired.ci95_low) && paired.ci95_low <= 0 && paired.ci95_high >= 0) analysis.append(el('p', 'small muted', '逐案例变化区间包含 0：这批结果不足以确认稳定提升或退步；还需要多随机种子与更独立的样本。')); if (metrics.breakdown) { const breakdown = metrics.breakdown; if (breakdown.per_label && typeof breakdown.per_label === 'object') analysis.append(details('各类别的表现（原始分析记录）', breakdown.per_label)); if (breakdown.confusion) analysis.append(details('哪些标签被混淆（原始分析记录）', breakdown.confusion)); } stack.append(analysis); }
    const params = new URLSearchParams({run_id: state.run, task: state.task, outcome: state.outcome, offset: state.offset, limit: 10, query: state.query}), payload = await get(`${baseURL()}/predictions?${params}`, signal);
    stack.append(notice(payload.explanation || '分组使用同一条样本的训练前和训练后结果。原始回答与解析出的标签均保留，便于追查格式问题。'));
    const samples = section('逐例查看：模型哪里变好了，哪里退步了', '“正确”依据该任务存档的主指标判断；它不代表回答所有方面都正确。');
    samples.id = 'samplesSection';
    if (!array(payload.rows).length) samples.append(el('p', 'empty-inline', '当前筛选没有匹配样本。切换“样本变化”或清除搜索后再试。'));
    array(payload.rows).forEach(row => { const sample = el('article', 'sample-card'), header = el('div', 'sample-row-label'); header.append(el('strong', 'small', `样本 ${row.id}`), el('span', `outcome-label ${row.outcome}`, OUTCOMES[row.outcome] || (row.outcome === 'unpaired' ? '未配对（缺少对应基线）' : '未分类'))); sample.append(header); const baseline = el('div'), optimized = el('div'); [[baseline, row.baseline_output, row.baseline_prediction, row.baseline_correct], [optimized, row.output, row.prediction, row.correct]].forEach(([node, output, prediction, correct]) => { node.append(el('div', '', text(output, '未存档回答')), el('div', 'row-meta', `解析标签：${text(prediction)}`), el('div', `row-meta ${typeof correct === 'boolean' ? correct ? 'correct' : 'incorrect' : ''}`, typeof correct === 'boolean' ? correct ? '判定：正确' : '判定：错误' : '判定：未记录')); }); sample.append(table(['模型输入 / 原题', '标准答案', '训练前的输出', '训练后的输出'], [[text(row.input), text(row.target), baseline, optimized]], 'sample-table')); if (array(row.responses).length) sample.append(details('展开所有存档回答与格式检查', {responses: row.responses, format_valid: row.format_valid})); samples.append(sample); });
    samples.append(pagination(payload)); stack.append(samples); return stack;
  }
  async function reportsView() {
    const stack = el('div', 'stack'), files = section('报告和复现材料', '下载的是已存在的本地文件。缺失产物不会被替换成示例文件。'); files.append(artifacts(array(study.artifacts))); stack.append(files);
    const replay = section('命令记录', '实际执行命令、存档步骤和当前等价复现命令分别标注。请先核对模型、路径和依赖，再在命令行运行。'); replay.append(commands(array(study.commands))); stack.append(replay);
    const problems = section('问题记录与结论边界'); problems.append(issues(array(study.issues))); if (array(study.caveats).length) problems.append(el('h3', '', '阅读结果时需要知道'), list(study.caveats)); stack.append(problems);
    if (study.sources) { const sources = section('证据来源'); sources.append(details('展开原始记录位置', study.sources)); stack.append(sources); } return stack;
  }
  const renderers = {data: dataView, checks: checksView, experiments: experimentsView, training: trainingView, results: resultsView, reports: reportsView};
  async function render() {
    const serial = ++renderSerial; if (controller) controller.abort(); controller = new AbortController(); const signal = controller.signal;
    document.querySelectorAll('[data-view]').forEach(node => { if (node.dataset.view === state.view) node.setAttribute('aria-current', 'page'); else node.removeAttribute('aria-current'); });
    const [title, description] = VIEWS[state.view]; $('pageTitle').textContent = title; $('pageDescription').textContent = description; document.title = `${title} · SaddleLLM 实验工作台`; $('content').setAttribute('aria-busy', 'true'); $('content').replaceChildren(stateBox('正在读取记录', '图表、样本和命令均来自当前项目的实际产物。')); $('readStatus').textContent = '读取中'; $('readStatus').className = 'badge neutral';
    try {
      if (!state.study) { $('content').replaceChildren(stateBox('没有找到实验项目', '启动时指定包含实验 outputs/studies 的项目目录。此页面不会自动生成示例数据。', false, () => initialize())); $('readStatus').textContent = '没有项目'; return; }
      if (!study || study.id !== state.study) study = await get(baseURL(), signal);
      if (serial !== renderSerial) return;
      const runs = array(study.runs); if (!runs.some(run => run.id === state.run)) { state.run = runs[0]?.id || ''; saveURL(true); }
      const datasets = array(study.datasets); if (datasets.length && !datasets.some(item => item.id === state.task)) { state.task = datasets[0].id; saveURL(true); }
      $('studyMeta').textContent = [study.name || study.id, modelName()].filter(Boolean).join(' · '); $('studySelect').value = state.study;
      const caveat = study.notice || study.summary; $('studyNotice').hidden = !caveat; $('studyNotice').textContent = text(caveat, '');
      const result = await renderers[state.view](signal); if (serial !== renderSerial) return; $('content').replaceChildren(result); $('readStatus').textContent = '存档已读取'; $('readStatus').className = 'badge success';
    } catch (error) { if (error.name === 'AbortError' || serial !== renderSerial) return; $('content').replaceChildren(stateBox('记录读取失败', `${error.message}。请检查启动目录和产物文件是否仍然存在。`, true, () => { study = null; render(); })); $('readStatus').textContent = '读取失败'; $('readStatus').className = 'badge error'; }
    finally { if (serial === renderSerial) $('content').setAttribute('aria-busy', 'false'); }
  }
  async function initialize() {
    try {
      const payload = await get('/api/workbench/studies'); studies = array(payload.studies); const select = $('studySelect'); select.replaceChildren(); studies.forEach(item => { const option = el('option', '', item.name || item.id); option.value = item.id; select.append(option); }); select.disabled = !studies.length;
      if (!state.study) state.study = payload.initial_study || studies[0]?.id || ''; if (!studies.length) { const option = el('option', '', '没有找到实验项目'); option.value = ''; select.append(option); } select.value = state.study; saveURL(true); await render();
    } catch (error) { $('content').setAttribute('aria-busy', 'false'); $('content').replaceChildren(stateBox('项目列表读取失败', error.message, true, initialize)); $('readStatus').textContent = '连接失败'; $('readStatus').className = 'badge error'; }
  }
  document.querySelectorAll('[data-view]').forEach(node => node.addEventListener('click', () => update({view: node.dataset.view, offset: 0})));
  $('studySelect').addEventListener('change', event => update({study: event.target.value, query: '', offset: 0})); $('refresh').addEventListener('click', () => { study = null; render(); }); $('copyLink').addEventListener('click', () => copy(location.href));
  window.addEventListener('popstate', () => { state = readURL(); render(); }); initialize();
})();
