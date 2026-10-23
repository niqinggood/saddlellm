/* A guide pane inside the chat application. Switching panes preserves the chat DOM. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const pane = $('saddleIntro'), title = $('saddleGuideTitle'), featuresNode = $('saddleGuideFeatures');
  if (!pane || !title || !featuresNode) return; // Keep older chat-only template renderers usable.
  const conversation = $('conversation'), composer = document.querySelector('.composer-area'), originalTitle = document.title;
  const UI = {
    zh: {
      overline: '认识工具 · 从这里开始', language: '语言', languageAria: '介绍语言', backToChat: '返回模型问答', title: 'SaddleLLM 介绍', documentTitle: 'SaddleLLM 介绍 · 模型问答', description: '从数据准备、预训练与微调，到评测、轻量化和部署。按用途查看操作步骤、命令示例与疑难问题的排查方法。', workflowTitle: '一条模型工作流程', referenceLabel: '介绍与命令参考', workflowDescription: '先准备数据、检查环境，再按任务选择训练方法；用评测结果判断效果，最后按需要部署模型。', workflowNote: '这个页面用于理解功能和复制命令。点击介绍、搜索或复制命令不会启动训练。', workflowLoading: '正在读取流程说明…', workflowUnavailable: '流程说明暂不可用。', workflowEmpty: '没有读取到工作流程说明。', directoryAria: '介绍目录', searchLabel: '查找功能与问题', searchPlaceholder: '如：SFT、显存、评估', directoryTitle: '功能目录', clearFilters: '清除筛选', categoriesAria: '按用途筛选功能', directoryNote: '每项功能包含适用场景、命令和可展开的问题排查建议。', footer: 'SaddleLLM 功能介绍 · 示例中的路径和参数需要按实际数据与环境调整。', allFeatures: '全部功能', featureHeading: '功能说明', count: '显示 {shown} / {total} 项', useWhen: '什么时候用：', steps: '基本步骤', commands: '命令入口', copy: '复制命令', copyAria: '复制“{label}”命令', commandExample: '命令示例', commandUnavailable: '没有可复制的命令。', dataExample: '查看数据格式示例', copySuccess: '命令已复制；请在命令行中按实际路径运行。', copyFailure: '复制失败，请手动选择命令文字复制。', emptyTitle: '没有匹配的功能或问题', emptyDescription: '试试“SFT”“显存”或“数据”，也可以清除筛选查看全部功能。', loadingTitle: '正在读取功能说明', loadingDescription: '说明、命令和排查建议来自当前服务内置的 SaddleLLM 目录。', loadingCount: '正在加载', errorTitle: '功能目录读取失败', errorDescription: '{reason}。请确认当前服务已提供介绍目录后重试。', timeout: '服务超过 15 秒未响应', formatError: '目录格式或语言不完整', errorCount: '读取失败', retry: '重新读取', troubleshooting: '常见问题与排查', troubleshootingNote: '以下是可能原因和排查建议，不表示当前机器已经遇到或完成验证。', causes: '可能原因', checks: '如何检查', fixes: '如何修复', verification: '如何验证修复有效'
    },
    en: {
      overline: 'Get to know the tool · Start here', language: 'Language', languageAria: 'Introduction language', backToChat: 'Back to model chat', title: 'SaddleLLM introduction', documentTitle: 'SaddleLLM introduction · Model chat', description: 'Prepare data, pretrain or fine-tune models, evaluate results, and deploy. Explore practical steps, command examples, and troubleshooting for each workflow.', workflowTitle: 'A model workflow', referenceLabel: 'Guide and command reference', workflowDescription: 'Prepare data and check the environment, choose a training method for your task, evaluate the results, then deploy the model when needed.', workflowNote: 'This page explains features and lets you copy commands. Browsing, searching and copying do not start training.', workflowLoading: 'Loading the workflow…', workflowUnavailable: 'Workflow information is unavailable.', workflowEmpty: 'No workflow information was found.', directoryAria: 'Introduction directory', searchLabel: 'Find features and problems', searchPlaceholder: 'e.g. SFT, memory, evaluation', directoryTitle: 'Feature directory', clearFilters: 'Clear filters', categoriesAria: 'Filter features by purpose', directoryNote: 'Each feature includes use cases, commands and expandable troubleshooting advice.', footer: 'SaddleLLM introduction · Adapt example paths and parameters to your data and environment.', allFeatures: 'All features', featureHeading: 'Feature guide', count: 'Showing {shown} of {total} features', useWhen: 'When to use it: ', steps: 'Basic steps', commands: 'Command entry points', copy: 'Copy command', copyAria: 'Copy the {label} command', commandExample: 'Example command', commandUnavailable: 'No command is available to copy.', dataExample: 'View a data format example', copySuccess: 'Command copied. Run it in your terminal with your own paths.', copyFailure: 'Could not copy. Select the command text and copy it manually.', emptyTitle: 'No matching features or problems', emptyDescription: 'Try “SFT”, “memory” or “data”, or clear the filters to see all features.', loadingTitle: 'Loading the feature guide', loadingDescription: 'Descriptions, commands and troubleshooting advice come from the catalog built into this SaddleLLM service.', loadingCount: 'Loading', errorTitle: 'Could not load the catalog', errorDescription: '{reason}. Check that this service provides the introduction catalog, then retry.', timeout: 'The service did not respond within 15 seconds', formatError: 'The catalog format or language is incomplete', errorCount: 'Could not load', retry: 'Retry', troubleshooting: 'Troubleshooting', troubleshootingNote: 'These are possible causes and troubleshooting suggestions. They do not mean the current machine has encountered or verified them.', causes: 'Possible causes', checks: 'What to check', fixes: 'How to fix it', verification: 'How to verify the fix'
    }
  };
  function readLanguage() { return new URL(location.href).searchParams.get('lang') === 'en' ? 'en' : 'zh'; }
  let lang = readLanguage(), catalog = null, category = 'all', query = '', request = null, requestSerial = 0, toastTimer;
  function t(key, values = {}) { let value = UI[lang][key] || UI.zh[key] || key; Object.entries(values).forEach(([name, replacement]) => { value = value.replaceAll(`{${name}}`, String(replacement)); }); return value; }

  function node(tag, className, content) { const item = document.createElement(tag); if (className) item.className = className; if (content !== undefined && content !== null) item.textContent = String(content); return item; }
  function items(value) { return Array.isArray(value) ? value : []; }
  function content(value) { return value === undefined || value === null ? '' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value); }
  function paneURL(path) { return lang === 'en' ? `${path}?lang=en` : path; }
  function closeSidebar() { document.body.classList.remove('sidebar-open'); $('mobileBackdrop').hidden = true; $('mobileMenu').setAttribute('aria-expanded', 'false'); }
  function applyUI() {
    pane.lang = lang; pane.setAttribute('aria-label', t('title')); $('saddleGuideLanguage').value = lang; $('saddleGuideLanguage').setAttribute('aria-label', t('languageAria'));
    pane.querySelectorAll('[data-intro-i18n]').forEach(item => { item.textContent = t(item.dataset.introI18n); });
    pane.querySelectorAll('[data-intro-aria-i18n]').forEach(item => { item.setAttribute('aria-label', t(item.dataset.introAriaI18n)); });
    $('saddleGuideSearch').placeholder = t('searchPlaceholder'); $('saddleBackToChat').href = paneURL('/'); $('saddleIntroNav').href = paneURL('/saddle'); $('saddleChatNav').href = paneURL('/');
    title.textContent = catalog?.lang === lang ? catalog.title || t('title') : t('title'); $('saddleGuideDescription').textContent = t('description');
    if (!pane.hidden) document.title = t('documentTitle');
  }
  function setPane(intro, focus = false) {
    pane.hidden = !intro; conversation.hidden = intro; composer.hidden = intro;
    document.body.classList.toggle('saddle-intro-active', intro);
    [['saddleIntroNav', intro], ['saddleChatNav', !intro]].forEach(([id, active]) => { if (active) $(id).setAttribute('aria-current', 'page'); else $(id).removeAttribute('aria-current'); });
    const currentSession = document.querySelector('.session-item'); if (currentSession) { if (intro) currentSession.removeAttribute('aria-current'); else currentSession.setAttribute('aria-current', 'page'); }
    applyUI(); document.title = intro ? t('documentTitle') : originalTitle; closeSidebar();
    if (intro && (!catalog || catalog.lang !== lang) && !request) loadCatalog();
    if (focus) (intro ? title : $('message')).focus({preventScroll: true});
  }
  function navigate(path) { const url = paneURL(path); if (`${location.pathname}${location.search}` !== url) history.pushState({saddlePane: path, saddleLang: lang}, '', url); setPane(path === '/saddle', true); }
  function setLanguage(next, saveURL = false) {
    const changed = next !== lang; lang = next === 'en' ? 'en' : 'zh';
    if (saveURL) { const url = new URL(location.href); if (lang === 'en') url.searchParams.set('lang', 'en'); else url.searchParams.delete('lang'); history.pushState({saddlePane: location.pathname, saddleLang: lang}, '', url); }
    if (changed) { requestSerial += 1; if (request) request.abort(); request = null; catalog = null; }
    applyUI(); if (!pane.hidden && (changed || !catalog) && !request) loadCatalog();
  }
  function toast(message) { $('saddleGuideToast').textContent = message; $('saddleGuideToast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('saddleGuideToast').hidden = true; }, 2800); }
  async function copy(command) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(command);
      else { const input = node('textarea'); input.value = command; input.style.position = 'fixed'; input.style.opacity = '0'; document.body.append(input); input.select(); const copied = document.execCommand('copy'); input.remove(); if (!copied) throw new Error('copy'); }
      toast(t('copySuccess'));
    } catch (_) { toast(t('copyFailure')); }
  }
  function stateBox(heading, explanation, kind = 'empty', retry) { const box = node('div', `intro-state ${kind}`); box.setAttribute('role', 'status'); if (kind === 'loading') box.append(node('span', 'intro-spinner')); box.append(node('h3', '', heading), node('p', '', explanation)); if (retry) { const button = node('button', 'intro-button secondary', t('retry')); button.type = 'button'; button.addEventListener('click', retry); box.append(button); } return box; }
  function categoryName(id) { return items(catalog?.categories).find(item => item.id === id)?.title || id; }
  function renderWorkflow() { const list = $('saddleWorkflow'); list.replaceChildren(); items(catalog.workflow).forEach(step => { const item = node('li'); item.append(node('h3', 'intro-workflow-heading', step.title), node('p', '', step.description)); list.append(item); }); if (!list.children.length) list.append(node('li', 'intro-workflow-loading', t('workflowEmpty'))); }
  function renderCategories() {
    const list = $('saddleGuideCategories'); list.replaceChildren();
    [{id: 'all', title: t('allFeatures')}, ...items(catalog.categories)].forEach(item => { const button = node('button'); button.type = 'button'; button.dataset.category = item.id; button.setAttribute('aria-pressed', String(item.id === category)); button.append(node('span', '', item.title), node('span', 'intro-category-count', item.id === 'all' ? items(catalog.features).length : items(catalog.features).filter(feature => feature.category === item.id).length)); button.addEventListener('click', () => { category = item.id; renderCategories(); renderFeatures(); }); list.append(button); });
  }
  function troubleshooting(feature) {
    const cases = items(feature.troubleshooting); if (!cases.length) return null;
    const group = node('section', 'intro-troubleshooting'); group.append(node('h4', '', t('troubleshooting')), node('p', 'intro-troubleshooting-note', t('troubleshootingNote')));
    cases.forEach((problem, index) => { const detail = node('details', 'intro-problem'); detail.dataset.problem = `${feature.id}-${index}`; detail.append(node('summary', '', problem.symptom)); const body = node('div', 'intro-problem-body'); ['causes', 'checks', 'fixes', 'verification'].forEach(key => { if (!items(problem[key]).length) return; const section = node('section', 'intro-problem-section'); section.append(node('h5', 'intro-problem-label', t(key))); const list = node('ul', 'intro-problem-list'); items(problem[key]).forEach(value => list.append(node('li', '', content(value)))); section.append(list); body.append(section); }); detail.append(body); group.append(detail); }); return group;
  }
  function featureCard(feature) {
    const article = node('article', 'intro-feature'); article.id = `saddle-feature-${feature.id}`;
    const heading = node('div', 'intro-feature-heading'); heading.append(node('h3', '', feature.title), node('span', 'intro-feature-tag', categoryName(feature.category))); article.append(heading, node('p', 'intro-feature-summary', feature.summary));
    if (feature.use_when) { const when = node('p', 'intro-use-when'); when.append(node('strong', '', t('useWhen')), document.createTextNode(content(feature.use_when))); article.append(when); }
    if (items(feature.steps).length) { article.append(node('h4', '', t('steps'))); const steps = node('ol', 'intro-steps'); items(feature.steps).forEach(step => steps.append(node('li', '', content(step)))); article.append(steps); }
    if (items(feature.commands).length) { article.append(node('h4', '', t('commands'))); items(feature.commands).forEach(command => { const block = node('div', 'intro-command'), header = node('div', 'intro-command-heading'), button = node('button', 'intro-copy', t('copy')), value = content(command.command); button.type = 'button'; button.disabled = !value; button.setAttribute('aria-label', t('copyAria', {label: command.label || feature.title})); button.addEventListener('click', () => copy(value)); header.append(node('strong', '', command.label || t('commandExample')), button); block.append(header, node('pre', '', value || t('commandUnavailable'))); if (command.note) block.append(node('p', 'intro-command-note', command.note)); article.append(block); }); }
    if (feature.data_example && feature.data_example.content) { const example = node('details', 'intro-example'); example.append(node('summary', '', feature.data_example.label || t('dataExample')), node('pre', '', content(feature.data_example.content))); article.append(example); }
    if (items(feature.notes).length) { const notes = node('ul', 'intro-notes'); items(feature.notes).forEach(note => notes.append(node('li', '', content(note)))); article.append(notes); }
    const problems = troubleshooting(feature); if (problems) article.append(problems); return article;
  }
  function searchable(feature) { return [feature.title, feature.summary, feature.use_when, categoryName(feature.category), ...items(feature.steps), ...items(feature.notes), ...items(feature.commands).flatMap(command => [command.label, command.command, command.note]), feature.data_example?.label, feature.data_example?.content, ...items(feature.troubleshooting).flatMap(problem => [problem.symptom, ...items(problem.causes), ...items(problem.checks), ...items(problem.fixes), ...items(problem.verification)])].map(content).join(' ').toLocaleLowerCase(); }
  function renderFeatures() {
    if (!catalog || catalog.lang !== lang) return;
    const lower = query.toLocaleLowerCase(), matches = items(catalog.features).filter(feature => (category === 'all' || feature.category === category) && (!lower || searchable(feature).includes(lower)));
    $('saddleGuideSectionTitle').textContent = category === 'all' ? t('featureHeading') : categoryName(category);
    $('saddleGuideCount').textContent = t('count', {shown: matches.length, total: items(catalog.features).length});
    $('saddleGuideClear').hidden = category === 'all' && !query;
    if (!matches.length) { featuresNode.replaceChildren(stateBox(t('emptyTitle'), t('emptyDescription'))); }
    else { const list = node('div', 'intro-feature-list'); matches.forEach(feature => list.append(featureCard(feature))); featuresNode.replaceChildren(list); }
    featuresNode.setAttribute('aria-busy', 'false');
  }
  async function loadCatalog() {
    if (request) request.abort(); const serial = ++requestSerial, expectedLang = lang, controller = new AbortController(); request = controller; let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 15000);
    $('saddleGuideSearch').disabled = true; $('saddleGuideCount').textContent = t('loadingCount'); $('saddleGuideSectionTitle').textContent = t('featureHeading'); $('saddleGuideCategories').replaceChildren(); $('saddleWorkflow').replaceChildren(node('li', 'intro-workflow-loading', t('workflowLoading'))); featuresNode.setAttribute('aria-busy', 'true'); featuresNode.replaceChildren(stateBox(t('loadingTitle'), t('loadingDescription'), 'loading'));
    try {
      const response = await fetch(`/api/saddle/catalog?lang=${expectedLang}`, {signal: controller.signal, cache: 'no-store'}); if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json(); if (serial !== requestSerial || expectedLang !== lang) return;
      if (!Array.isArray(payload.features) || !Array.isArray(payload.categories) || payload.lang !== expectedLang) throw new Error(t('formatError')); catalog = payload;
      if (category !== 'all' && !items(payload.categories).some(item => item.id === category)) category = 'all';
      applyUI(); $('saddleGuideSearch').disabled = false; renderWorkflow(); renderCategories(); renderFeatures();
    } catch (error) { if (serial !== requestSerial || expectedLang !== lang || (error.name === 'AbortError' && !timedOut)) return; featuresNode.replaceChildren(stateBox(t('errorTitle'), t('errorDescription', {reason: timedOut ? t('timeout') : error.message}), 'error', loadCatalog)); $('saddleGuideCount').textContent = t('errorCount'); featuresNode.setAttribute('aria-busy', 'false'); $('saddleWorkflow').replaceChildren(node('li', 'intro-workflow-loading', t('workflowUnavailable'))); }
    finally { clearTimeout(timer); if (serial === requestSerial) request = null; }
  }
  document.addEventListener('click', event => {
    const link = event.target.closest('a#saddleChatNav, a#saddleIntroNav, a#saddleBackToChat');
    if (link && event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) { event.preventDefault(); navigate(link.pathname); }
  });
  $('saddleGuideLanguage').addEventListener('change', event => setLanguage(event.target.value, true));
  $('saddleGuideSearch').addEventListener('input', event => { query = event.target.value.trim(); renderFeatures(); });
  $('saddleGuideClear').addEventListener('click', () => { category = 'all'; query = ''; $('saddleGuideSearch').value = ''; if (catalog) { renderCategories(); renderFeatures(); } $('saddleGuideSearch').focus(); });
  $('newChat').addEventListener('click', () => { if (!pane.hidden) navigate('/'); });
  $('historyList').addEventListener('click', event => { if (event.target.closest('.history-open') && !pane.hidden) navigate('/'); });
  window.addEventListener('popstate', () => { setLanguage(readLanguage()); setPane(location.pathname === '/saddle'); });
  applyUI(); setPane(location.pathname === '/saddle');
})();
