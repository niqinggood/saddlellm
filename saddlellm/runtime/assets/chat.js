(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const messageInput = $('message');
  const messagesNode = $('messages');
  const conversationNode = $('conversation');
  const maxTokensInput = $('maxTokens');
  const temperatureInput = $('temperature');
  const apiKeyInput = $('apiKey');
  const historyListNode = $('historyList');
  const configuredMax = Number(document.body.dataset.maxTokens) || 512;
  const historyEnabled = document.body.dataset.historyEnabled !== 'false';

  let modelName = '模型';
  let ready = false;
  let requiresAuth = false;
  let history = [];
  let sessionSummaries = [];
  let activeSessionId = null;
  let activeController = null;
  let conversationEpoch = 0;
  let sessionListEpoch = 0;
  let loadingSession = false;
  let settingsSnapshot = null;
  let modelIdentityKnown = false;
  let restoreAttempted = false;
  let restoreBlocked = false;
  let restoringSession = false;
  let clearPointerOnModelReady = false;
  const sessionIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

  function sessionPointerKey() {
    return modelIdentityKnown ? `saddlellm.activeChatSession.v1:${encodeURIComponent(modelName)}` : null;
  }

  function rememberedSessionId() {
    if (!historyEnabled || !modelIdentityKnown) return null;
    try {
      const key = sessionPointerKey(), id = sessionStorage.getItem(key);
      if (id && !sessionIdPattern.test(id)) { sessionStorage.removeItem(key); return null; }
      return id;
    } catch (_) { return null; } // Storage is optional in private/restricted browser modes.
  }

  function rememberSession(id) {
    if (!historyEnabled || !modelIdentityKnown || !sessionIdPattern.test(id)) return;
    try { sessionStorage.setItem(sessionPointerKey(), id); } catch (_) { /* Chat still works without tab storage. */ }
  }

  function clearRememberedSession(id = null) {
    if (!modelIdentityKnown) return;
    try {
      const key = sessionPointerKey();
      if (id === null || sessionStorage.getItem(key) === id) sessionStorage.removeItem(key);
    } catch (_) { /* Chat still works without tab storage. */ }
  }

  function blockAutomaticRestore() {
    restoreBlocked = true;
    if (restoringSession) {
      conversationEpoch += 1; restoringSession = false; loadingSession = false;
      historyHint('已保留当前输入；可从历史记录手动打开对话');
      updateSendState();
    }
  }

  async function responseError(response) {
    let detail = '';
    try {
      const raw = await response.text();
      try { const data = JSON.parse(raw); const value = data.detail ?? data.error?.message; detail = value == null ? raw : typeof value === 'string' ? value : JSON.stringify(value); }
      catch (_) { detail = raw; }
    } catch (_) { /* The HTTP status remains useful when the response body is unavailable. */ }
    const error = new Error(`HTTP ${response.status}${detail ? ` · ${detail.replace(/\s+/g, ' ').slice(0, 180)}` : ''}`);
    error.status = response.status;
    return error;
  }

  async function restoreRememberedSession() {
    if (!historyEnabled || !ready || !modelIdentityKnown || restoreAttempted || restoreBlocked || loadingSession || activeController || activeSessionId || history.length || messageInput.value.trim() || (requiresAuth && !apiKeyInput.value.trim())) return;
    restoreAttempted = true;
    const id = rememberedSessionId();
    if (!id) return; // Never guess the newest session when this tab has no pointer.
    if (!sessionSummaries.some(session => session.id === id && session.model === modelName)) { clearRememberedSession(id); return; }
    historyHint('正在恢复本标签页上次使用的对话…');
    await openSession(id, {automatic: true});
  }

  function requestHeaders(json = false) {
    const headers = json ? {'Content-Type': 'application/json'} : {};
    const key = apiKeyInput.value.trim();
    if (key) headers.Authorization = `Bearer ${key}`;
    return headers;
  }

  function newSessionId() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  function historyHint(message, error = false) {
    $('sessionHint').textContent = message;
    $('sessionHint').classList.toggle('history-error', error);
  }

  function renderSessionList() {
    historyListNode.replaceChildren();
    if (!sessionSummaries.length) {
      const empty = document.createElement('p');
      empty.className = 'history-empty';
      empty.textContent = historyEnabled ? '暂无历史对话' : '历史保存已关闭';
      historyListNode.append(empty);
      return;
    }
    for (const session of sessionSummaries) {
      const row = document.createElement('div');
      row.className = 'history-row';
      if (session.id === activeSessionId) row.classList.add('active');
      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'history-open';
      open.textContent = session.title;
      open.title = session.title;
      open.addEventListener('click', () => openSession(session.id));
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'history-delete';
      remove.textContent = '×';
      remove.title = `删除 ${session.title}`;
      remove.setAttribute('aria-label', remove.title);
      remove.addEventListener('click', () => deleteSession(session.id));
      row.append(open, remove);
      historyListNode.append(row);
    }
  }

  async function refreshSessions() {
    if (!historyEnabled) {
      sessionSummaries = [];
      renderSessionList();
      historyHint('历史保存已关闭；刷新页面将清空当前对话');
      return false;
    }
    if (!ready || (requiresAuth && !apiKeyInput.value.trim())) {
      if (requiresAuth && !apiKeyInput.value.trim()) historyHint('输入 API Key 后查看历史', true);
      return false;
    }
    const ticket = ++sessionListEpoch;
    try {
      const response = await fetch('/api/chat/sessions', {
        headers: requestHeaders(), cache: 'no-store',
      });
      if (ticket !== sessionListEpoch) return false;
      if (!response.ok) {
        sessionSummaries = [];
        renderSessionList();
        if (response.status === 401) historyHint('输入 API Key 后查看历史', true);
        else if (response.status === 403) historyHint('远程访问历史需要 API Key', true);
        else historyHint(`读取历史失败：HTTP ${response.status}`, true);
        return false;
      }
      const data = await response.json();
      if (ticket !== sessionListEpoch) return false;
      if (!Array.isArray(data.sessions)) throw new Error('历史列表格式无效');
      sessionSummaries = data.sessions;
      renderSessionList();
      historyHint('保存在启动目录的 .saddlellm 文件夹（共享工作区）');
      await restoreRememberedSession();
      return true;
    } catch (_) {
      if (ticket === sessionListEpoch) historyHint('暂时无法读取历史记录', true);
      return false;
    }
  }

  async function persistSession(messages, epoch) {
    if (!historyEnabled) return;
    const id = activeSessionId || newSessionId();
    try {
      const response = await fetch(`/api/chat/sessions/${id}`, {
        method: 'PUT', headers: requestHeaders(true),
        body: JSON.stringify({messages}),
      });
      if (!response.ok) throw await responseError(response);
      const saved = await response.json();
      if (epoch === conversationEpoch) {
        activeSessionId = id;
        $('sessionTitle').textContent = saved.title;
        rememberSession(id);
        await refreshSessions();
      }
    } catch (error) {
      if (epoch === conversationEpoch) {
        const reason = {401: '鉴权失败，请检查 API Key', 403: '没有访问历史记录的权限', 400: '消息或会话参数不符合要求'}[error.status] || '保存请求失败';
        historyHint(`本次对话未保存：${reason} · ${error.message}`, true);
      }
    }
  }

  async function openSession(id, {automatic = false} = {}) {
    if (!historyEnabled) return false;
    if (automatic && (restoreBlocked || messageInput.value.trim() || history.length || activeSessionId || activeController)) return false;
    if (!automatic) blockAutomaticRestore();
    if (activeController) activeController.abort();
    const epoch = ++conversationEpoch;
    activeController = null;
    loadingSession = true;
    restoringSession = automatic;
    updateSendState();
    try {
      const response = await fetch(`/api/chat/sessions/${id}`, {
        headers: requestHeaders(), cache: 'no-store',
      });
      if (!response.ok) throw await responseError(response);
      const session = await response.json();
      if (epoch !== conversationEpoch) return false;
      if (automatic && (restoreBlocked || messageInput.value.trim())) { blockAutomaticRestore(); return false; }
      if (session.model !== modelName || !Array.isArray(session.messages)) throw new Error('会话模型或消息格式不匹配');
      activeSessionId = id;
      history = session.messages.map(({role, content}) => ({role, content}));
      messagesNode.replaceChildren();
      $('welcome').hidden = history.length > 0;
      for (const message of history) addMessage(message.role, message.content);
      $('sessionTitle').textContent = session.title;
      rememberSession(id);
      renderSessionList();
      if (automatic) historyHint('已恢复本标签页的对话；保存在启动目录的 .saddlellm 文件夹');
      closeMobileSidebar();
      scrollToLatest();
      return true;
    } catch (error) {
      if (epoch === conversationEpoch) {
        if (automatic && error.status === 404) clearRememberedSession(id);
        historyHint(`打开历史对话失败：${error.message}`, true);
      }
      return false;
    } finally {
      if (epoch === conversationEpoch) {
        loadingSession = false;
        restoringSession = false;
        updateSendState();
        messageInput.focus();
      }
    }
  }

  async function deleteSession(id) {
    if (!historyEnabled) return;
    blockAutomaticRestore();
    if (!window.confirm('删除这条历史对话？此操作无法撤销。')) return;
    try {
      const response = await fetch(`/api/chat/sessions/${id}`, {
        method: 'DELETE', headers: requestHeaders(),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      clearRememberedSession(id);
      if (id === activeSessionId) resetConversation();
      await refreshSessions();
    } catch (_) {
      historyHint('删除历史对话失败', true);
    }
  }

  function setStatus(label, state) {
    $('health').textContent = label;
    $('sidebarStatus').textContent = label;
    $('topStatus').className = `status-${state}`;
    $('sidebarStatus').parentElement.parentElement.className = `sidebar-status status-${state}`;
  }

  async function refreshHealth() {
    try {
      const response = await fetch('/health', {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      modelName = String(data.model || '模型');
      modelIdentityKnown = true;
      if (clearPointerOnModelReady) { clearRememberedSession(); clearPointerOnModelReady = false; }
      ready = data.status === 'ok' && data.ready !== false;
      requiresAuth = Boolean(data.requires_auth);
      $('modelName').textContent = modelName;
      $('sidebarModel').textContent = modelName;
      $('welcomeModel').textContent = modelName;
      $('settingsModel').value = modelName;
      messageInput.placeholder = '发送消息';
      if (!ready) setStatus('模型未就绪', 'warning');
      else if (requiresAuth && !apiKeyInput.value.trim()) setStatus('需要 API Key', 'warning');
      else setStatus('已连接', 'ready');
    } catch (_) {
      ready = false;
      setStatus('连接失败', 'error');
    }
    updateSendState();
    if (ready) await refreshSessions();
  }

  function updateSendState() {
    conversationNode.setAttribute('aria-busy', String(loadingSession || Boolean(activeController)));
    $('send').disabled = !ready || loadingSession || Boolean(activeController) || !messageInput.value.trim();
    $('send').hidden = Boolean(activeController);
    $('stop').hidden = !activeController;
  }

  function resizeInput() {
    messageInput.style.height = 'auto';
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 210)}px`;
  }

  function scrollToLatest() {
    conversationNode.scrollTop = conversationNode.scrollHeight;
  }

  const safeTags = new Set(['P', 'BR', 'STRONG', 'EM', 'DEL', 'H1', 'H2', 'H3', 'H4',
    'UL', 'OL', 'LI', 'PRE', 'CODE', 'BLOCKQUOTE', 'A', 'HR', 'TABLE', 'THEAD',
    'TBODY', 'TR', 'TH', 'TD']);
  const blockedTags = new Set(['SCRIPT', 'STYLE', 'IFRAME', 'OBJECT', 'EMBED', 'SVG',
    'MATH', 'FORM', 'INPUT', 'BUTTON', 'VIDEO', 'AUDIO', 'LINK', 'META']);

  function safeNode(node) {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.textContent || '');
    if (node.nodeType !== Node.ELEMENT_NODE || blockedTags.has(node.tagName)) return null;
    if (node.tagName === 'IMG') return document.createTextNode(node.getAttribute('alt') || '[图片]');

    const tag = node.tagName;
    const element = safeTags.has(tag) ? document.createElement(tag.toLowerCase()) : document.createDocumentFragment();
    if (tag === 'A' && element.nodeType === Node.ELEMENT_NODE) {
      const href = node.getAttribute('href') || '';
      try {
        const url = new URL(href, location.href);
        if (['http:', 'https:', 'mailto:'].includes(url.protocol)) {
          element.href = url.href;
          element.target = '_blank';
          element.rel = 'noopener noreferrer';
        }
      } catch (_) { /* Keep link text without an unsafe destination. */ }
    }
    if (tag === 'OL') {
      const start = Number(node.getAttribute('start'));
      if (Number.isInteger(start) && start > 0 && start < 10000) element.start = start;
    }
    for (const child of node.childNodes) {
      const safe = safeNode(child);
      if (safe) element.append(safe);
    }
    if (tag === 'TABLE') {
      const wrapper = document.createElement('div');
      wrapper.className = 'markdown-table';
      wrapper.append(element);
      return wrapper;
    }
    if (tag === 'PRE') {
      const wrapper = document.createElement('div');
      wrapper.className = 'markdown-code';
      const toolbar = document.createElement('div');
      toolbar.className = 'markdown-code-toolbar';
      const language = document.createElement('span');
      const code = node.querySelector(':scope > code');
      const match = (code?.getAttribute('class') || '').match(/(?:^|\s)language-([\w+#.-]+)/);
      language.textContent = match ? match[1] : '代码';
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'markdown-code-copy';
      copy.textContent = '复制代码';
      copy.setAttribute('aria-label', '复制代码块');
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(element.textContent || '');
          copy.textContent = '已复制';
          window.setTimeout(() => { copy.textContent = '复制代码'; }, 1800);
        } catch (_) { copy.textContent = '复制失败'; }
      });
      toolbar.append(language, copy);
      wrapper.append(toolbar, element);
      return wrapper;
    }
    return element;
  }

  function extractMath(source) {
    const formulas = [];
    const mark = (latex, display) => {
      const id = formulas.length;
      formulas.push({latex, display});
      return `\uE000${id}\uE001`;
    };
    // Fenced and inline code are deliberately excluded from formula extraction.
    const code = /(```[\s\S]*?(?:```|$)|`[^`\n]*(?:`|$))/g;
    const parts = source.split(code);
    const text = parts.map((part, index) => {
      if (index % 2) return part;
      return part
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, latex) => mark(latex, true))
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, latex) => mark(latex, true))
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, latex) => mark(latex, false))
        .replace(/(^|[^\\])\$([^\n$]+?)\$/g, (_, prefix, latex) => prefix + mark(latex, false));
    }).join('');
    return {text, formulas};
  }

  function restoreMath(fragment, formulas) {
    if (!formulas.length) return;
    const walker = document.createTreeWalker(fragment, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      if (/\uE000\d+\uE001/.test(walker.currentNode.textContent || '')) nodes.push(walker.currentNode);
    }
    for (const node of nodes) {
      const value = node.textContent || '';
      const replacement = document.createDocumentFragment();
      let cursor = 0;
      for (const match of value.matchAll(/\uE000(\d+)\uE001/g)) {
        replacement.append(document.createTextNode(value.slice(cursor, match.index)));
        const formula = formulas[Number(match[1])];
        if (formula) {
          const holder = document.createElement('span');
          holder.className = formula.display ? 'math-display' : 'math-inline';
          if (window.katex?.render) {
            try {
              window.katex.render(formula.latex, holder, {displayMode: formula.display, throwOnError: false, trust: false, strict: 'ignore'});
            } catch (_) { holder.textContent = formula.latex; }
          } else {
            holder.textContent = formula.display ? `$$${formula.latex}$$` : `$${formula.latex}$`;
          }
          replacement.append(holder);
        }
        cursor = match.index + match[0].length;
      }
      replacement.append(document.createTextNode(value.slice(cursor)));
      node.replaceWith(replacement);
    }
  }

  function renderMarkdown(value) {
    const fragment = document.createDocumentFragment();
    if (!window.marked?.parse) {
      const paragraph = document.createElement('p');
      paragraph.textContent = value;
      fragment.append(paragraph);
      return fragment;
    }
    const extracted = extractMath(value);
    try {
      const parsed = new DOMParser().parseFromString(window.marked.parse(extracted.text, {gfm: true, breaks: true}), 'text/html');
      for (const child of parsed.body.childNodes) {
        const safe = safeNode(child);
        if (safe) fragment.append(safe);
      }
      restoreMath(fragment, extracted.formulas);
    } catch (_) {
      const paragraph = document.createElement('p');
      paragraph.textContent = value;
      fragment.replaceChildren(paragraph);
    }
    return fragment;
  }

  function addMessage(role, content) {
    $('welcome').hidden = true;
    const row = document.createElement('article');
    row.className = `message-row ${role}`;
    if (role === 'assistant') row.dataset.plainText = content;
    if (role === 'assistant') {
      const header = document.createElement('div');
      header.className = 'message-header';
      const name = document.createElement('strong');
      name.textContent = modelName;
      header.append(name);
      row.append(header);
    }
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    if (role === 'assistant') bubble.replaceChildren(renderMarkdown(content));
    else bubble.textContent = content;
    row.append(bubble);
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    const time = document.createElement('time');
    time.textContent = new Date().toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'});
    meta.append(time);
    if (role === 'assistant') {
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.className = 'copy-button';
      copy.textContent = '复制';
      copy.setAttribute('aria-label', '复制回答');
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(row.dataset.plainText || '');
          copy.textContent = '已复制';
          window.setTimeout(() => { copy.textContent = '复制'; }, 1800);
        } catch (_) { copy.textContent = '复制失败'; }
      });
      meta.append(copy);
    }
    row.append(meta);
    messagesNode.append(row);
    scrollToLatest();
    return {row, bubble};
  }

  function closeMobileSidebar() {
    document.body.classList.remove('sidebar-open');
    $('mobileBackdrop').hidden = true;
    $('mobileMenu').setAttribute('aria-expanded', 'false');
  }

  function setSidebarCollapsed(collapsed) {
    document.querySelector('.app-shell').classList.toggle('sidebar-collapsed', collapsed);
    $('sidebarToggle').setAttribute('aria-expanded', String(!collapsed));
    $('sidebarToggle').setAttribute('aria-label', collapsed ? '展开侧栏' : '收起侧栏');
    $('sidebarToggle').title = collapsed ? '展开侧栏' : '收起侧栏';
    try { localStorage.setItem('saddlellm.sidebarCollapsed', collapsed ? '1' : '0'); } catch (_) { /* Private mode. */ }
  }

  function resetConversation() {
    blockAutomaticRestore();
    if (modelIdentityKnown) clearRememberedSession(); else clearPointerOnModelReady = true;
    conversationEpoch += 1;
    if (activeController) activeController.abort();
    activeController = null;
    activeSessionId = null;
    loadingSession = false;
    restoringSession = false;
    history = [];
    messagesNode.replaceChildren();
    $('welcome').hidden = false;
    $('sessionTitle').textContent = '新对话';
    renderSessionList();
    messageInput.value = '';
    resizeInput();
    updateSendState();
    closeMobileSidebar();
    messageInput.focus();
  }

  function openSettings() {
    settingsSnapshot = {
      temperature: temperatureInput.value,
      maxTokens: maxTokensInput.value,
      apiKey: apiKeyInput.value,
    };
    $('settingsPanel').hidden = false;
    $('settingsBackdrop').hidden = false;
    $('topSettings').setAttribute('aria-expanded', 'true');
    closeMobileSidebar();
    temperatureInput.focus();
  }

  function closeSettings(save) {
    if (!save && settingsSnapshot) {
      temperatureInput.value = settingsSnapshot.temperature;
      maxTokensInput.value = settingsSnapshot.maxTokens;
      apiKeyInput.value = settingsSnapshot.apiKey;
    }
    if (save) {
      const requested = Number(maxTokensInput.value);
      maxTokensInput.value = String(Number.isFinite(requested) ? Math.min(configuredMax, Math.max(1, Math.floor(requested))) : Number(document.body.dataset.defaultMaxTokens) || 256);
    }
    $('temperatureValue').textContent = temperatureInput.value;
    $('settingsPanel').hidden = true;
    $('settingsBackdrop').hidden = true;
    $('topSettings').setAttribute('aria-expanded', 'false');
    settingsSnapshot = null;
    if (ready) setStatus(requiresAuth && !apiKeyInput.value.trim() ? '需要 API Key' : '已连接', requiresAuth && !apiKeyInput.value.trim() ? 'warning' : 'ready');
    if (save) refreshSessions();
    messageInput.focus();
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = messageInput.value.trim();
    if (!text || !ready || loadingSession || activeController) return;
    if (requiresAuth && !apiKeyInput.value.trim()) {
      openSettings();
      apiKeyInput.focus();
      return;
    }
    blockAutomaticRestore();
    const epoch = conversationEpoch;
    messageInput.value = '';
    resizeInput();
    if (!history.length) $('sessionTitle').textContent = text;
    history.push({role: 'user', content: text});
    addMessage('user', text);
    const answer = addMessage('assistant', '');
    answer.bubble.classList.add('typing-cursor');
    const controller = new AbortController();
    activeController = controller;
    updateSendState();
    let full = '';
    let shouldSave = false;
    let paintFrame = 0;
    const paintAnswer = () => {
      paintFrame = 0;
      if (epoch !== conversationEpoch) return;
      answer.bubble.replaceChildren(renderMarkdown(full));
      answer.row.dataset.plainText = full;
      scrollToLatest();
    };
    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST', headers: requestHeaders(true), signal: controller.signal,
        body: JSON.stringify({messages: history, temperature: Number(temperatureInput.value), max_tokens: Number(maxTokensInput.value), stream: true}),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`HTTP ${response.status}${detail ? ` · ${detail.slice(0, 180)}` : ''}`);
      }
      if (!response.body) throw new Error('浏览器不支持流式响应');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';
      let streamDone = false;
      const consumeLine = (line) => {
        if (!line.startsWith('data: ')) return;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') { streamDone = true; return; }
        const data = JSON.parse(raw);
        if (data.error) throw new Error(data.error.message || '模型生成失败');
        const piece = data.choices?.[0]?.delta?.content || '';
        if (piece) {
          full += piece;
          if (!paintFrame) paintFrame = window.requestAnimationFrame(paintAnswer);
        }
      };
      while (!streamDone) {
        const {value, done} = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), {stream: !done});
        const lines = pending.split('\n');
        pending = lines.pop() || '';
        for (const line of lines) consumeLine(line.trimEnd());
        if (done) { if (pending) consumeLine(pending.trimEnd()); break; }
      }
      if (epoch === conversationEpoch && full) {
        history.push({role: 'assistant', content: full});
        shouldSave = true;
      }
    } catch (error) {
      if (epoch !== conversationEpoch) return;
      if (error.name === 'AbortError') {
        if (full) {
          history.push({role: 'assistant', content: full});
          shouldSave = true;
        }
        else answer.bubble.textContent = '已停止生成';
      } else {
        answer.row.classList.add('message-error');
        answer.bubble.textContent = `请求失败：${error.message}`;
      }
    } finally {
      if (paintFrame) window.cancelAnimationFrame(paintFrame);
      if (epoch === conversationEpoch) {
        if (full && !answer.row.classList.contains('message-error')) paintAnswer();
        answer.bubble.classList.remove('typing-cursor');
        if (shouldSave) {
          const snapshot = history.map(({role, content}) => ({role, content}));
          await persistSession(snapshot, epoch);
        }
      }
      if (epoch === conversationEpoch) {
        if (activeController === controller) activeController = null;
        updateSendState();
        messageInput.focus();
      }
    }
  }

  $('form').addEventListener('submit', sendMessage);
  messageInput.addEventListener('input', () => { blockAutomaticRestore(); resizeInput(); updateSendState(); });
  messageInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('form').requestSubmit();
    }
  });
  $('stop').addEventListener('click', () => activeController?.abort());
  $('newChat').addEventListener('click', resetConversation);
  $('sidebarToggle').addEventListener('click', () => {
    if (window.matchMedia('(max-width: 800px)').matches) closeMobileSidebar();
    else setSidebarCollapsed(!document.querySelector('.app-shell').classList.contains('sidebar-collapsed'));
  });
  $('topSettings').addEventListener('click', openSettings);
  $('sidebarSettings').addEventListener('click', openSettings);
  $('closeSettings').addEventListener('click', () => closeSettings(false));
  $('cancelSettings').addEventListener('click', () => closeSettings(false));
  $('saveSettings').addEventListener('click', () => closeSettings(true));
  $('settingsBackdrop').addEventListener('click', () => closeSettings(false));
  temperatureInput.addEventListener('input', () => { $('temperatureValue').textContent = temperatureInput.value; });
  $('mobileMenu').addEventListener('click', () => {
    const opening = !document.body.classList.contains('sidebar-open');
    if (opening) setSidebarCollapsed(false);
    document.body.classList.toggle('sidebar-open', opening);
    $('mobileBackdrop').hidden = !opening;
    $('mobileMenu').setAttribute('aria-expanded', String(opening));
  });
  $('mobileBackdrop').addEventListener('click', closeMobileSidebar);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (!$('settingsPanel').hidden) closeSettings(false);
      else closeMobileSidebar();
    }
  });

  try {
    if (localStorage.getItem('saddlellm.sidebarCollapsed') === '1' && !window.matchMedia('(max-width: 800px)').matches) setSidebarCollapsed(true);
  } catch (_) { /* Private mode. */ }
  refreshHealth();
  messageInput.focus();
})();
