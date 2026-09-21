"""
Floating Chat Widget — AI Data Analyst Platform 11.0
=====================================================
Generates a self-contained HTML/CSS/JS floating chat bubble
injected via st.components.v1.html().

The widget:
  - Renders a persistent 🤖 bubble at bottom-right
  - Opens a compact drawer (380×520px) on click
  - Communicates with Streamlit via window.parent.postMessage()
  - Streamlit receives messages via st.session_state and component_value

No external dependencies — pure vanilla HTML/CSS/JS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_floating_widget_html(
    messages: List[Dict[str, str]],
    suggested_questions: List[str],
    dataset_name: str,
    is_ai_active: bool = False,
    height: int = 640,
) -> str:
    """
    Build the complete floating chat widget HTML.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        suggested_questions: Pre-built suggested questions list
        dataset_name: Name of the active dataset
        is_ai_active: Whether Gemini API key is configured
        height: Component iframe height in pixels

    Returns:
        Full HTML string to pass to st.components.v1.html()
    """
    # Serialize messages and questions for injection into JS
    import json as _json
    messages_json = _json.dumps(messages, ensure_ascii=False)
    questions_json = _json.dumps(suggested_questions[:6], ensure_ascii=False)
    status_color = "#22c55e" if is_ai_active else "#f59e0b"
    status_text = "Gemini AI Active" if is_ai_active else "Keyword Mode"
    dataset_label = dataset_name[:28] + "…" if len(dataset_name) > 28 else dataset_name

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}

  /* ── Bubble Button ── */
  #chat-bubble {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #38bdf8 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 8px 32px rgba(99,102,241,0.5);
    z-index: 9999;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: none;
    color: white;
    font-size: 1.5rem;
  }}
  #chat-bubble:hover {{
    transform: scale(1.08);
    box-shadow: 0 12px 40px rgba(99,102,241,0.65);
  }}
  #chat-bubble.open {{ background: linear-gradient(135deg, #ef4444 0%, #f97316 100%); }}

  /* Notification dot */
  #notif-dot {{
    position: absolute;
    top: 2px; right: 2px;
    width: 13px; height: 13px;
    border-radius: 50%;
    background: #ef4444;
    border: 2px solid #0f172a;
    display: none;
  }}

  /* ── Drawer ── */
  #chat-drawer {{
    position: fixed;
    bottom: 98px;
    right: 28px;
    width: 380px;
    height: 520px;
    background: #1e293b;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.6);
    display: none;
    flex-direction: column;
    z-index: 9998;
    overflow: hidden;
    animation: slideUp 0.22s ease;
  }}
  #chat-drawer.visible {{ display: flex; }}
  @keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  /* ── Drawer Header ── */
  .drawer-header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }}
  .drawer-avatar {{
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #38bdf8);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
  }}
  .drawer-title {{ flex: 1; }}
  .drawer-title h3 {{ color: #f8fafc; font-size: 0.9rem; font-weight: 700; }}
  .drawer-title p {{ color: #64748b; font-size: 0.72rem; margin-top: 1px; }}
  .status-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: {status_color};
    flex-shrink: 0;
    box-shadow: 0 0 6px {status_color};
  }}
  .close-btn {{
    background: none; border: none; color: #64748b;
    font-size: 1.1rem; cursor: pointer; padding: 4px;
    border-radius: 6px; transition: color 0.15s;
  }}
  .close-btn:hover {{ color: #f8fafc; }}

  /* ── Suggestions ── */
  .suggestions-strip {{
    padding: 10px 12px;
    display: flex;
    gap: 6px;
    overflow-x: auto;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    scrollbar-width: none;
  }}
  .suggestions-strip::-webkit-scrollbar {{ display: none; }}
  .suggest-chip {{
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.7rem;
    white-space: nowrap;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    flex-shrink: 0;
  }}
  .suggest-chip:hover {{
    background: rgba(99,102,241,0.35);
    color: #fff;
  }}

  /* ── Messages ── */
  .messages-area {{
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
  }}
  .empty-state {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%; color: #475569; text-align: center; gap: 8px;
  }}
  .empty-state .icon {{ font-size: 2.5rem; }}
  .empty-state p {{ font-size: 0.8rem; line-height: 1.5; max-width: 240px; }}
  .msg-row {{ display: flex; gap: 8px; max-width: 100%; }}
  .msg-row.user {{ flex-direction: row-reverse; }}
  .msg-avatar {{
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0; margin-top: 2px;
  }}
  .msg-avatar.bot {{ background: linear-gradient(135deg, #6366f1, #38bdf8); }}
  .msg-avatar.user {{ background: rgba(255,255,255,0.1); }}
  .msg-bubble {{
    max-width: 80%;
    padding: 9px 13px;
    border-radius: 14px;
    font-size: 0.78rem;
    line-height: 1.55;
    word-wrap: break-word;
  }}
  .msg-bubble.bot {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0;
    border-top-left-radius: 4px;
  }}
  .msg-bubble.user {{
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border-top-right-radius: 4px;
  }}
  .msg-source {{
    font-size: 0.62rem;
    color: #475569;
    margin-top: 4px;
  }}
  .typing-indicator {{
    display: flex; gap: 4px; padding: 9px 13px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; border-top-left-radius: 4px;
    width: fit-content;
  }}
  .typing-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: #64748b;
    animation: typingPulse 1.2s infinite;
  }}
  .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
  .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
  @keyframes typingPulse {{
    0%, 60%, 100% {{ opacity: 0.3; transform: scale(0.8); }}
    30% {{ opacity: 1; transform: scale(1); }}
  }}

  /* ── Input Area ── */
  .input-area {{
    padding: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
    flex-shrink: 0;
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }}
  #chat-input {{
    flex: 1;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    color: #f8fafc;
    padding: 9px 13px;
    font-size: 0.78rem;
    resize: none;
    outline: none;
    min-height: 38px;
    max-height: 100px;
    transition: border-color 0.15s;
    font-family: inherit;
    line-height: 1.4;
  }}
  #chat-input::placeholder {{ color: #475569; }}
  #chat-input:focus {{ border-color: rgba(99,102,241,0.5); }}
  #send-btn {{
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6366f1, #38bdf8);
    border: none; cursor: pointer; color: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    transition: opacity 0.15s, transform 0.15s;
    flex-shrink: 0;
  }}
  #send-btn:hover {{ opacity: 0.85; transform: scale(1.05); }}
  #send-btn:disabled {{ opacity: 0.4; cursor: not-allowed; transform: none; }}
</style>
</head>
<body style="background:transparent;height:{height}px;overflow:hidden;">

<!-- Floating Bubble -->
<button id="chat-bubble" onclick="toggleDrawer()" title="AI Data Analyst Chatbot">
  🤖
  <div id="notif-dot"></div>
</button>

<!-- Chat Drawer -->
<div id="chat-drawer">
  <!-- Header -->
  <div class="drawer-header">
    <div class="drawer-avatar">🤖</div>
    <div class="drawer-title">
      <h3>AI Data Analyst</h3>
      <p>Dataset: <strong style="color:#38bdf8;">{dataset_label}</strong></p>
    </div>
    <div class="status-dot" title="{status_text}"></div>
    <button class="close-btn" onclick="toggleDrawer()">✕</button>
  </div>

  <!-- Suggestion Chips -->
  <div class="suggestions-strip" id="suggestions-strip">
    <!-- Injected by JS -->
  </div>

  <!-- Messages -->
  <div class="messages-area" id="messages-area">
    <div class="empty-state" id="empty-state">
      <div class="icon">💬</div>
      <p>Ask anything about <strong style="color:#38bdf8;">{dataset_label}</strong><br>
         I have the full analysis report, KPIs, insights, and recommendations loaded.</p>
    </div>
  </div>

  <!-- Input -->
  <div class="input-area">
    <textarea id="chat-input" placeholder="Ask about your data…" rows="1"
      onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
    <button id="send-btn" onclick="sendMessage()" title="Send">➤</button>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const MESSAGES_INIT = {messages_json};
const SUGGESTIONS = {questions_json};
let messages = [...MESSAGES_INIT];
let isOpen = false;
let isWaiting = false;

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {{
  renderSuggestions();
  renderMessages();
  if (messages.length === 0) {{
    document.getElementById('notif-dot').style.display = 'block';
  }}
}});

function toggleDrawer() {{
  isOpen = !isOpen;
  const drawer = document.getElementById('chat-drawer');
  const bubble = document.getElementById('chat-bubble');
  const notif = document.getElementById('notif-dot');
  if (isOpen) {{
    drawer.classList.add('visible');
    bubble.classList.add('open');
    bubble.innerHTML = '✕<div id="notif-dot"></div>';
    notif.style.display = 'none';
    document.getElementById('chat-input').focus();
    scrollToBottom();
  }} else {{
    drawer.classList.remove('visible');
    bubble.classList.remove('open');
    bubble.innerHTML = '🤖<div id="notif-dot"></div>';
  }}
}}

// ── Suggestions ───────────────────────────────────────────────────────────
function renderSuggestions() {{
  const strip = document.getElementById('suggestions-strip');
  strip.innerHTML = SUGGESTIONS.map(q =>
    `<div class="suggest-chip" onclick="askSuggestion(this, '${{q.replace(/'/g, "\\'")}}')">
      ${{q.length > 35 ? q.substring(0, 35) + '…' : q}}
    </div>`
  ).join('');
}}

function askSuggestion(el, question) {{
  el.style.opacity = '0.5';
  el.style.pointerEvents = 'none';
  submitQuery(question);
}}

// ── Message Rendering ──────────────────────────────────────────────────────
function renderMessages() {{
  const area = document.getElementById('messages-area');
  const empty = document.getElementById('empty-state');
  if (messages.length === 0) {{
    if (empty) empty.style.display = 'flex';
    return;
  }}
  if (empty) empty.style.display = 'none';

  area.innerHTML = messages.map(m => buildMsgHTML(m)).join('');
  scrollToBottom();
}}

function buildMsgHTML(m) {{
  const isUser = m.role === 'user';
  const avatar = isUser ? '👤' : '🤖';
  const cls = isUser ? 'user' : 'bot';
  const content = m.content.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g, '<br>');
  const sources = m.sources ? `<div class="msg-source">📚 ${{m.sources.join(' · ')}}</div>` : '';
  return `
    <div class="msg-row ${{cls}}">
      <div class="msg-avatar ${{cls}}">${{avatar}}</div>
      <div>
        <div class="msg-bubble ${{cls}}">${{content}}</div>
        ${{sources}}
      </div>
    </div>`;
}}

function showTyping() {{
  const area = document.getElementById('messages-area');
  const empty = document.getElementById('empty-state');
  if (empty) empty.style.display = 'none';
  area.innerHTML += `
    <div class="msg-row bot" id="typing-row">
      <div class="msg-avatar bot">🤖</div>
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  scrollToBottom();
}}

function removeTyping() {{
  const el = document.getElementById('typing-row');
  if (el) el.remove();
}}

// ── Input Handling ─────────────────────────────────────────────────────────
function handleKey(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}}

function autoResize(el) {{
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}}

function sendMessage() {{
  const input = document.getElementById('chat-input');
  const query = input.value.trim();
  if (!query || isWaiting) return;
  input.value = '';
  input.style.height = 'auto';
  submitQuery(query);
}}

function submitQuery(query) {{
  isWaiting = true;
  document.getElementById('send-btn').disabled = true;

  // Add user message to local state
  messages.push({{ role: 'user', content: query }});
  renderMessages();
  showTyping();

  // Send to Streamlit parent
  window.parent.postMessage({{
    type: 'streamlit:setComponentValue',
    value: {{ action: 'chat', query: query, timestamp: Date.now() }}
  }}, '*');
}}

// ── Receive reply from Streamlit ───────────────────────────────────────────
window.addEventListener('message', (e) => {{
  if (e.data && e.data.type === 'rag_reply') {{
    removeTyping();
    messages.push({{
      role: 'assistant',
      content: e.data.answer || 'No response.',
      sources: e.data.sources || []
    }});
    renderMessages();
    isWaiting = false;
    document.getElementById('send-btn').disabled = false;
    if (!isOpen) {{
      document.getElementById('notif-dot').style.display = 'block';
    }}
  }}
}});

function scrollToBottom() {{
  const area = document.getElementById('messages-area');
  area.scrollTop = area.scrollHeight;
}}
</script>
</body>
</html>"""
