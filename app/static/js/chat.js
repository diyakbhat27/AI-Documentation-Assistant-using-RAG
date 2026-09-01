const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const sidebarList = document.getElementById('sidebarList');
const sidebarEmpty = document.getElementById('sidebarEmpty');

// History stored in memory
const history = [];

// Configure marked
marked.setOptions({
    breaks: true,
    gfm: true
});

// Auto-resize textarea
questionInput.addEventListener('input', () => {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
});

// Send on Enter, newline on Shift+Enter
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

function sendMessage() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Disable input while waiting
    setInputState(false);

    // Append user bubble
    appendUserBubble(question);

    // Clear input
    questionInput.value = '';
    questionInput.style.height = 'auto';

    // Show thinking indicator
    const thinkingEl = appendThinking();

    // Call API
    fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(res => {
        if (!res.ok) throw new Error('API error');
        return res.json();
    })
    .then(data => {
        thinkingEl.remove();
        appendAssistantBubble(data.answer, data.sources);
        addToSidebar(question);
    })
    .catch(() => {
        thinkingEl.remove();
        appendErrorBubble();
    })
    .finally(() => {
        setInputState(true);
        scrollToBottom();
    });
}

function appendUserBubble(text) {
    const msg = document.createElement('div');
    msg.className = 'message user';
    msg.innerHTML = `<div class="message-bubble">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function appendAssistantBubble(answer, sources) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';

    // Parse markdown — this is what fixes your broken formatting
    const parsedAnswer = marked.parse(answer);

    // Build sources HTML
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const links = sources.map(s =>
            `<a class="source-link" href="${s.url}" target="_blank" rel="noopener">
                ${escapeHtml(s.title)}
             </a>`
        ).join('');

        sourcesHtml = `
            <div class="sources-section">
                <span class="sources-label">Sources</span>
                <div class="sources-list">${links}</div>
            </div>
        `;
    }

    msg.innerHTML = `
        <div class="message-bubble">
            <div class="answer-content">${parsedAnswer}</div>
            ${sourcesHtml}
        </div>
    `;

    // Run syntax highlighting on all code blocks
    msg.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });

    chatMessages.appendChild(msg);
    scrollToBottom();
}

function appendErrorBubble() {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `
        <div class="message-bubble">
            Something went wrong. Please try again.
        </div>
    `;
    chatMessages.appendChild(msg);
}

function appendThinking() {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `
        <div class="thinking">
            <span></span><span></span><span></span>
        </div>
    `;
    chatMessages.appendChild(msg);
    scrollToBottom();
    return msg;
}

function addToSidebar(question) {
    // Remove empty state
    if (sidebarEmpty) sidebarEmpty.style.display = 'none';

    // Remove active from all items
    document.querySelectorAll('.sidebar-item').forEach(el => {
        el.classList.remove('active');
    });

    const item = document.createElement('div');
    item.className = 'sidebar-item active';
    // Truncate long questions for display
    item.textContent = question.length > 35
        ? question.substring(0, 35) + '...'
        : question;
    item.title = question; // Full text on hover

    sidebarList.appendChild(item);
    history.push(question);
}

function setInputState(enabled) {
    questionInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
    if (enabled) questionInput.focus();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
