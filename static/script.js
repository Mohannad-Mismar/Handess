
// Global state
let chatHistory = [];
let currentSessionId = generateSessionId();
const API_BASE = 'http://127.0.0.1:5000';
let enableThinking = false; // UI toggle for "thinking"

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Initialize demo chat history
function initDemoChat() {
    chatHistory = [
        {
            type: 'user',
            content: 'كم رسوم ترخيص مبنى سكن (ب) مساحة 340 متر مربع + حوض سباحة 20 متر مربع + أسوار 100 متر طولي؟'
        },
        {
            type: 'bot',
            content: `**السند التشريعي:**
حسب نظام رقم (13) لسنة 2025 – جدول رسوم الترخيص لمنطقة السكن (ب):

**الأرقام الرسمية:**
- مساحة البناء: **1.5 دينار لكل متر مربع**
- حوض السباحة: **2.5 دينار لكل متر مربع**
- الأسوار: **0.4 دينار لكل متر طولي**

**الحساب التفصيلي:**
- البناء: 340 × 1.5 = **510 دينار**
- الحوض: 20 × 2.5 = **50 دينار**
- الأسوار: 100 × 0.4 = **40 دينار**

**المجموع الكلي: 600 دينار**

**التطبيق العملي:**
التصميم مطابق للاشتراطات المالية. تأكد من تضمين هذه الرسوم في طلب الترخيص. الأرقام سارية بموجب التعديلات الأخيرة لعام 2025.`,
            sources: ['fees_2025_residential_only.md'],
            responseTime: '1.8s'
        }
    ];
}

// Navigation


// Chat functionality
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const typingIndicator = document.getElementById('typingIndicator');

const thinkingToggleBtn = document.getElementById('thinkingToggleBtn');
if (thinkingToggleBtn) {
    thinkingToggleBtn.addEventListener('click', () => {
        enableThinking = !enableThinking;
        thinkingToggleBtn.classList.toggle('active', enableThinking);
        thinkingToggleBtn.title = enableThinking ? 'Thinking enabled' : 'Thinking disabled';
        addTraceEntry('UI', `Thinking ${enableThinking ? 'enabled' : 'disabled'}`);
    });
}

function initChat() {
    initDemoChat();
    renderMessages();
    if (messageInput) messageInput.focus();
}

function renderMessages() {
    if (!messagesContainer) return;
    messagesContainer.innerHTML = '';
    chatHistory.forEach((msg, index) => {
        addMessageToDOM(msg, index);
    });
    scrollToBottom();
}

function addMessageToDOM(message, index) {
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.type}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    if (message.type === 'bot') {
        content.innerHTML = parseContent(message.content);
        
        // Add thinking container if thinking is present
        if (message.thinking && message.thinking.trim()) {
            const thinkingContainer = document.createElement('div');
            thinkingContainer.className = 'thinking-container';
            
            const thinkingHeader = document.createElement('div');
            thinkingHeader.className = 'thinking-header';
            thinkingHeader.innerHTML = '<span class="thinking-icon">🧠</span> <span>AI Thinking Process</span> <span class="thinking-toggle">▼</span>';
            
            const thinkingContent = document.createElement('div');
            thinkingContent.className = 'thinking-content';
            thinkingContent.innerHTML = parseContent(message.thinking);
            
            thinkingHeader.addEventListener('click', () => {
                thinkingContent.classList.toggle('expanded');
                const toggle = thinkingHeader.querySelector('.thinking-toggle');
                toggle.textContent = thinkingContent.classList.contains('expanded') ? '▲' : '▼';
            });
            
            thinkingContainer.appendChild(thinkingHeader);
            thinkingContainer.appendChild(thinkingContent);
            bubble.appendChild(thinkingContainer);
        }
        
        // Add metadata for bot messages
        if (message.sources || message.responseTime) {
            const metadata = document.createElement('div');
            metadata.style.cssText = 'margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.1); font-size: 12px; color: #666;';
            if (message.sources) {
                metadata.innerHTML += `<strong>Sources:</strong> ${message.sources.join(', ')}<br>`;
            }
            if (message.responseTime) {
                metadata.innerHTML += `<strong>Response Time:</strong> ${message.responseTime}<br>`;
            }
            bubble.appendChild(metadata);
        }
    } else {
        content.textContent = message.content;
    }
    bubble.appendChild(content);
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);
    
    // Add animation with delay
    setTimeout(() => {
        messageDiv.style.opacity = '1';
        messageDiv.style.transform = 'translateY(0)';
    }, 100);
}

function parseContent(content) {
    if (!content || typeof content !== 'string') {
        return '<em>No content returned</em>';
    }

    // Basic markdown styling
    let html = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

    // Block math: $$ ... $$
    html = html.replace(/\$\$([\s\S]*?)\$\$/g, '<div class="math-block">$1</div>');
    // Inline math: $ ... $ (avoid matching $$)
    html = html.replace(/\$(?!\$)([^\n$]+?)\$(?!\$)/g, '<span class="math-inline">$1</span>');

    // Line breaks last
    html = html.replace(/\n/g, '<br>');

    return html;
}


function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

async function sendMessage() {
    const question = messageInput.value.trim();
    if (!question) return;

    // Add user message
    const userMessage = {
        type: 'user',
        content: question
    };
    chatHistory.push(userMessage);
    addMessageToDOM(userMessage, chatHistory.length - 1);
    messageInput.value = '';

    typingIndicator.style.display = 'flex';
    scrollToBottom();

    try {
        // Convert chatHistory to backend format: [{role: 'user'|'assistant', content: string}, ...]
        // Include previous messages BEFORE the current question for context
        const conversationHistory = chatHistory.slice(0, -1).map(msg => ({
            role: msg.type === 'user' ? 'user' : 'assistant',
            content: msg.content
        }));

        const response = await fetch(API_BASE + '/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                enable_thinking: enableThinking,
                conversation_history: conversationHistory,  // NEW: Send conversation history
                session_id: currentSessionId
            })
        });

        // ✅ اقرأ الرد كنص أولاً (آمن)
        const rawText = await response.text();

        let result;
        try {
            result = JSON.parse(rawText);
        } catch {
            throw new Error('Server returned invalid response');
        }

        if (!response.ok) {
            throw new Error(result.error || 'Unknown server error');
        }

        if (!result.content || typeof result.content !== 'string') {
            throw new Error('Empty response from server');
        }

        // Add bot response
        const botMessage = {
            type: 'bot',
            content: result.content,
            thinking: result.thinking || '',
            sources: Array.isArray(result.sources) ? result.sources : [],
            responseTime: result.response_time || null
        };

        chatHistory.push(botMessage);
        addMessageToDOM(botMessage, chatHistory.length - 1);

    } catch (error) {
        const errorMessage = {
            type: 'bot',
            content: `❌ Error: ${error.message || 'Unexpected error'}`
        };
        chatHistory.push(errorMessage);
        addMessageToDOM(errorMessage, chatHistory.length - 1);

    } finally {
        typingIndicator.style.display = 'none';
        scrollToBottom();
    }
}


function clearChat() {
    chatHistory = [];
    renderMessages();
    
    // Also clear on server
    fetch(API_BASE + '/api/clear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: currentSessionId
        })
    });
}



// Event listeners
sendBtn.addEventListener('click', sendMessage);
clearBtn.addEventListener('click', clearChat);

messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing chat...');
    initChat();
});
