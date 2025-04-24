/**
 * ═══════════════════════════════════════════════════
 * LoanZone Voice Bot — Frontend Module
 * AI-powered conversational voice assistant
 * ═══════════════════════════════════════════════════
 */

(function () {
    'use strict';

    // ── SVG Icons ──
    const ICONS = {
        mic: `<svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>`,
        stop: `<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`,
        send: `<svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`,
        bot: `<svg viewBox="0 0 24 24"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1.17A7.002 7.002 0 0 1 8 18H7a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 3-5.77V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2zm-3 11a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3zm6 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3z"/></svg>`,
        close: `<svg viewBox="0 0 24 24" width="16" height="16"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/></svg>`,
        chat: `<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" fill="white"/></svg>`,
    };

    // ── Session ID ──
    const SESSION_ID = 'vb_' + Math.random().toString(36).substr(2, 9);

    // ── State ──
    let isOpen = false;
    let isListening = false;
    let isSpeaking = false;
    let isProcessing = false;
    let recognition = null;
    let synthesis = window.speechSynthesis;
    let currentLang = 'en'; // 'en' or 'te'

    const LANG_CONFIG = {
        en: { code: 'en', sttLang: 'en-US', ttsLang: 'en-US', label: 'EN', fullName: 'English', placeholder: 'Type or tap mic to speak...', welcome: '\u{1F44B} Hello! I\'m your LoanZone AI Assistant. I can help you apply for loans, calculate EMIs, navigate the site, or answer questions. How can I help you today?' },
        te: { code: 'te', sttLang: 'te-IN', ttsLang: 'te-IN', label: '\u0C24\u0C46', fullName: '\u0C24\u0C46\u0C32\u0C41\u0C17\u0C41', placeholder: '\u0C1F\u0C48\u0C2A\u0C4D \u0C1A\u0C47\u0C2F\u0C02\u0C21\u0C3F \u0C32\u0C47\u0C26\u0C3E \u0C2E\u0C48\u0C15\u0C4D \u0C28\u0C4A\u0C15\u0C4D\u0C15\u0C02\u0C21\u0C3F...', welcome: '\u{1F44B} \u0C28\u0C2E\u0C38\u0C4D\u0C15\u0C3E\u0C30\u0C02! \u0C28\u0C47\u0C28\u0C41 \u0C2E\u0C40 LoanZone AI \u0C38\u0C39\u0C3E\u0C2F\u0C15\u0C41\u0C21\u0C3F\u0C28\u0C3F. \u0C30\u0C41\u0C23\u0C3E\u0C32\u0C15\u0C41 \u0C26\u0C30\u0C16\u0C3E\u0C38\u0C4D\u0C24\u0C41 \u0C1A\u0C47\u0C2F\u0C21\u0C02, EMI \u0C32\u0C46\u0C15\u0C4D\u0C15\u0C3F\u0C02\u0C1A\u0C21\u0C02, \u0C32\u0C47\u0C26\u0C3E \u0C2A\u0C4D\u0C30\u0C36\u0C4D\u0C28\u0C32\u0C15\u0C41 \u0C38\u0C2E\u0C3E\u0C27\u0C3E\u0C28\u0C3E\u0C32\u0C41 \u0C2A\u0C4A\u0C02\u0C26\u0C21\u0C02 \u0C35\u0C02\u0C1F\u0C3F \u0C35\u0C3E\u0C1F\u0C3F\u0C32\u0C4B \u0C28\u0C47\u0C28\u0C41 \u0C2E\u0C40\u0C15\u0C41 \u0C38\u0C39\u0C3E\u0C2F\u0C02 \u0C1A\u0C47\u0C2F\u0C17\u0C32\u0C28\u0C41.' }
    };

    // ── Build UI ──
    function buildUI() {
        // FAB Button
        const fab = document.createElement('button');
        fab.id = 'voicebot-fab';
        fab.setAttribute('aria-label', 'Open voice assistant');
        fab.innerHTML = ICONS.chat;
        fab.addEventListener('click', togglePanel);

        // Chat Panel
        const panel = document.createElement('div');
        panel.id = 'voicebot-panel';
        panel.innerHTML = `
            <div class="vb-header">
                <div class="vb-header-avatar">${ICONS.bot}</div>
                <div class="vb-header-info">
                    <div class="vb-header-title">LoanZone Assistant</div>
                    <div class="vb-header-status">
                        <span class="vb-status-dot"></span>
                        <span id="vb-status-text">Online</span>
                    </div>
                </div>
                <div class="vb-lang-toggle" id="vb-lang-toggle" title="Switch language">
                    <button class="vb-lang-btn active" data-lang="en">EN</button>
                    <button class="vb-lang-btn" data-lang="te">\u0C24\u0C46</button>
                </div>
                <button class="vb-header-close" id="vb-close-btn" aria-label="Close">${ICONS.close}</button>
            </div>
            <div class="vb-messages" id="vb-messages"></div>
            <div class="vb-quick-actions" id="vb-quick-actions">
                <button class="vb-chip" data-text="What can you do?">\u{1F4A1} What can you do?</button>
                <button class="vb-chip" data-text="Apply for a loan">\u{1F4DD} Apply for loan</button>
                <button class="vb-chip" data-text="Calculate EMI">\u{1F9EE} Calculate EMI</button>
                <button class="vb-chip" data-text="Show dashboard">\u{1F4CA} Dashboard</button>
            </div>
            <div class="vb-input-area">
                <input type="text" class="vb-text-input" id="vb-text-input" placeholder="Type or tap mic to speak..." autocomplete="off">
                <button class="vb-mic-btn" id="vb-mic-btn" aria-label="Voice input">${ICONS.mic}</button>
                <button class="vb-send-btn" id="vb-send-btn" aria-label="Send message">${ICONS.send}</button>
            </div>
            <div class="vb-powered">Powered by AI \u2728</div>
        `;

        document.body.appendChild(fab);
        document.body.appendChild(panel);

        // Event listeners
        document.getElementById('vb-close-btn').addEventListener('click', togglePanel);
        document.getElementById('vb-mic-btn').addEventListener('click', toggleListening);
        document.getElementById('vb-send-btn').addEventListener('click', sendTextInput);
        document.getElementById('vb-text-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTextInput();
            }
        });

        // Quick action chips
        document.querySelectorAll('.vb-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const text = chip.getAttribute('data-text');
                processUserInput(text);
            });
        });

        // Language toggle buttons
        document.querySelectorAll('.vb-lang-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const lang = btn.getAttribute('data-lang');
                switchLanguage(lang);
            });
        });

        // Add welcome message after a short delay
        setTimeout(() => {
            addMessage('bot', LANG_CONFIG[currentLang].welcome);
        }, 500);
    }

    // ── Toggle Panel ──
    function togglePanel() {
        isOpen = !isOpen;
        const panel = document.getElementById('voicebot-panel');
        const fab = document.getElementById('voicebot-fab');

        if (isOpen) {
            panel.classList.add('open');
            fab.classList.add('open');
            fab.innerHTML = ICONS.close;
            document.getElementById('vb-text-input').focus();
        } else {
            panel.classList.remove('open');
            fab.classList.remove('open');
            fab.innerHTML = ICONS.chat;
            stopListening();
            stopSpeaking();
        }
    }

    // ── Add Message to Chat ──
    function addMessage(type, text) {
        const container = document.getElementById('vb-messages');
        const msg = document.createElement('div');
        msg.className = `vb-message ${type}`;

        const avatarContent = type === 'bot' ? '🤖' : '👤';
        msg.innerHTML = `
            <div class="vb-msg-avatar">${avatarContent}</div>
            <div class="vb-msg-bubble">${text}</div>
        `;

        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        return msg;
    }

    // ── Typing Indicator ──
    function showTyping() {
        const container = document.getElementById('vb-messages');
        const typing = document.createElement('div');
        typing.className = 'vb-message bot';
        typing.id = 'vb-typing';
        typing.innerHTML = `
            <div class="vb-msg-avatar">🤖</div>
            <div class="vb-msg-bubble">
                <div class="vb-typing">
                    <div class="vb-typing-dot"></div>
                    <div class="vb-typing-dot"></div>
                    <div class="vb-typing-dot"></div>
                </div>
            </div>
        `;
        container.appendChild(typing);
        container.scrollTop = container.scrollHeight;
    }

    function hideTyping() {
        const typing = document.getElementById('vb-typing');
        if (typing) typing.remove();
    }

    // ── Update Status ──
    function setStatus(text) {
        const el = document.getElementById('vb-status-text');
        if (el) el.textContent = text;
    }

    // ── Speech Recognition (STT) ──
    function initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.warn('Speech Recognition not supported in this browser.');
            return null;
        }

        const recog = new SpeechRecognition();
        recog.continuous = false;
        recog.interimResults = false;
        recog.lang = LANG_CONFIG[currentLang].sttLang;
        recog.maxAlternatives = 1;

        recog.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            console.log('[VoiceBot] Heard:', transcript);
            stopListeningUI();
            processUserInput(transcript);
        };

        recog.onerror = (event) => {
            console.error('[VoiceBot] Speech error:', event.error);
            stopListeningUI();
            if (event.error === 'no-speech') {
                setStatus('Online');
            } else if (event.error === 'not-allowed') {
                addMessage('bot', "⚠️ Microphone access was denied. Please allow microphone access in your browser settings, or you can type your message instead.");
            }
        };

        recog.onend = () => {
            stopListeningUI();
        };

        return recog;
    }

    function toggleListening() {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    }

    function startListening() {
        // Stop any ongoing speech
        stopSpeaking();

        // Re-init recognition to pick up language changes
        recognition = initSpeechRecognition();
        if (!recognition) {
            addMessage('bot', "❌ Voice input is not supported in this browser. Please use Chrome for the best experience, or type your message below.");
            return;
        }

        try {
            recognition.start();
            isListening = true;
            const micBtn = document.getElementById('vb-mic-btn');
            micBtn.classList.add('listening');
            micBtn.innerHTML = ICONS.stop;
            setStatus('🎤 Listening...');
        } catch (e) {
            console.error('[VoiceBot] Start error:', e);
        }
    }

    function stopListening() {
        if (recognition && isListening) {
            recognition.stop();
        }
        stopListeningUI();
    }

    function stopListeningUI() {
        isListening = false;
        const micBtn = document.getElementById('vb-mic-btn');
        if (micBtn) {
            micBtn.classList.remove('listening');
            micBtn.innerHTML = ICONS.mic;
        }
        setStatus('Online');
    }

    // ── Speech Synthesis (TTS) ──
    function speak(text) {
        if (!synthesis) return;

        // Stop current speech
        stopSpeaking();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = LANG_CONFIG[currentLang].ttsLang;
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        // Try to use a natural-sounding voice for the selected language
        const voices = synthesis.getVoices();
        const langPrefix = currentLang === 'te' ? 'te' : 'en';
        const preferred = voices.find(v =>
            v.lang.startsWith(langPrefix) && (v.name.includes('Google') || v.name.includes('Natural'))
        ) || voices.find(v => v.lang.startsWith(langPrefix)) || voices[0];

        if (preferred) utterance.voice = preferred;

        utterance.onstart = () => {
            isSpeaking = true;
            setStatus('🔊 Speaking...');
        };

        utterance.onend = () => {
            isSpeaking = false;
            setStatus('Online');
        };

        utterance.onerror = () => {
            isSpeaking = false;
            setStatus('Online');
        };

        synthesis.speak(utterance);
    }

    function stopSpeaking() {
        if (synthesis) {
            synthesis.cancel();
            isSpeaking = false;
        }
    }

    // ── Send Text Input ──
    function sendTextInput() {
        const input = document.getElementById('vb-text-input');
        const text = input.value.trim();
        if (!text || isProcessing) return;
        input.value = '';
        processUserInput(text);
    }

    // ── Process User Input (core flow) ──
    async function processUserInput(text) {
        if (isProcessing) return;
        isProcessing = true;

        // Add user message
        addMessage('user', text);

        // Hide quick actions after first message
        const quickActions = document.getElementById('vb-quick-actions');
        if (quickActions) quickActions.style.display = 'none';

        // Show typing indicator
        showTyping();
        setStatus('🤔 Thinking...');

        try {
            const response = await fetch('/voice/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    session_id: SESSION_ID,
                    current_page: window.location.pathname,
                    language: currentLang
                })
            });

            hideTyping();

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();

            // Add bot reply
            addMessage('bot', data.reply || "I'm here to help!");

            // Speak the reply
            speak(data.reply || "I'm here to help!");

            // Execute action
            if (data.action) {
                executeAction(data.action);
            }

        } catch (error) {
            hideTyping();
            console.error('[VoiceBot] Error:', error);
            const errorMsg = "I'm having trouble connecting right now. Please check that the server is running and try again.";
            addMessage('bot', '⚠️ ' + errorMsg);
        } finally {
            isProcessing = false;
            setStatus('Online');
        }
    }

    // ── Execute Actions ──
    function executeAction(action) {
        if (!action || !action.type) return;

        switch (action.type) {
            case 'navigate':
                if (action.target) {
                    // Small delay so user can hear the response first
                    setTimeout(() => {
                        window.location.href = action.target;
                    }, 2000);
                }
                break;

            case 'fill_form':
                if (action.fields) {
                    fillFormFields(action.fields);
                }
                break;

            case 'calculate_emi':
                if (action.amount && action.rate && action.tenure) {
                    calculateAndShowEMI(action.amount, action.rate, action.tenure);
                }
                break;

            case 'explain':
            case 'greet':
                // No additional action needed — reply is already shown
                break;

            case 'open_url':
                if (action.url) {
                    setTimeout(() => {
                        window.open(action.url, '_blank');
                    }, 1500);
                }
                break;

            default:
                break;
        }
    }

    // ── Fill Form Fields ──
    function fillFormFields(fields) {
        let filledCount = 0;

        for (const [fieldId, value] of Object.entries(fields)) {
            // Try multiple selectors to find the field
            let element = document.getElementById(fieldId);

            // Try by name attribute
            if (!element) {
                element = document.querySelector(`[name="${fieldId}"]`);
            }

            // Try case-insensitive ID match
            if (!element) {
                const allInputs = document.querySelectorAll('input, select, textarea');
                for (const input of allInputs) {
                    if (input.id && input.id.toLowerCase() === fieldId.toLowerCase()) {
                        element = input;
                        break;
                    }
                    if (input.name && input.name.toLowerCase() === fieldId.toLowerCase()) {
                        element = input;
                        break;
                    }
                }
            }

            if (element) {
                // Handle disabled fields
                const wasDisabled = element.disabled;
                if (wasDisabled) element.disabled = false;

                // Set value
                if (element.tagName === 'SELECT') {
                    // For select elements, match by value
                    const options = element.options;
                    for (let i = 0; i < options.length; i++) {
                        if (options[i].value === String(value) ||
                            options[i].textContent.toLowerCase().includes(String(value).toLowerCase())) {
                            element.selectedIndex = i;
                            break;
                        }
                    }
                } else {
                    element.value = value;
                }

                // Re-disable if it was disabled
                if (wasDisabled) element.disabled = true;

                // Trigger events so form validation picks up the change
                element.dispatchEvent(new Event('input', { bubbles: true }));
                element.dispatchEvent(new Event('change', { bubbles: true }));

                // Visual feedback — brief highlight
                element.style.transition = 'box-shadow 0.3s ease, border-color 0.3s ease';
                element.style.boxShadow = '0 0 0 3px rgba(0, 209, 255, 0.4)';
                element.style.borderColor = '#00D1FF';
                setTimeout(() => {
                    element.style.boxShadow = '';
                    element.style.borderColor = '';
                }, 2000);

                filledCount++;
            }
        }

        if (filledCount > 0) {
            console.log(`[VoiceBot] Filled ${filledCount} form fields`);
        }
    }

    // ── Calculate EMI ──
    function calculateAndShowEMI(principal, annualRate, tenureMonths) {
        const r = annualRate / 12 / 100;
        const n = tenureMonths;

        if (r === 0) {
            const emi = principal / n;
            addMessage('bot', `📊 EMI: ₹${emi.toFixed(2)}/month (0% interest)`);
            return;
        }

        const emi = principal * r * Math.pow(1 + r, n) / (Math.pow(1 + r, n) - 1);
        const totalPayment = emi * n;
        const totalInterest = totalPayment - principal;

        addMessage('bot', `📊 <strong>EMI Breakdown:</strong><br>
            Monthly EMI: ₹${emi.toFixed(2)}<br>
            Total Payment: ₹${totalPayment.toFixed(2)}<br>
            Total Interest: ₹${totalInterest.toFixed(2)}`);
    }

    // ── Switch Language ──
    function switchLanguage(lang) {
        if (lang === currentLang) return;
        currentLang = lang;

        // Update toggle button styles
        document.querySelectorAll('.vb-lang-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-lang') === lang);
        });

        // Update placeholder text
        const input = document.getElementById('vb-text-input');
        if (input) input.placeholder = LANG_CONFIG[lang].placeholder;

        // Notify user of language change
        const langName = LANG_CONFIG[lang].fullName;
        addMessage('bot', lang === 'te' 
            ? `\u{1F310} \u0C2D\u0C3E\u0C37 \u0C24\u0C46\u0C32\u0C41\u0C17\u0C41\u0C15\u0C41 \u0C2E\u0C3E\u0C30\u0C1A\u0C2C\u0C21\u0C3F\u0C02\u0C26\u0C3F. \u0C2E\u0C40\u0C30\u0C41 \u0C07\u0C2A\u0C4D\u0C2A\u0C41\u0C21\u0C41 \u0C24\u0C46\u0C32\u0C41\u0C17\u0C41\u0C32\u0C4B \u0C2E\u0C3E\u0C1F\u0C4D\u0C32\u0C3E\u0C21\u0C35\u0C1A\u0C4D\u0C1A\u0C41!`
            : `\u{1F310} Language switched to ${langName}. You can now speak or type in ${langName}!`);

        // Clear conversation history for fresh start in new language
        fetch('/voice/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: SESSION_ID })
        }).catch(() => {});
    }

    // ── Initialize ──
    function init() {
        // Wait for DOM
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', buildUI);
        } else {
            buildUI();
        }

        // Load voices (needed for TTS)
        if (synthesis) {
            synthesis.getVoices();
            synthesis.onvoiceschanged = () => synthesis.getVoices();
        }
    }

    // Start!
    init();
})();
