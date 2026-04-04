"""
VoiceBot Blueprint - AI-powered conversational voice assistant for LoanZone.
Uses Google Gemini REST API directly (no SDK dependency).
"""

from flask import Blueprint, request, jsonify
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

voicebot_bp = Blueprint('voicebot', __name__)

# ──────────────────────────────────────────────────
# Configure Gemini API
# ──────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# ──────────────────────────────────────────────────
# System prompt with full LoanZone knowledge
# ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LoanZone AI Assistant, a friendly and knowledgeable voice assistant for the LoanZone loan prediction website. You help users navigate the site, fill out loan forms, understand loan concepts, and get predictions.

IMPORTANT RULES:
1. Keep responses SHORT and conversational (2-3 sentences max). Users are LISTENING, not reading.
2. Be warm, professional, and helpful.
3. Always respond with valid JSON in the exact format specified below.
4. Never mention that you are an AI or language model. You are "LoanZone Assistant".
5. ALWAYS use standard English numerals (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) for ALL numbers. NEVER use Devanagari, Arabic-Indic, or any non-Latin numerals.
6. Respond in the language specified by the LANGUAGE instruction below. Use rupee symbol as Rs. or INR, not ₹.

AVAILABLE PAGES (use exact paths for navigation):
- "/" or "/index" — Home page (landing page with overview of LoanZone)
- "/Dashboard_page" — Dashboard (analytics & visualizations of loan data)
- "/personalLoan_page" — Personal Loan Application (multi-step form)
- "/LoanApplication_page" — General Loan Default Prediction form
- "/calculator_page" — EMI Calculator
- "/RiskAnalysis_page" — Data Analysis & Risk Analysis page
- "/contactus_page" — Contact Us page
- "/defaultPrediction_page" — Default Prediction info page
- "/autoLoan_page" — Auto Loan info
- "/homeLoan_page" — Home Loan info
- "/educationLoan_page" — Education Loan info
- "/businessLoan_page" — Business Loan info

LOAN APPLICATION FORM FIELDS (for /LoanApplication_page):
- age: Age (18-100)
- income: Annual Income (min 10000)
- loanAmount: Loan Amount (min 100000)
- creditScore: Credit Score (300-850)
- monthsEmployed: Months Employed (0+)
- numCreditLines: Number of Credit Lines (0-20)
- loanTerm: Loan Term in months (6-360)
- dtiRatio: DTI Ratio (0-100, percentage)
- education: Education level (1=High School, 2=Bachelor's, 3=Master's, 4=PhD)
- employmentType: Employment (1=Unemployed, 2=Part-time, 3=Full-time, 4=Self-employed)
- maritalStatus: Marital Status (1=Single, 2=Married, 3=Divorced)
- hasMortgage: Has Mortgage (1=Yes, 0=No)
- hasDependents: Has Dependents (1=Yes, 0=No)
- loanPurpose: Loan Purpose (1=Auto, 2=Business, 3=Education, 4=Home, 5=Other)
- hasCosigner: Has Co-signer (1=Yes, 0=No)

PERSONAL LOAN FORM FIELDS (for /personalLoan_page):
- fullName: Full Name
- pannumber: PAN Number (format: ABCDE1234F)
- email: Email address
- phone: 10-digit phone number
- dob: Date of Birth (YYYY-MM-DD)
- monthlyIncome: Monthly Income (min 10000)
- monthlyDebt: Monthly Debt Payments
- employmentType: "salaried" or "self-employed"
- companyName: Company Name
- CreditScore: Credit Score (300-850)
- NumCreditLines: Number of Credit Lines
- loanAmount: Loan Amount (min 10000)
- tenure: Loan Tenure in months (min 6)

LOAN KNOWLEDGE:
- DTI Ratio = (Monthly Debt / Monthly Income) x 100. Below 20% is excellent, 20-35% good, 36-43% manageable, above 43% high risk.
- Credit Score: 800+ Very Low Risk, 750-799 Low Risk, 700-749 Moderate Risk, 650-699 High Risk, below 650 Very High Risk.
- EMI = P x r x (1+r)^n / ((1+r)^n - 1) where P=principal, r=monthly interest rate, n=number of months.
- Personal loans: For personal needs, competitive interest rates.
- Home loans: For property purchase, usually lower interest rates, longer tenure.
- Auto loans: For vehicle purchase, moderate interest rates.
- Education loans: For studies, can have moratorium period.
- Business loans: For business expansion, requires business documentation.

RESPONSE FORMAT — You MUST always respond with ONLY valid JSON (no markdown, no code fences):
{
    "reply": "Your conversational response text here",
    "action": {
        "type": "navigate|fill_form|calculate_emi|explain|greet",
        "target": "/page_path",
        "fields": {"field_id": "value"},
        "amount": 0,
        "rate": 0,
        "tenure": 0
    }
}

ACTION TYPES:
- "navigate": Set "target" to the page path. Use when user wants to go to a page.
- "fill_form": Set "fields" dict with form field IDs and values. Use when user provides details for a form.
- "calculate_emi": Set "amount", "rate", "tenure". Use when user asks for EMI calculation.
- "explain": Just reply with information, no other action needed.
- "greet": For greetings and casual conversation.

EXAMPLES:
User: "Take me to the dashboard"
{"reply": "Taking you to the dashboard now! Here you can see all the analytics and loan data visualizations.", "action": {"type": "navigate", "target": "/Dashboard_page"}}

User: "My age is 28 and income is 50000"
{"reply": "Got it! I've filled in your age as 28 and income as 50,000. What other details would you like to provide?", "action": {"type": "fill_form", "fields": {"age": "28", "income": "50000"}}}

User: "What is a good credit score?"
{"reply": "A credit score of 750 or above is considered good. Scores above 800 are excellent and give you the best loan terms. Below 650 is considered high risk.", "action": {"type": "explain"}}

User: "Calculate EMI for 5 lakh at 10% for 3 years"
{"reply": "For a loan of 5,00,000 at 10% for 3 years (36 months), your monthly EMI would be approximately 16,134. Shall I take you to the EMI calculator for a detailed breakdown?", "action": {"type": "calculate_emi", "amount": 500000, "rate": 10, "tenure": 36}}

User: "Hi" or "Hello"
{"reply": "Hello! Welcome to LoanZone. I'm your AI assistant. I can help you apply for loans, calculate EMIs, navigate the site, or answer any questions about loans. How can I help you today?", "action": {"type": "greet"}}
"""

# ──────────────────────────────────────────────────
# Conversation memory (per-session, simple approach)
# ──────────────────────────────────────────────────
conversation_histories = {}

def get_conversation_history(session_id):
    """Get or create conversation history for a session."""
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    return conversation_histories[session_id]

def add_to_history(session_id, role, text):
    """Add a message to conversation history, keep last 10 exchanges."""
    history = get_conversation_history(session_id)
    history.append({"role": role, "text": text})
    # Keep only last 20 messages (10 exchanges)
    if len(history) > 20:
        conversation_histories[session_id] = history[-20:]

def format_history_for_prompt(session_id):
    """Format conversation history for the prompt."""
    history = get_conversation_history(session_id)
    if not history:
        return ""
    
    formatted = "\n\nCONVERSATION HISTORY:\n"
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role_label}: {msg['text']}\n"
    return formatted


def call_gemini_api(prompt):
    """Call Gemini API via REST (no SDK needed)."""
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 500
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    # Build URL with current API key (in case env var changed)
    api_key = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    print(f"[VoiceBot] Calling Gemini API with key: {api_key[:10]}...")
    
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    
    if response.status_code != 200:
        print(f"[VoiceBot] API Error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
    
    result = response.json()
    
    # Extract text from Gemini response
    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "")
    
    return ""

# Print loaded key on startup
print(f"[VoiceBot] Loaded GEMINI_API_KEY: {GEMINI_API_KEY[:10]}..." if len(GEMINI_API_KEY) > 10 else f"[VoiceBot] WARNING: API key looks invalid: {GEMINI_API_KEY}")



# ──────────────────────────────────────────────────
# Voice Chat Endpoint
# ──────────────────────────────────────────────────
@voicebot_bp.route('/voice/chat', methods=['POST'])
def voice_chat():
    """Process user text input and return AI response with action."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "reply": "I didn't catch that. Could you please say that again?",
                "action": {"type": "explain"}
            }), 400

        user_text = data['text'].strip()
        session_id = data.get('session_id', 'default')
        current_page = data.get('current_page', '/')
        language = data.get('language', 'en')  # 'en' or 'te'

        if not user_text:
            return jsonify({
                "reply": "I didn't hear anything. Please try again.",
                "action": {"type": "explain"}
            }), 400

        # Add context about current page
        page_context = f"\n\nCURRENT PAGE: The user is currently on '{current_page}'."
        
        # Add language instruction
        if language == 'te':
            lang_instruction = "\n\nLANGUAGE: Respond ENTIRELY in Telugu (తెలుగు). Use Telugu script for all text in the 'reply' field. Keep JSON keys in English. Use standard English numerals (0-9) for numbers even in Telugu responses."
        else:
            lang_instruction = "\n\nLANGUAGE: Respond entirely in English."
        
        # Build the full prompt
        history_text = format_history_for_prompt(session_id)
        full_prompt = SYSTEM_PROMPT + page_context + lang_instruction + history_text + f"\n\nUser: {user_text}"

        # Call Gemini REST API
        response_text = call_gemini_api(full_prompt)
        
        # Clean up response — remove markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            response_text = response_text.strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If Gemini didn't return valid JSON, wrap it
            result = {
                "reply": response_text if len(response_text) < 500 else "I can help you with loan applications, EMI calculations, and navigating the site. What would you like to do?",
                "action": {"type": "explain"}
            }

        # Ensure required fields exist
        if "reply" not in result:
            result["reply"] = "I'm here to help! What would you like to know about our loan services?"
        if "action" not in result:
            result["action"] = {"type": "explain"}

        # Save to conversation history
        add_to_history(session_id, "user", user_text)
        add_to_history(session_id, "assistant", result["reply"])

        return jsonify(result)

    except requests.exceptions.Timeout:
        return jsonify({
            "reply": "I'm taking too long to respond. Please try again.",
            "action": {"type": "explain"}
        }), 504
    except requests.exceptions.RequestException as e:
        print(f"VoiceBot API Error: {str(e)}")
        return jsonify({
            "reply": "I'm having trouble connecting to my brain right now. Please make sure the API key is set correctly and try again.",
            "action": {"type": "explain"}
        }), 500
    except Exception as e:
        print(f"VoiceBot Error: {str(e)}")
        return jsonify({
            "reply": "I'm having a bit of trouble right now. Please try again in a moment.",
            "action": {"type": "explain"}
        }), 500


@voicebot_bp.route('/voice/clear', methods=['POST'])
def clear_history():
    """Clear conversation history for a session."""
    data = request.get_json() or {}
    session_id = data.get('session_id', 'default')
    
    if session_id in conversation_histories:
        del conversation_histories[session_id]
    
    return jsonify({"status": "cleared"})
