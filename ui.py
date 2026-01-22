"""
Minimal Modern Chatbot UI for Financial RAG System
-------------------------------------------------
⚠️ IMPORTANT:
- Pure chatbot UI only
- Assumes documents are ALREADY embedded & stored
- No sidebar configs, no processing buttons
- Clean ChatGPT-like experience

Run using:
streamlit run ui.py
"""

import streamlit as st
from datetime import datetime
import os
from apps import FinancialRAGSystem

# Import your existing RAG system (DO NOT MODIFY THAT FILE)
# from rag_system import FinancialRAGSystem

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💰",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 820px;
    margin: auto;
}
.user-msg {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 14px 18px;
    border-radius: 18px;
    margin: 10px 0;
    width: fit-content;
    max-width: 85%;
    margin-left: auto;
}
.bot-msg {
    background: #f8fafc;
    color: #0f172a;
    padding: 16px 18px;
    border-radius: 18px;
    margin: 10px 0;
    max-width: 85%;
}
.meta {
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
}
.header {
    text-align: center;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="header">
    <h1>💰 Financial Chatbot</h1>
    <p>Ask questions from your embedded financial documents</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INIT RAG (USING ENV VARIABLES)
# -----------------------------
@st.cache_resource
def init_rag():
    return FinancialRAGSystem(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("INDEX_NAME", "financial-docs"),
        mongo_uri=os.getenv("MONGO_URI")
    )

rag = init_rag()

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# CHAT HISTORY
# -----------------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='bot-msg'>{msg['content']}"
            f"<div class='meta'>Answered at {msg['time']}</div></div>",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# INPUT
# -----------------------------
user_input = st.chat_input("Ask about revenue, profit, growth, orders…")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.spinner("Analyzing documents..."):
        answer = rag.query(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%H:%M")
    })

    st.rerun()