import streamlit as st
import google.generativeai as genai
import PyPDF2
import re
import os
from dotenv import load_dotenv
import csv
from datetime import datetime
import pandas as pd


# ---- SETUP GEMINI ----
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")

# ---- SESSION STATE INIT ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_chunks" not in st.session_state:
    st.session_state.file_chunks = []

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "tone" not in st.session_state:
    st.session_state.tone = "Friendly"  # Default tone

# ---- UTILS ----
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def chunk_text(text, max_len=500):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += sentence + " "
        else:
            chunks.append(current.strip())
            current = sentence + " "
    if current:
        chunks.append(current.strip())
    return chunks

def search_relevant_chunks(query, chunks, top_k=3):
    query_words = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words & chunk_words)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [chunk for score, chunk in scored if score > 0][:top_k]

def format_chat_history(history):
    formatted = ""
    for speaker, msg in history:
        if speaker == "user":
            formatted += f"User: {msg}\n"
        else:
            formatted += f"Bot: {msg}\n"
    return formatted.strip()

# ---- APP UI ----
st.set_page_config(page_title="🤖💬📚Conversational Chatbot", layout="centered")

# Sidebar
with st.sidebar:
    st.header("Settings")
    tone_options = ["Friendly", "Formal", "Professional", "Casual", "Empathetic", "Humorous"]
    tone = st.selectbox(
        "Choose the bot's tone:",
        tone_options,
        index=tone_options.index(st.session_state.tone),
        key="tone",
    )
    st.markdown("---")
    st.header("About Me")
    st.markdown("""
    **Conversational Chatbot**  
    Powered by [Google Gemini] and Streamlit.  
    Upload your documents and chat with an AI that answers in your chosen tone!  
    Developed by *Maria Hameed* 🚀
    """)

    st.markdown("---")
    st.header("🗣️ Feedback")

    with st.form("feedback_form"):
        feedback_text = st.text_area("Share your thoughts or suggestions:")
        submit_feedback = st.form_submit_button("Submit")

        if submit_feedback and feedback_text.strip():
            feedback_file = "feedback.csv"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check if file exists to write headers
            file_exists = os.path.isfile(feedback_file)

            with open(feedback_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Feedback"])
                writer.writerow([timestamp, feedback_text.strip()])

            st.success("✅ Thank you! Your feedback has been recorded.")
# Main content
st.title("🤖💬📚Conversational Chatbot with File Support")

uploaded_file = st.file_uploader("Upload a .pdf or .txt file (optional)", type=["pdf", "txt"])
if uploaded_file and not st.session_state.file_uploaded:
    with st.spinner("Processing file..."):
        raw_text = extract_text_from_file(uploaded_file)
        chunks = chunk_text(raw_text)
        st.session_state.file_chunks = chunks
        st.session_state.file_uploaded = True
    st.success("✅ File uploaded and processed!")

    if st.button("🔄 Reset Uploaded File"):
        st.session_state.file_uploaded = False
        st.session_state.file_chunks = []
        st.success("File upload reset. You can now upload a new file.")

user_input = st.chat_input("Ask something...")

# Display chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message("user" if speaker == "user" else "assistant"):
        st.markdown(msg)

# Chat processing
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            tone_instruction = {
                "Friendly": "Answer in a warm, conversational, and helpful tone.",
                "Formal": "Answer in a professional, concise, and respectful manner.",
                "Professional": "Answer with clarity, expertise, and a business-oriented tone.",
                "Casual": "Answer informally and naturally, using relaxed language.",
                "Empathetic": "Answer with compassion and understanding, acknowledging emotions.",
                "Humorous": "Answer with light humor and wit, while staying accurate and respectful."
            }[tone]

            doc_context = ""
            if st.session_state.file_chunks:
                relevant_chunks = search_relevant_chunks(user_input, st.session_state.file_chunks)
                doc_context = "\n\n".join(relevant_chunks)

            prompt = f"""{tone_instruction}

Refer to this document content if relevant:
{doc_context}

Conversation so far:
{format_chat_history(st.session_state.chat_history)}

User: {user_input}
Bot:""".strip()

            try:
                response = model.generate_content(prompt)
                reply = response.text.strip()
            except Exception as e:
                reply = "❌ Error getting response from Gemini."

            st.markdown(reply)
            st.session_state.chat_history.append(("assistant", reply))

            
# Summarize chat button
if st.button("📄 Summarize This Chat"):
    with st.spinner("Summarizing conversation..."):
        summary_prompt = f"""Summarize the following conversation between a user and a bot in bullet points:

{format_chat_history(st.session_state.chat_history)}
"""
        try:
            summary_response = model.generate_content(summary_prompt)
            summary = summary_response.text.strip()
        except Exception as e:
            summary = "❌ Error summarizing conversation."

    st.markdown("### 📝 Summary")
    st.markdown(summary)

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.file_chunks = []
    st.session_state.file_uploaded = False

