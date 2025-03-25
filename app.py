import streamlit as st
import ollama as ol
from voice import record_voice
import subprocess
import requests
import os
import random
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from PyPDF2 import PdfReader
from docx import Document


# Function to check if Ollama is running
def is_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


# Function to select the LLM model
def llm_selector():
    try:
        response = ol.list()
        all_models = [m.get('name', 'UNKNOWN') for m in response.get('models', [])]
        chat_models = [m for m in all_models if "nomic-embed" not in m]

        if not chat_models:
            st.sidebar.error("No compatible chat models found. Try `ollama pull mistral:latest`.")
            return None
    except Exception as e:
        st.sidebar.error(f"Error fetching models: {e}")
        return None

    return st.sidebar.selectbox("Select LLM", chat_models, index=0)


# Function to extract text from uploaded documents
def extract_text_from_document(file_path):
    text = ""
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == ".pdf":
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    elif file_extension.lower() == ".docx":
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_extension.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    return text.strip()


# Function to process uploaded documents
def process_uploaded_document(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join("uploaded_docs", uploaded_file.name)
        os.makedirs("uploaded_docs", exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text_from_document(file_path)
        st.sidebar.success(f"Document Processed: {uploaded_file.name}")

        return extracted_text
    return None


# Function to calculate credibility score
def calculate_credibility_score(source_name, content):
    score = 0

    # 1. Source Reliability (0-40 points)
    source_name_lower = source_name.lower()
    top_tier_sources = {"bbc": 40, "reuters": 40, "nytimes": 35}
    mid_tier_sources = {"cnn": 25, "guardian": 30}
    if any(source in source_name_lower for source in top_tier_sources):
        score += top_tier_sources.get(next(s for s in top_tier_sources if s in source_name_lower), 40)
    elif any(source in source_name_lower for source in mid_tier_sources):
        score += mid_tier_sources.get(next(s for s in mid_tier_sources if s in source_name_lower), 25)
    else:
        score += 10

    # 2. Content Quality (0-40 points)
    content_lower = content.lower()
    words = content_lower.split()
    word_count = len(words)

    emotional_words = ["shocking", "unbelievable", "outrageous", "scandal", "disaster"]
    sensationalism_penalty = -sum(5 for word in emotional_words if word in content_lower)
    sensationalism_penalty = max(-20, sensationalism_penalty)

    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    max_repetition = max(word_freq.values()) if word_freq else 0
    repetition_penalty = -min(10, (max_repetition - 3) * 2) if max_repetition > 3 else 0

    length_bonus = min(word_count // 50, 30)

    content_score = 20 + length_bonus + sensationalism_penalty + repetition_penalty
    content_score = max(0, min(40, content_score))
    score += content_score

    # 3. Historical Accuracy of Publisher (0-20 points)
    publisher_accuracy = {"bbc": 20, "reuters": 20, "nytimes": 18, "cnn": 15, "guardian": 16}
    publisher = next((p for p in publisher_accuracy if p in source_name_lower), None)
    score += publisher_accuracy.get(publisher, 5)

    score = max(0, min(100, score))
    return score


# Main app function
def main():
    CHROMA_PATH = "chroma"
    PROMPT_TEMPLATE = """
    Answer the question based on the below context if you think that the context is relevant, if you can directly answer the question you may do it.
    You are a helping assembly line assistant of Flexible Production System (FPS), answer the question based on the following context:

    {context}
    ---
    question: {question}
    """

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = llm_selector()

    if not model:
        st.error("No valid chat model selected. Please select or pull a model.")
        return

    # Initialize session state
    if "credibility_score" not in st.session_state:
        st.session_state.credibility_score = None
    if "flagged_reports" not in st.session_state:
        st.session_state.flagged_reports = []
    if "show_admin" not in st.session_state:
        st.session_state.show_admin = False

    # Add Admin button at top right
    col1, col2 = st.columns([4, 1])  # Split layout: 80% left, 20% right
    with col1:
        st.title("🔎 AI powered Fake News Detection")
    with col2:
        if st.button("Admin", key="admin_button"):
            st.session_state.show_admin = not st.session_state.show_admin  # Toggle admin view

    # Custom CSS to ensure button is top-right
    st.markdown(
        """
        <style>
        div.stButton > button[kind="primary"] {
            position: relative;
            top: 0px;
            right: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Admin view (display only when toggled)
    if st.session_state.show_admin:
        st.subheader("Admin Dashboard - Flagged News Updates")
        if st.session_state.flagged_reports:
            for i, report in enumerate(st.session_state.flagged_reports):
                random_user_id = random.randint(1000, 9999)  # Random 4-digit user ID
                st.write(f"**Update {i + 1}:**")
                st.write(f"- User ID: {random_user_id}")
                st.write(f"- Source: {report['file_name']}")
                st.write(f"- Preview: {report['content_preview']}")
                st.write(f"- Credibility Score: {report['credibility_score']}/100")
                st.write(f"- Status: {report['status']}")
        else:
            st.write("No flagged news updates yet.")
        st.markdown("---")  # Separator between admin and main UI

    # Add document upload UI (optional context)
    st.sidebar.header("📄 Upload Documents (Optional)")
    uploaded_file = st.sidebar.file_uploader("Upload a document for context", type=["pdf", "txt", "docx"])

    document_text = ""
    if uploaded_file:
        document_text = process_uploaded_document(uploaded_file)
        if document_text:
            credibility_score = calculate_credibility_score(uploaded_file.name, document_text)
            st.session_state.credibility_score = credibility_score

    # Display document credibility score (if uploaded)
    st.subheader("Document Credibility (If Uploaded)")
    if st.session_state.credibility_score is not None:
        st.write(f"**Credibility Score:** {st.session_state.credibility_score}/100")
    else:
        st.write("No document uploaded yet.")

    # Chat section with flagging feature
    st.subheader("Check Suspicious News")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "actual_history" not in st.session_state:
        st.session_state.actual_history = {}
    if model not in st.session_state.actual_history:
        st.session_state.actual_history[model] = []
    if model not in st.session_state.chat_history:
        st.session_state.chat_history[model] = []

    chat_history = st.session_state.chat_history[model]
    actual_history = st.session_state.actual_history[model]

    query_text = st.text_input("Enter news text to check:")
    if query_text:
        # Calculate credibility score for the query text
        query_credibility_score = calculate_credibility_score("user_input", query_text)

        # Display credibility score for the query
        st.write(f"**Credibility Score for Entered Text:** {query_credibility_score}/100")

        # Flag as Suspicious button
        if st.button("Flag as Suspicious", key="flag_query"):
            report = {
                "file_name": "USer based reference source",
                "content_preview": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "credibility_score": query_credibility_score,
                "status": "Verification send to application organization"
            }
            st.session_state.flagged_reports.append(report)
            st.success(f"Text flagged for review!")

        # Chat with LLM
        question = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=document_text, question=query_text)
        chat_history.append({"role": "user", "content": query_text})
        actual_history.append({"role": "user", "content": question})

        try:
            response = ol.chat(model=model, messages=actual_history)
            answer = response['message']['content']
            chat_history.append({"role": "assistant", "content": answer})
            actual_history.append({"role": "assistant", "content": answer})
            st.write("**Assistant Response:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error during chat: {e}")

    # Display flagged reports
    st.subheader("Flagged Reports")
    if st.session_state.flagged_reports:
        for i, report in enumerate(st.session_state.flagged_reports):
            st.write(f"**Report {i + 1}:**")
            st.write(f"- Source: {report['file_name']}")
            st.write(f"- Preview: {report['content_preview']}")
            st.write(f"- Credibility Score: {report['credibility_score']}/100")
            st.write(f"- Status: {report['status']}")
            if st.button("Verify", key=f"verify_{i}"):
                report["status"] = "Verified - Under Review"
                st.success(f"Report for '{report['file_name']}' marked as under review!")
    else:
        st.write("No reports flagged yet.")


# Run the app
if __name__ == "__main__":
    st.set_page_config(page_title="🎙️ Voice Bot", layout="wide")
    st.sidebar.title("Speak with the Assistant")

    if not is_ollama_running():
        subprocess.Popen(["ollama", "serve"])
    main()