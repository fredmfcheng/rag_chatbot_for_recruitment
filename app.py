import streamlit as st
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.model = self._initialize_model()
        self.chain = self._setup_chain()

    def _initialize_model(self) -> Optional[OllamaLLM]:
        try:
            return OllamaLLM(model="llama3.2")
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            return None

    def _setup_chain(self) -> Optional[ChatPromptTemplate]:
        if not self.model:
            return None
        
        template = """
        You are a HR expert in answering questions about hiring and recruiting.
        Here are some resumes: {resumes}
        Here is the question to answer: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.model

    def process_query(self, query: str) -> Tuple[bool, str]:
        try:
            docs = retriever.invoke(query)
            result = self.chain.invoke({"resumes": docs, "question": query})
            return True, str(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return False, f"Error: {e}"

def handle_file_upload(uploaded_files):
    if not uploaded_files:
        return

    try:
        doc_folder = Path("doc_folder")
        doc_folder.mkdir(exist_ok=True)

        uploaded_count = {"pdf": 0, "docx": 0}
        
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext not in ['pdf', 'docx']:
                st.sidebar.warning(f"Unsupported file type: {file_ext}")
                continue
                
            file_path = doc_folder / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            uploaded_count[file_ext] += 1
            
        if sum(uploaded_count.values()) > 0:
            st.sidebar.success(
                f"Uploaded {uploaded_count['pdf']} PDF(s) and "
                f"{uploaded_count['docx']} DOCX file(s)! "
                "Refreshing vector database..."
            )
            refresh_vector_db()
            
    except Exception as e:
        st.sidebar.error(f"Error processing files: {e}")
        logger.error(f"File processing error: {e}")

def refresh_vector_db(force: bool = False):
    try:
        with st.spinner("Refreshing vector database..."):
            cmd = [sys.executable, "vector.py"]
            if force:
                cmd.append("--force")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.sidebar.success("Vector database refreshed!")
    except subprocess.CalledProcessError as e:
        st.sidebar.error(f"Refresh failed: {e.stderr}")
        logger.error(f"Vector refresh error: {e}")

def main():
    st.set_page_config(page_title="Resume RAG Chatbot", page_icon="📄")
    st.title("Resume RAG Chatbot")

    chatbot = ChatBot()
    if not chatbot.model or not chatbot.chain:
        st.error("Failed to initialize the chatbot. Please check the logs.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Supports PDF and DOCX formats"
        )
        handle_file_upload(uploaded_files)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh DB"):
                refresh_vector_db()
        with col2:
            if st.button("Force Refresh"):
                refresh_vector_db(force=True)

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with the HR Bot")
    user_input = st.text_area(
        "Ask a question about the resumes:",
        height=100,
        placeholder="Type your question here..."
    )

    if st.button("Send", disabled=not user_input.strip()) and user_input:
        st.session_state.chat_history.append(("You", user_input))
        success, response = chatbot.process_query(user_input)
        st.session_state.chat_history.append(("Bot", response))

    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()