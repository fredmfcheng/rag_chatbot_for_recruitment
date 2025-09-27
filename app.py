import streamlit as st
import os
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Set up model and prompt
model = OllamaLLM(model="llama3.2")
template = """
You are a HR expert in answering questions about hiring and recruiting.

Here are some resumes: {docs}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

st.title("Resume RAG Chatbot")

# File uploader
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload .pdf or .docx files", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("doc_folder", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("doc_folder", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.sidebar.success("File(s) uploaded! Please restart the backend to re-index documents.")

# Add this to your Streamlit sidebar in app.py
if st.sidebar.button("Refresh Vector Database"):
    os.system("python vector.py")
    st.sidebar.success("Vector database refreshed!")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.header("Chat with the HR Bot")

user_input = st.text_input("Ask a question about the resumes:")

if st.button("Send") and user_input:
    docs = retriever.invoke(user_input)
    result = chain.invoke({"docs": docs, "question": user_input})
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", str(result)))

for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")