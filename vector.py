from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    # Load all .docx and .pdf files from doc_folder
    loader_docx = DirectoryLoader(path="doc_folder", glob="*.docx", show_progress=True)
    loader_pdf = DirectoryLoader(path="doc_folder", glob="*.pdf", show_progress=True)
    raw_docs = loader_docx.load() + loader_pdf.load()
    
    # Split documents into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = splitter.split_documents(raw_docs)
    ids = [str(i) for i in range(len(documents))]

vector_store = Chroma(
    collection_name="resume_docs",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)