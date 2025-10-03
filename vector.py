from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass
import os
import logging
import argparse
import shutil
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VectorConfig:
    """Configuration settings for vector store"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "mxbai-embed-large"
    doc_folder: Path = Path("doc_folder")
    db_location: Path = Path("./chrome_langchain_db")
    retriever_k: int = 5

class DocumentProcessor:
    def __init__(self, config: VectorConfig = VectorConfig()):
        self.config = config
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self) -> OllamaEmbeddings:
        try:
            return OllamaEmbeddings(model=self.config.embedding_model)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def load_documents(self) -> List[Document]:
        """Load and process documents from the doc folder"""
        if not self.config.doc_folder.exists():
            logger.warning(f"Document folder {self.config.doc_folder} not found")
            return []

        try:
            # Enhanced document loading with better PDF support
            loaders = []
            
            # PDF loader with PyPDF for reliable text extraction
            pdf_loader = DirectoryLoader(
                path=str(self.config.doc_folder),
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            loaders.append(pdf_loader)
            
            # DOCX loader
            docx_loader = DirectoryLoader(
                path=str(self.config.doc_folder),
                glob="*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=True
            )
            loaders.append(docx_loader)

            raw_docs = []
            for loader in loaders:
                try:
                    docs = loader.load()
                    logger.info(f"Loaded {len(docs)} documents with {loader.__class__.__name__}")
                    raw_docs.extend(docs)
                except Exception as e:
                    logger.error(f"Error with loader {loader.__class__.__name__}: {e}")

            if not raw_docs:
                logger.warning("No documents found")
                return []

            # Enhanced document splitting for better handling of PDFs
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )
            documents = [d for d in splitter.split_documents(raw_docs) if d.page_content.strip()]
            
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source_type"] = "pdf" if doc.metadata.get("source", "").endswith(".pdf") else "docx"
            
            return documents

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return []

    def setup_vector_store(self, documents: Optional[List[Document]] = None, force_refresh: bool = False) -> Chroma:
        """Initialize or update the vector store"""
        try:
            # Remove existing DB if force refresh
            if force_refresh and self.config.db_location.exists():
                shutil.rmtree(self.config.db_location)
                logger.info("Removed existing vector store for refresh")

            vector_store = Chroma(
                collection_name="resume_docs",
                persist_directory=str(self.config.db_location),
                embedding_function=self.embeddings
            )

            if documents:
                ids = []
                for i, doc in enumerate(documents):
                    file_hash = hashlib.md5(doc.metadata["source"].encode()).hexdigest()[:8]
                    ids.append(f"{file_hash}_{i}")
                vector_store.add_documents(documents=documents, ids=ids)
                logger.info(f"Added {len(documents)} documents to vector store")
                logger.info(f"Total documents in store: {vector_store._collection.count()}")

            return vector_store

        except Exception as e:
            logger.error(f"Vector store error: {e}")
            raise

def main(force_refresh: bool = False) -> Chroma:
    """Initialize the RAG system"""
    config = VectorConfig()
    processor = DocumentProcessor(config)
    
    try:
        documents = processor.load_documents()
        if documents:
            logger.info(f"Found {len(documents)} documents to process")
            vector_store = processor.setup_vector_store(documents, force_refresh)
        else:
            logger.warning("No documents found to process")
            vector_store = processor.setup_vector_store(force_refresh=force_refresh)

        return vector_store.as_retriever(
            search_kwargs={"k": config.retriever_k}
        )

    except Exception as e:
        logger.error(f"System initialization error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup vector store for resumes")
    parser.add_argument("--force", action="store_true", help="Force refresh by deleting existing DB")
    args = parser.parse_args()
    main(force_refresh=args.force)
else:
    # For imports in app.py
    retriever = main()