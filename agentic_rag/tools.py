"""4. Tools Component (Knowledge Retrieval, File Inspection & Computation).

Implements callable LangChain/LangGraph tools for the Agent:
1. Document Knowledge Base Retrieval (RAG vector search)
2. Local File Reading (PDF / text)
3. Mathematical Computation Tool
as described in Section 3 of Agent.md.
"""

import os
import math
from typing import List, Optional, Any
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf


class DocumentRetrieverManager:
    """Manages the in-memory/persisted Chroma vector store and document chunks."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.vector_store: Optional[Chroma] = None
        self.retriever: Optional[Any] = None
        self.ingested_files: List[str] = []

    def ingest_pdf(self, file_path: str) -> str:
        """Ingests a PDF file into the vector store."""
        if not os.path.exists(file_path):
            return f"Error: File not found at '{file_path}'"

        docs = PyPDFLoader(file_path=file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        file_name = os.path.basename(file_path)
        if file_name not in self.ingested_files:
            self.ingested_files.append(file_name)
        return f"Successfully ingested '{file_name}' ({len(chunks)} chunks)."

    def query(self, query_text: str, top_k: int = 3) -> str:
        """Queries the vector store for top relevant document chunks."""
        if self.vector_store is None or self.retriever is None:
            return "No documents have been ingested into the knowledge base yet. Please upload and ingest a document first."

        results = self.vector_store.similarity_search(query_text, k=top_k)
        if not results:
            return f"No matching passages found in the knowledge base for query: '{query_text}'."

        formatted = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"--- [Document Match #{i}] Source: {os.path.basename(source)} (Page {page}) ---\n{doc.page_content.strip()}")

        return "\n\n".join(formatted)

    def clear(self):
        """Clears the current vector store."""
        self.vector_store = None
        self.retriever = None
        self.ingested_files.clear()


# Global document retriever instance for tools
doc_manager = DocumentRetrieverManager()


@tool
def query_document_knowledge_base(query: str) -> str:
    """Search and retrieve relevant context from the ingested PDF/document knowledge base.

    Args:
        query: The semantic search question or keywords to look up in the documents.

    Returns:
        Relevant document snippets and citation details.
    """
    return doc_manager.query(query)


@tool
def read_local_file(file_path: str) -> str:
    """Reads a local PDF, text, or markdown file and returns its raw text contents.

    Args:
        file_path: The local filesystem path to the file.

    Returns:
        Extracted text contents of the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'"

    try:
        if file_path.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(file_path)
            pages = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt and txt.strip():
                    pages.append(f"--- Page {idx + 1} ---\n{txt.strip()}")
            return "\n\n".join(pages) if pages else f"PDF '{file_path}' has no readable text."
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def calculate_expression(expression: str) -> str:
    """Evaluates a mathematical expression or calculation safely.

    Args:
        expression: The mathematical expression to compute (e.g. 'sum([80, 90, 99, 70])/4' or 'math.sqrt(144)').

    Returns:
        The calculated result as a string.
    """
    try:
        allowed_names = {
            "math": math,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "len": len,
            "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error calculating expression '{expression}': {str(e)}"


def get_default_tools() -> List[Any]:
    """Returns the list of default callable tools for the agent."""
    return [query_document_knowledge_base, read_local_file, calculate_expression]
