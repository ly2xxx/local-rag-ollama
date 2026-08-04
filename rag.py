import joblib
from operator import itemgetter

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="glm-5.2:cloud", base_url="http://127.0.0.1:11434")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.memory = ConversationBufferMemory()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context and previous conversation history to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.",
                ),
                (
                    "human",
                    "Previous conversation:\n{history}\n\nQuestion: {question}\nContext: {context}",
                ),
            ]
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        modelPath = "all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=modelPath)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
            },
        )

        self.chain = (
            {
                "context": itemgetter("question") | self.retriever,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        # 1. Fetch conversation history from memory
        history_str = self.memory.load_memory_variables({})["history"]
        print("--- Conversation History ---")
        print(history_str)
        print("----------------------------")

        # 2. Invoke chain with question and history
        response = self.chain.invoke({"question": query, "history": history_str})

        # 3. Save question and answer to conversation memory
        self.memory.save_context({"question": query}, {"answer": response})

        return response

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.memory.clear()
