"""Configuration settings loaded from environment variables / .env file."""

import os
import tempfile
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# Repository root (this file lives in <root>/agentic_rag/config.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LLM & Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm-5.2:cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Redis Checkpointer Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Vector Store Configuration
# Knowledge bases are persisted to disk so they survive a restart, matching the
# durability of the Redis conversation checkpointer.
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", os.path.join(PROJECT_ROOT, ".chroma_agentic")
)

# Namespace used when no session / thread scope is supplied. Each namespace maps
# to its own Chroma collection so concurrent sessions cannot read one another's
# documents.
DEFAULT_NAMESPACE = os.getenv("AGENTIC_RAG_NAMESPACE", "default_session")

# Directories the `read_local_file` tool is permitted to read from,
# separated by the OS path separator (';' on Windows, ':' elsewhere).
# Defaults to the project root plus the system temp dir, which is where
# Streamlit writes uploaded files.
FILE_TOOL_ROOTS = os.getenv(
    "FILE_TOOL_ROOTS", os.pathsep.join([PROJECT_ROOT, tempfile.gettempdir()])
)
