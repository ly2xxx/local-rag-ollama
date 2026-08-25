"""Configuration settings loaded from environment variables / .env file."""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

# LLM & Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm-5.2:cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Redis Checkpointer Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
