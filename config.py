# Embedding Model Configuration
# This is a small, fast, and high-quality model for creating vector embeddings.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Model Configuration
# THE FIX: We are prioritizing speed for a smooth demo experience on the CPU.
# TinyLlama is the fastest viable option.
LLM_MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


# Vector Database Configuration
# Settings for our local Qdrant instance running in Docker.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "hackathon_collection"

# Chunking Configuration
# The target size for our text chunks in characters.
CHUNK_SIZE = 1000
