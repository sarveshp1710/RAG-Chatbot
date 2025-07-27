# Embedding Model Configuration
# We are using a powerful embedding model for the best relevance detection.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# LLM Model Configuration
# THE FIX: We are reverting to TinyLlama. It is the fastest model that can
# still follow instructions reasonably well, making it the best choice for a responsive demo.
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
