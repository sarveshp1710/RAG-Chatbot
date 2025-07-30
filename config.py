# Embedding Model Configuration
# We are using a powerful embedding model for the best relevance detection.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# LLM Model Configuration
# We are using TinyLlama for the best balance of speed and rule-following for the demo.
LLM_MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


# Vector Database Configuration
# Settings for our local Qdrant instance running in Docker.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "hackathon_collection"

# Chunking Configuration
# Reverting to a larger chunk size that works better with the sentence-grouping method.
CHUNK_SIZE = 350
