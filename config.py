# Embedding Model Configuration
# This is a small, fast, and high-quality model for creating vector embeddings.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM Model Configuration
# We are using the powerful Mistral-7B model in its GPTQ-quantized version.
# This is designed to run efficiently on a GPU with ~16GB of VRAM.
LLM_MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

# Vector Database Configuration
# Settings for our local Qdrant instance running in Docker.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "hackathon_collection"

# Chunking Configuration
# The target size for our text chunks in characters.
CHUNK_SIZE = 1000