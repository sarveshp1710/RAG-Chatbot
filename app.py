import os
import streamlit as st
import torch
import fitz  # PyMuPDF
import docx
import uuid
import nltk
import io

# --- GPU AND CACHE SETUP ---
# THE FIX: Force the app to only see and use GPU 1 (NVIDIA GeForce MX330).
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set the cache directory to ensure models are downloaded to your E: drive.
CACHE_DIR = "E:/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

# We now handle the NLTK 'punkt' download robustly within the app itself.
# 1. Define a local path for NLTK data.
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
# 2. Check if the 'punkt' tokenizer is available in our local path.
punkt_tokenizer_path = os.path.join(nltk_data_path, "tokenizers", "punkt")
# 3. If not, download it to our local path.
if not os.path.exists(punkt_tokenizer_path):
    print(f"Downloading NLTK 'punkt' model to {nltk_data_path}...")
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.download('punkt', download_dir=nltk_data_path)
# 4. Add our local path to NLTK's search paths to ensure it's found.
if nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_path)


# --- ML/AI Library Imports ---
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from ctransformers import AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Import configuration from our config.py file
from config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    LLM_MODEL_FILE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE
)

# --- MODEL AND CLIENT LOADING ---

@st.cache_resource
def load_models_and_clients():
    """Load all required models and database clients and cache them."""
    print("Loading models and clients for GPU (TinyLlama)...")
    
    # --- THE FIX IS HERE ---
    # We are switching back to 'cuda' for the embedding model.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    
    # Download the GGUF model file from Hugging Face Hub if it doesn't exist
    model_path = hf_hub_download(repo_id=LLM_MODEL_NAME, filename=LLM_MODEL_FILE)

    # Load the GGUF model using ctransformers, configured for GPU offloading.
    # Because TinyLlama is small, we can try to offload more layers.
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama",
        gpu_layers=25,  # Offload 25 layers to the GPU. This is a good starting point.
        context_length=4096 
    )
    
    # We increase the timeout to 60 seconds to prevent timeout errors on slower systems.
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    
    print("Models and clients loaded successfully.")
    return embedding_model, llm_model, qdrant_client

# --- DOCUMENT PROCESSING LOGIC ---

def process_documents(uploaded_files, qdrant_client, embedding_model):
    """Loads, chunks, embeds, and indexes the content of uploaded files."""
    if not uploaded_files:
        st.warning("Please upload some documents first.")
        return

    with st.spinner("Processing documents... This may take a while."):
        # We now use the modern, non-deprecated way to ensure a clean collection.
        # First, check if the collection exists.
        if qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
            # If it exists, delete it.
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        
        # Now, create a new, empty collection.
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            ),
        )

        all_chunks_text = []
        all_chunks_meta = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"Processing {file_name}...")
            
            # Read file content into memory
            file_bytes = uploaded_file.read()
            text = ""
            if file_name.endswith(".pdf"):
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "".join(page.get_text() for page in doc)
            elif file_name.endswith(".docx"):
                with io.BytesIO(file_bytes) as doc_stream:
                    doc = docx.Document(doc_stream)
                    text = "\n".join([para.text for para in doc.paragraphs])

            # We use the standard, high-level sent_tokenize function.
            # Because we've set the path correctly, it will find the local data.
            sentences = nltk.sent_tokenize(text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        all_chunks_text.append(current_chunk.strip())
                        all_chunks_meta.append({"file_name": file_name})
                    current_chunk = sentence
            if current_chunk:
                all_chunks_text.append(current_chunk.strip())
                all_chunks_meta.append({"file_name": file_name})

        st.write(f"Created {len(all_chunks_text)} chunks. Now embedding...")

        # Embed all chunks at once
        chunk_embeddings = embedding_model.encode(all_chunks_text, show_progress_bar=True)
        
        # Upload to Qdrant
        chunk_ids = [str(uuid.uuid4()) for _ in all_chunks_text]
        qdrant_client.upload_points(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=vector.tolist(),
                    payload={
                        "file_name": meta["file_name"],
                        "content": text_chunk,
                        "chunk_id": chunk_id
                    }
                )
                for chunk_id, vector, text_chunk, meta in zip(chunk_ids, chunk_embeddings, all_chunks_text, all_chunks_meta)
            ],
            wait=True
        )
    st.success("Documents processed and indexed successfully!")
    return True

# --- RAG AND CHAT LOGIC ---

def get_answer(query, embedding_model, llm_model, qdrant_client):
    """Performs the RAG pipeline to get an answer for the user query."""
    query_vector = embedding_model.encode(query).tolist()
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    
    context = ""
    for result in search_results:
        context += f"Source: {result.payload['file_name']}, Chunk ID: {result.payload['chunk_id']}\n"
        context += f"Content: {result.payload['content']}\n---\n"
        
    if not context:
        return "Sorry, I could not find any relevant information in the documents."

    # This prompt template is specifically for TinyLlama Instruct models
    prompt_template = """
<|system|>
You are a helpful assistant. Your task is to answer the user's question based ONLY on the context provided below.
- If the context contains the answer, provide the answer.
- After the answer, on a new line, cite the exact source using the format: SOURCE: [filename], Chunk ID: [chunk_id].
- If you cannot find the answer, respond with "I could not find the answer in the provided documents."
- Do not use any external knowledge.</s>
<|user|>
CONTEXT:
{context}

QUESTION:
{query}</s>
<|assistant|>
"""
    
    final_prompt = prompt_template.format(context=context, query=query)
    
    # Generate the answer using the GGUF model
    response = llm_model(final_prompt, max_new_tokens=512, temperature=0.1)
    
    return response.strip()

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Dynamic RAG Chatbot", layout="wide")
st.title("📄 Dynamic Document Chatbot (GPU Version)")

try:
    embedding_model, llm_model, qdrant_client = load_models_and_clients()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF or Word documents",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    if st.button("Process Documents") and uploaded_files:
        st.session_state.documents_processed = process_documents(
            uploaded_files, qdrant_client, embedding_model
        )

st.header("2. Chat with Your Documents")

if st.session_state.get("documents_processed", False):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Finding an answer..."):
                full_response = get_answer(prompt, embedding_model, llm_model, qdrant_client)
                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process your documents in the sidebar to begin.")
