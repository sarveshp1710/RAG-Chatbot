import os
import streamlit as st
import torch
import fitz  # PyMuPDF
import docx
import uuid
import nltk
import io
import psutil # Import psutil for system monitoring
import time # Import time to measure response duration

# --- CACHE AND NLTK SETUP ---
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
    print("Loading models and clients for CPU...")
    
    # Load the embedding model onto the CPU
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    
    # Download the GGUF model file from Hugging Face Hub if it doesn't exist
    model_path = hf_hub_download(repo_id=LLM_MODEL_NAME, filename=LLM_MODEL_FILE)

    # Load the GGUF model using ctransformers, configured to run on the CPU.
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="llama", # Set model type for TinyLlama
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
                    for page_num, page in enumerate(doc):
                        # We now store page number in the metadata for each chunk
                        sentences = nltk.sent_tokenize(page.get_text())
                        current_chunk = ""
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                                current_chunk += " " + sentence
                            else:
                                if current_chunk:
                                    all_chunks_text.append(current_chunk.strip())
                                    all_chunks_meta.append({"file_name": file_name, "page_number": page_num + 1})
                                current_chunk = sentence
                        if current_chunk:
                            all_chunks_text.append(current_chunk.strip())
                            all_chunks_meta.append({"file_name": file_name, "page_number": page_num + 1})
            elif file_name.endswith(".docx"):
                with io.BytesIO(file_bytes) as doc_stream:
                    doc = docx.Document(doc_stream)
                    full_text = "\n".join([para.text for para in doc.paragraphs])
                    sentences = nltk.sent_tokenize(full_text)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                            current_chunk += " " + sentence
                        else:
                            if current_chunk:
                                all_chunks_text.append(current_chunk.strip())
                                all_chunks_meta.append({"file_name": file_name, "page_number": 1})
                            current_chunk = sentence
                    if current_chunk:
                        all_chunks_text.append(current_chunk.strip())
                        all_chunks_meta.append({"file_name": file_name, "page_number": 1})


        st.write(f"Created {len(all_chunks_text)} chunks. Now embedding...")

        # The BGE embedding model performs best when you add a specific instruction
        # to the text before embedding it. This significantly improves relevance detection.
        prefixed_chunks = [f"Represent this document for retrieval: {chunk}" for chunk in all_chunks_text]

        # Embed all chunks at once
        chunk_embeddings = embedding_model.encode(prefixed_chunks, show_progress_bar=True)
        
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
                        "content": text_chunk, # We store the original, non-prefixed text
                        "chunk_id": chunk_id,
                        "page_number": meta.get("page_number", 1) # Store page number
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
    start_time = time.time() # Start the timer

    # Note: For the BGE model, you do NOT add a prefix to the user's query.
    # This is called "asymmetric" search and is the recommended approach.
    query_vector = embedding_model.encode(query).tolist()
    
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=3
    )
    
    if search_results:
        st.session_state.scores = [result.score for result in search_results]
    else:
        return "I could not find any relevant information in the provided documents.", 0, []

    # With the better embedding model, we can use a more reliable and stricter threshold.
    relevance_threshold = 0.4
    if search_results[0].score < relevance_threshold:
        return "I could not find any relevant information in the provided documents.", time.time() - start_time, []

    
    context = ""
    sources = []
    for result in search_results:
        if result.score >= relevance_threshold:
            context += f"Content: {result.payload['content']}\n---\n"
            sources.append({
                "file_name": result.payload['file_name'],
                "page_number": result.payload.get('page_number', 'N/A'),
                "chunk_id": result.payload['chunk_id']
            })
        
    if not context:
        return "I could not find any relevant information in the provided documents.", time.time() - start_time, []

    # A much stronger, more explicit prompt to force the model to obey.
    # This prompt format is specific to TinyLlama Chat.
    prompt_template = """
<|system|>
You are a specialized AI assistant. You must follow these instructions exactly.
Your task is to answer a question based only on the provided context.
If the context contains the answer, you will state the answer and its source.
If the context does NOT contain the answer, you MUST reply with the exact phrase: "I could not find the answer in the provided documents."

<|user|>
CONTEXT:
{context}

QUESTION:
{query}</s>
<|assistant|>
"""
    
    final_prompt = prompt_template.format(context=context, query=query)
    
    # We set temperature to 0.0 to make the model more deterministic and less creative.
    response = llm_model(final_prompt, max_new_tokens=512, temperature=0.0)
    
    end_time = time.time() # End the timer
    duration = end_time - start_time
    
    return response.strip(), duration, sources

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Dynamic RAG Chatbot", layout="wide")
st.title("📄 Dynamic Document Chatbot (CPU Version)")

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
        # Clear previous scores when processing new docs
        if 'scores' in st.session_state:
            del st.session_state['scores']
        st.session_state.documents_processed = process_documents(
            uploaded_files, qdrant_client, embedding_model
        )
    
    st.header("2. System Analysis")
    with st.expander("Memory & CPU Usage", expanded=True):
        mem = psutil.virtual_memory()
        total_mem_gb = mem.total / (1024**3)
        used_mem_gb = mem.used / (1024**3)
        mem_percent = mem.percent
        
        cpu_percent = psutil.cpu_percent()
        
        st.markdown(f"**Total RAM:** {total_mem_gb:.2f} GB")
        st.markdown(f"**Used RAM:** {used_mem_gb:.2f} GB")
        st.progress(mem_percent / 100)
        
        st.markdown(f"**CPU Usage:** {cpu_percent}%")
        st.progress(cpu_percent / 100)

        st.markdown("---")
        st.subheader("Optimization Strategies Implemented")
        st.markdown("""
        - **Execution Mode:** This app is running in **CPU-only mode** due to local hardware constraints preventing stable GPU initialization. On a target system like a Tesla T4, the `gpu_layers` parameter would be set to offload computation for a >100x speed increase.
        - **CPU-Centric Model:** Switched back to `TinyLlama-1.1B-GGUF`, a quantized model optimized for CPU inference, to ensure a responsive demo.
        - **Efficient Model Caching:** Implemented `@st.cache_resource` to load the AI models into memory only once per session, preventing slow reloads on every interaction.
        - **Local Caching:** All models are downloaded to a local cache on the E: drive, avoiding re-downloads and saving internet bandwidth.
        - **Relevance Thresholding:** A programmatic check ensures the LLM is only queried if the retrieved documents have a high relevance score, saving computational resources on irrelevant questions.
        """)
        
    if 'scores' in st.session_state:
        st.sidebar.markdown("### Top Retrieved Chunk Scores:")
        for i, score in enumerate(st.session_state.scores):
            st.sidebar.write(f"Chunk {i+1}: {score:.4f}")


st.header("3. Chat with Your Documents")

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
            with st.spinner("Finding an answer... (This may be slow on CPU)"):
                answer, duration, sources = get_answer(prompt, embedding_model, llm_model, qdrant_client)
                
                # We now construct the final response string with the time and sources.
                if sources and "I could not find the answer" not in answer:
                    # Create a unique list of sources to avoid repetition
                    unique_sources = []
                    seen_sources = set()
                    for src in sources:
                        source_id = f"{src['file_name']}_p{src['page_number']}"
                        if source_id not in seen_sources:
                            unique_sources.append(src)
                            seen_sources.add(source_id)
                    
                    source_str = ", ".join([f"'{s['file_name']}' (page {s['page_number']})" for s in unique_sources])
                    
                    full_response = f"{answer}\n\n*Answered in {duration:.2f} seconds from {source_str}*"
                else:
                    full_response = f"{answer}\n\n*Answered in {duration:.2f} seconds*"

                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process your documents in the sidebar to begin.")

# The continuous update loop has been removed.
