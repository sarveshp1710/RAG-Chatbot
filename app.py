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
import re # Import regular expressions for paragraph splitting

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
from sentence_transformers import SentenceTransformer, CrossEncoder
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
    
    # Load the embedding model (bi-encoder) for the first retrieval step
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    
    # Load the Cross-Encoder model for the second, more accurate re-ranking step
    cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')

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
    return embedding_model, cross_encoder_model, llm_model, qdrant_client

# --- DOCUMENT PROCESSING LOGIC ---

def process_documents(uploaded_files, qdrant_client, embedding_model):
    """Loads, chunks, embeds, and indexes the content of uploaded files."""
    if not uploaded_files:
        st.warning("Please upload some documents first.")
        return

    with st.spinner("Processing documents... This may take a while."):
        if qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME):
            qdrant_client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
        
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
            
            file_bytes = uploaded_file.read()
            
            if file_name.endswith(".pdf"):
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        # --- THE FIX IS HERE: Reverting to sentence-grouping chunker ---
                        sentences = nltk.sent_tokenize(text)
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
        
        st.session_state.created_chunks = all_chunks_text

        prefixed_chunks = [f"Represent this document for retrieval: {chunk}" for chunk in all_chunks_text]

        chunk_embeddings = embedding_model.encode(prefixed_chunks, show_progress_bar=True)
        
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
                        "chunk_id": chunk_id,
                        "page_number": meta.get("page_number", 1)
                    }
                )
                for chunk_id, vector, text_chunk, meta in zip(chunk_ids, chunk_embeddings, all_chunks_text, all_chunks_meta)
            ],
            wait=True
        )
    st.success("Documents processed and indexed successfully!")
    return True

# --- RAG AND CHAT LOGIC ---

def get_answer(query, embedding_model, cross_encoder_model, llm_model, qdrant_client):
    """Performs the entire RAG pipeline to get an answer for the user query."""
    start_time = time.time()

    query_vector = embedding_model.encode(query).tolist()
    
    # 1. Retrieve a larger number of initial candidates
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=10 
    )
    
    if not search_results:
        return "I could not find any information in the provided documents.", 0, [], 0

    # 2. Re-rank the results with the more powerful Cross-Encoder
    cross_inp = [[query, result.payload['content']] for result in search_results]
    cross_scores = cross_encoder_model.predict(cross_inp)

    # Combine results with their new, more accurate scores
    for i in range(len(search_results)):
        search_results[i].score = cross_scores[i]

    # Sort the results by the new cross-encoder score in descending order
    search_results.sort(key=lambda x: x.score, reverse=True)

    top_score = search_results[0].score
    st.session_state.scores = [res.score for res in search_results] # Display the new scores

    # 3. Verify with a high-confidence threshold
    relevance_threshold = 0.0
    if top_score < relevance_threshold:
        return "I could not find any relevant information in the provided documents.", time.time() - start_time, [], top_score
    
    # 4. Build context with the top 3 high-confidence chunks for synthesis
    context = ""
    sources = []
    for result in search_results[:3]: 
        if result.score >= relevance_threshold:
            context += f"Content: {result.payload['content']}\n---\n"
            sources.append({
                "file_name": result.payload['file_name'],
                "page_number": result.payload.get('page_number', 'N/A'),
                "chunk_id": result.payload['chunk_id']
            })
    
    if not context:
        return "I could not find any relevant information in the provided documents.", time.time() - start_time, [], top_score


    # --- THE FIX IS HERE ---
    # We are switching to a simpler, more direct "zero-shot" prompt.
    # This prompt is cleaner and explicitly asks for a brief answer.
    prompt_template = """
<|system|>
You are a helpful assistant. Your task is to answer the user's question briefly, based ONLY on the context provided.
- If the context contains the answer, provide a concise answer.
- If the context does NOT contain the answer, you MUST reply with the exact words: "I could not find the answer in the provided documents."
- Do not use any information from outside the context.</s>
<|user|>
CONTEXT:
{context}

QUESTION:
{query}</s>
<|assistant|>
"""
    
    final_prompt = prompt_template.format(context=context, query=query)
    
    response = llm_model(final_prompt, max_new_tokens=150, temperature=0.0) # Reduced max tokens for brevity
    
    end_time = time.time()
    duration = end_time - start_time
    
    return response.strip(), duration, sources, top_score

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Dynamic RAG Chatbot", layout="wide")
st.title("📄 Dynamic Document Chatbot (CPU Version)")

try:
    embedding_model, cross_encoder_model, llm_model, qdrant_client = load_models_and_clients()
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
        if 'scores' in st.session_state:
            del st.session_state['scores']
        if 'created_chunks' in st.session_state:
            del st.session_state['created_chunks']
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
        - **Execution Mode:** This app is running in **CPU-only mode** due to local hardware constraints preventing stable GPU initialization.
        - **CPU-Centric Model:** Using `TinyLlama-1.1B-Chat`, a fast quantized model, to ensure a responsive demo.
        - **Advanced RAG Pipeline:** Implemented a two-stage retrieval process with a Bi-Encoder for initial search and a more accurate Cross-Encoder for re-ranking to improve relevance detection.
        - **Relevance Thresholding:** A high-confidence threshold on the Cross-Encoder score ensures the LLM is not queried for irrelevant questions.
        """)
        
    if 'scores' in st.session_state:
        st.sidebar.markdown("### Top Re-Ranked Chunk Scores:")
        for i, score in enumerate(st.session_state.scores):
            st.sidebar.write(f"Chunk {i+1}: {score:.4f}")
            
    if 'created_chunks' in st.session_state:
        with st.expander("View Created Chunks for Debugging"):
            for i, chunk in enumerate(st.session_state.created_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.info(chunk)


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
                answer, duration, sources, score = get_answer(prompt, embedding_model, cross_encoder_model, llm_model, qdrant_client)
                
                if sources and "I could not find the answer" not in answer:
                    unique_sources = []
                    seen_sources = set()
                    for src in sources:
                        source_id = f"{src['file_name']}_p{src['page_number']}"
                        if source_id not in seen_sources:
                            unique_sources.append(src)
                            seen_sources.add(source_id)
                    
                    source_str = ", ".join([f"'{s['file_name']}' (page {s['page_number']})" for s in unique_sources])
                    
                    full_response = f"{answer}\n\n*Answered in {duration:.2f}s from {source_str} with a re-ranked score of {score:.4f}*"
                else:
                    full_response = f"{answer}\n\n*Answered in {duration:.2f}s with a re-ranked score of {score:.4f}*"

                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process your documents in the sidebar to begin.")
