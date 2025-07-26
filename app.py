import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import fitz  # PyMuPDF
import docx
import uuid
import nltk
import io

# --- CACHE DIRECTORY SETUP (THE FIX) ---
# This is the most important step to ensure models are downloaded to your E: drive.
# We set the environment variable programmatically before any other imports.
CACHE_DIR = "E:/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# Create the directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


# Import configuration from our config.py file
from config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE
)

# --- MODEL AND CLIENT LOADING ---

@st.cache_resource
def load_models_and_clients():
    """Load all required models and database clients and cache them."""
    print("Loading models and clients for GPU...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=False
    )
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print("Models and clients loaded successfully.")
    return embedding_model, tokenizer, llm_model, qdrant_client

# --- DOCUMENT PROCESSING LOGIC ---

def process_documents(uploaded_files, qdrant_client, embedding_model):
    """Loads, chunks, embeds, and indexes the content of uploaded files."""
    if not uploaded_files:
        st.warning("Please upload some documents first.")
        return

    with st.spinner("Processing documents... This may take a while."):
        # Ensure the 'punkt' tokenizer is downloaded for sentence splitting
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

        # Clear the old collection in Qdrant to start fresh
        qdrant_client.recreate_collection(
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

            # Chunking logic using NLTK
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

def get_answer(query, embedding_model, tokenizer, llm_model, qdrant_client):
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

    prompt_template = """
INSTRUCTION: You are a helpful assistant. Your task is to answer the user's question based ONLY on the context provided below.
- If the context contains the answer, provide the answer.
- After the answer, on a new line, cite the exact source using the format: SOURCE: [filename], Chunk ID: [chunk_id].
- If you cannot find the answer, respond with "I could not find the answer in the provided documents."
- Do not use any external knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    final_prompt = prompt_template.format(context=context, query=query)
    inputs = tokenizer(final_prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=512)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_index = response_text.find("ANSWER:")
    if answer_start_index != -1:
        return response_text[answer_start_index + len("ANSWER:"):].strip()
    return "Could not parse the model's response."

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Dynamic RAG Chatbot", layout="wide")
st.title("📄 Dynamic Document Chatbot")

# Load models and clients
try:
    embedding_model, tokenizer, llm_model, qdrant_client = load_models_and_clients()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# Sidebar for file upload
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

# Main chat interface
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
                full_response = get_answer(prompt, embedding_model, tokenizer, llm_model, qdrant_client)
                message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Please upload and process your documents in the sidebar to begin.")
