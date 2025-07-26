import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Import configuration from our config.py file
from config import (
    EMBEDDING_MODEL_NAME,
    LLM_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME
)

# --- MODEL AND CLIENT LOADING ---

# The @st.cache_resource decorator ensures this function runs only once,
# caching the models in memory for performance.
@st.cache_resource
def load_models_and_clients():
    """
    Load all required models and database clients and cache them.
    This function will only run once, on the first run of the app.
    """
    print("Loading models and clients for GPU...")
    
    # Load the embedding model from Sentence Transformers, placing it on the GPU
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cuda')
    
    # Load the LLM tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto", # Automatically select the device (GPU)
        torch_dtype=torch.float16, # Use float16 for memory efficiency
        trust_remote_code=False # Set to False for security unless required by model
    )
    
    # Initialize the client to connect to our Qdrant database
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print("Models and clients loaded successfully.")
    return embedding_model, tokenizer, llm_model, qdrant_client

# --- CORE RAG LOGIC ---

def get_answer(query, embedding_model, tokenizer, llm_model, qdrant_client):
    """
    Performs the entire RAG pipeline to get an answer for the user query.
    """
    # 1. Embed the user's query into a vector
    query_vector = embedding_model.encode(query).tolist()
    
    # 2. Search Qdrant for relevant context (the "Retrieval" part)
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=3  # Retrieve the top 3 most relevant chunks
    )
    
    # 3. Format the retrieved context for the LLM
    context = ""
    for result in search_results:
        # Include the source metadata in the context for the LLM to see
        context += f"Source: {result.payload['file_name']}, Page: {result.payload['page_number']}, Chunk ID: {result.payload['chunk_id']}\n"
        context += f"Content: {result.payload['content']}\n---\n"
        
    if not context:
        return "Sorry, I could not find any relevant information in the documents."

    # 4. Create the detailed prompt for the LLM (the "Augmented" part)
    prompt_template = """
INSTRUCTION: You are a helpful assistant. Your task is to answer the user's question based ONLY on the context provided below.
- If the context contains the answer, provide the answer.
- After the answer, on a new line, cite the exact source using the format: SOURCE: [filename], Page: [page_number], Chunk ID: [chunk_id].
- You MUST use the metadata from the specific chunk that contains the answer for the citation.
- If you cannot find the answer in the context, you MUST respond with "I could not find the answer in the provided documents."
- Do not use any external knowledge or make up information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    
    final_prompt = prompt_template.format(context=context, query=query)
    
    # 5. Generate the answer using the LLM (the "Generation" part)
    # Tokenize the prompt and send it to the GPU
    inputs = tokenizer(final_prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    
    # Generate text without tracking gradients for efficiency
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=512)
    
    # Decode the generated tokens back into a string
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part from the full generated text
    # The model's output includes the original prompt, so we find where the "ANSWER:" part ends.
    answer_start_index = response_text.find("ANSWER:")
    if answer_start_index != -1:
        answer = response_text[answer_start_index + len("ANSWER:"):].strip()
    else:
        # Fallback if the model doesn't follow the format perfectly
        answer = "Could not parse the model's response. The model generated:\n" + response_text

    return answer


# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="RAG Hackathon Chatbot", layout="wide")
st.title("📄 Document-based RAG Chatbot (GPU Version)")
st.write("This chatbot answers questions based only on the documents provided for the hackathon.")

# Load models and clients once on app startup
try:
    embedding_model, tokenizer, llm_model, qdrant_client = load_models_and_clients()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Finding an answer..."):
            full_response = get_answer(prompt, embedding_model, tokenizer, llm_model, qdrant_client)
            message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
