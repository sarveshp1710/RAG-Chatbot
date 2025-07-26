import os
import glob
from typing import List
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid
import nltk

# Import configuration from our config.py file
from config import (
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    CHUNK_SIZE
)

print("Starting data ingestion process...")

def load_documents(folder_path: str) -> List[dict]:
    """Loads all PDF and Word documents from a specified folder."""
    documents = []
    # Find all files ending with .pdf or .docx in the given folder
    file_paths = glob.glob(os.path.join(folder_path, "*.pdf")) + \
                 glob.glob(os.path.join(folder_path, "*.docx"))

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Loading document: {file_name}")
        try:
            if file_path.endswith(".pdf"):
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    documents.append({
                        "text": page.get_text(),
                        "meta": {"file_name": file_name, "page_number": page_num + 1}
                    })
                doc.close()
            elif file_path.endswith(".docx"):
                doc = docx.Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                documents.append({
                    "text": full_text,
                    "meta": {"file_name": file_name, "page_number": 1}
                })
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    return documents

def create_chunks(documents: List[dict], chunk_size: int) -> List[dict]:
    """Splits documents into chunks based on sentences using NLTK."""
    all_chunks = []
    print("Chunking documents using NLTK sentence tokenizer...")

    # Download the 'punkt' tokenizer models if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')

    for doc in documents:
        # Split the text into sentences
        sentences = nltk.sent_tokenize(doc["text"])
        
        current_chunk = ""
        for sentence in sentences:
            # If adding the next sentence doesn't exceed the chunk size, add it
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += " " + sentence
            else:
                # Otherwise, the chunk is complete. Add it to our list.
                if current_chunk:
                    all_chunks.append({
                        "text": current_chunk.strip(),
                        "meta": doc["meta"]
                    })
                # Start a new chunk with the current sentence
                current_chunk = sentence
        
        # Add the last remaining chunk
        if current_chunk:
            all_chunks.append({
                "text": current_chunk.strip(),
                "meta": doc["meta"]
            })
            
    print(f"Created {len(all_chunks)} chunks.")
    return all_chunks

def main():
    # 1. Load documents from the 'documents' folder
    documents_data = load_documents("documents/")
    if not documents_data:
        print("No documents found in the 'documents' folder. Exiting.")
        return

    # 2. Create chunks from the loaded documents
    all_chunks = create_chunks(documents_data, CHUNK_SIZE)

    # 3. Initialize the embedding model, loading it onto the GPU
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    print("Embedding model loaded.")

    # 4. Convert all chunk texts into vector embeddings
    print("Embedding chunks (this may take a while)...")
    chunk_embeddings = embedding_model.encode(
        [chunk['text'] for chunk in all_chunks],
        show_progress_bar=True
    )
    print("Chunk embedding complete.")

    # 5. Connect to the Qdrant database
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 6. Set up the Qdrant collection
    vector_size = embedding_model.get_sentence_embedding_dimension()
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created.")

    # 7. Upload all the vectors and their metadata to Qdrant
    print("Uploading data to Qdrant...")
    # Generate a unique ID for each chunk/point
    chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
    
    qdrant_client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=vector.tolist(),
                payload={
                    "file_name": chunk["meta"]["file_name"],
                    "page_number": chunk["meta"]["page_number"],
                    "content": chunk["text"],
                    "chunk_id": chunk_id # This ID is required by the hackathon rules
                }
            )
            for chunk_id, vector, chunk in zip(chunk_ids, chunk_embeddings, all_chunks)
        ],
        wait=True
    )
    print("Data ingestion complete!")

if __name__ == "__main__":
    main()
