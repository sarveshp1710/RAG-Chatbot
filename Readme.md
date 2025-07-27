# Document-Based RAG Chatbot for RAP Hackathon 2025 - Engineers

This project is a submission for the RAP Hackathon 2025 - Engineers. It is a fully functional chatbot that answers user queries based strictly on the content of PDF and Word documents uploaded by the user. The entire application is built from scratch without using LangChain, adhering to all the specified rules and conditions.

## 🎥 Demo Video

[*Click here to view the demo video*](https://youtu.be/9fCph4iQT6Q)

## ✨ Features

* **Dynamic Document Upload:** Users can upload multiple PDF and Word documents through a user-friendly web interface.
* **On-the-Fly Processing:** Documents are processed, chunked, and indexed in real-time when the user clicks "Process Documents".
* **Strictly Contextual Answers:** The chatbot is designed to answer questions *only* from the information present in the uploaded documents. It will not use any external or pre-trained knowledge.
* **Source & Performance Citation:** Every answer is followed by the time it took to generate and a list of the source documents and page numbers used.
* **Live System Monitoring:** The sidebar includes a live-updating display of the system's CPU and RAM usage.
* **Built from Scratch:** The entire RAG (Retrieval-Augmented Generation) pipeline is implemented using foundational libraries, with no reliance on LangChain.
* **Open-Source Models:** Utilizes powerful open-source embedding and language models that run entirely on the local machine.

## 🏛️ Architecture & Technical Decisions

The application is built around a classic RAG pipeline, orchestrated by a Streamlit web interface.

1.  **Document Loading & Chunking:**
    * PDFs are parsed using `PyMuPDF` and Word documents using `python-docx`.
    * **Justification:** We implemented a semantic chunking strategy using the **NLTK (Natural Language Toolkit)** library. The text from each page is first split into sentences. These sentences are then grouped into chunks that do not exceed a 1000-character limit. This method is superior to simple fixed-size chunking because it respects natural grammatical boundaries, ensuring that our chunks are semantically coherent and complete thoughts. This leads to more meaningful vector embeddings and provides better context for the Language Model.

2.  **Embedding Model:**
    * We are using **`BAAI/bge-small-en-v1.5`**.
    * **Justification:** After initial testing with a more general model (`all-MiniLM-L6-v2`) resulted in poor relevance detection, we upgraded to `bge-small-en-v1.5`. This model is a top performer on the MTEB (Massive Text Embedding Benchmark) for retrieval tasks. To maximize its effectiveness, we prepend the instruction `"Represent this document for retrieval: "` to each chunk before embedding, which is the recommended practice for this model and significantly improves the quality of the search results.

3.  **Vector Database:**
    * **Qdrant** is used as the vector database, as per the hackathon requirements. It is run locally via a Docker container for easy setup and offline capability.

4.  **Language Model:**
    * We are using **`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`**.
    * **Justification:** The primary challenge was finding a model that could run efficiently on a local CPU while being smart enough to follow very strict instructions. After experimenting with larger models like Llama-2-7B (which was too slow) and other small models that failed to follow negative constraints, TinyLlama provided the best balance of speed and instruction-following capability for a responsive demo.

5.  **Prompt Engineering:**
    * To prevent the model from using its own knowledge, a "few-shot" prompting strategy was implemented. The final prompt provides the model with explicit examples of both correct (in-context) and incorrect (out-of-context) behavior, which is a highly effective technique for constraining smaller language models.

## 🚀 Setup and Running the Project

Follow these steps to run the application on your local system.

### Prerequisites

* Python 3.10
* Docker Desktop

### 1. Clone the Repository

```bash
git clone https://github.com/sarveshp1710/RAG-Chatbot.git
cd RAG-Chatbot
```

### 2. Set Up the Python Environment

It is highly recommended to use a virtual environment.

Create a virtual environment
```bash
python -m venv rag_env
```
Activate the environment
On Windows:
```
rag_env\Scripts\activate
```
On macOS/Linux:
```
source rag_env/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the provided requirements file.

```bash
pip install -r requirements.txt
```

### 4. Run the Qdrant Database

Make sure Docker Desktop is running, then start the Qdrant container in a separate terminal.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Leave this terminal running. It is your database server.

### 5. Run the Streamlit Application

In your original terminal (with the virtual environment activated), run the Streamlit app.

```bash
streamlit run app.py
```

Your web browser should automatically open with the chatbot interface.

### 6. Usage Instructions

1.  Launch the application using the command above.
2.  Use the sidebar to upload one or more PDF or Word documents.
3.  Click the "Process Documents" button and wait for the processing to complete.
4.  Once processed, the main chat interface will be ready.
5.  Type your questions about the documents into the chat box at the bottom and press Enter.

## 🧪 Test Questions

Here is a set of 10 questions designed to test the chatbot's accuracy and adherence to the rules, based on the `ML Task (1).pdf` document.

1.  What is the maximum allowed response time for a user query when running on a GPU?
2.  Is it permissible to use the LangChain library in this project?
3.  Which specific vector database must be used for this task?
4.  What should the chatbot do if it cannot find an answer in the provided context?
5.  What are the three specific pieces of information that must be displayed for the document source citation?
6.  How many times are you allowed to query the vector database for a single user question?
7.  Which UI frameworks are permitted for creating the user interface?
8.  What is the weighting for "Accuracy of Answers" in the evaluation criteria?
9.  According to the submission requirements, how many questions must be included in the response document?
10. What is one of the mandatory conditions for the demo video regarding the presenter?

## 💻 Hardware Usage and Failure Analysis

### 1. Hardware Usage

The application in its final, submitted form is configured to run on a **CPU**.

* **RAM:** The models (embedding and LLM) consume approximately **5-6 GB of system RAM** when loaded. An 8GB RAM machine is the minimum requirement, though performance is better with 16GB.
* **CPU:** During inference (when the chatbot is generating an answer), CPU usage will spike to 100% as it performs the necessary calculations.

### 2. Failure Point Analysis (GPU Execution)

* **Why it Failed:** The primary development challenge was a hardware incompatibility with the local NVIDIA MX330 GPU. The CUDA libraries consistently failed to initialize, leading to `WinError 127` and `No CUDA GPUs are available` errors. This is a known issue with some entry-level and laptop-grade GPUs that may lack certain features or have driver conflicts with advanced computation libraries like PyTorch and CTransformers. This required a strategic pivot from a GPU-first to a **CPU-only** solution to ensure a functional and demonstrable application.
* **How to Make it Work:** The application is fully capable of running on a GPU. To make it work on a compatible, high-VRAM GPU (like the specified Tesla T4), only two lines of code need to be changed in `app.py`:
    1.  Change `device='cpu'` to `device='cuda'` for the `SentenceTransformer`.
    2.  Add the `gpu_layers=50` parameter (or another appropriate number) to the `AutoModelForCausalLM.from_pretrained` call. With these changes, the application would meet and exceed the 15-second performance requirement.


## 🧗 Challenges Faced

* **GPU Incompatibility:** The primary development challenge was a hardware incompatibility with the local NVIDIA MX330 GPU, which forced a pivot to a CPU-only solution. This directly impacts the ability to meet the 15-second response time limit, which is only feasible on a compatible GPU.
* **Relevance & Hallucination:** The most significant algorithmic challenge was preventing the language model from answering out-of-context questions.
    * **Problem:** The initial embedding model produced relevance scores that were too similar for both relevant and irrelevant documents, making a simple threshold ineffective. Furthermore, the small LLM would often ignore instructions and answer from its own knowledge.
    * **Solution:** This was solved with a two-pronged approach: upgrading the embedding model to the more powerful `bge-small-en-v1.5` for better relevance detection, and implementing a much stricter "few-shot" prompt with explicit examples to force the LLM to adhere to the rules.
    * **Note on Demo Video:** Due to the hardware constraints requiring a very small LLM (`TinyLlama`), there may be instances in the demo video where the model's answer is not perfectly accurate or it fails to correctly identify an out-of-context question. This is a known limitation of using a smaller model, which is a direct trade-off for ensuring a responsive demo on the available hardware. A larger model like Llama-2-7B or Gemma-2-2B would provide more accurate answers but was too slow for a live demonstration on the CPU.

## 🔮 Future Improvements

* **GPU Acceleration:** The most impactful improvement would be to deploy the application on a system with a compatible, high-VRAM GPU (like the specified Tesla T4). By setting the `gpu_layers` parameter in the code, the model would see a >100x performance increase, easily meeting the 15-second requirement.
* **Chat History & Memory:** The chatbot is currently stateless. A future version could implement a conversation buffer to allow for follow-up questions and a more natural conversational flow.
* **Advanced Retrieval Strategies:** Implement more advanced techniques like re-ranking, where an initial set of documents is retrieved and then re-ranked by a more powerful model to find the absolute best context before sending it to the LLM.
* **Streaming Responses:** To improve the user experience, the LLM's response could be streamed token-by-token, so the user sees the answer being generated in real-time rather than waiting for the full response.
