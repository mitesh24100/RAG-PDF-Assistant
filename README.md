****📄 PDF RAG Assistant (Ollama + Streamlit + LangChain)****  

A simple and powerful RAG (Retrieval-Augmented Generation) project that:
* Let users upload a PDF
* Creates vector embeddings using Ollama embeddings models
* Runs queries using phi3-mini (via Ollama)
* Uses Streamlit for a clean interactive UI
* Uses FAISS vector store for document retrieval
  
This project runs fully locally (privacy-friendly) and does not require any paid API keys.

📂 Project Structure
```
rag-pdf-assistant/
│
├── app.py                 # Streamlit UI
├── rag.py                 # RAG logic
│
├── pdfs/                  # Uploaded PDFs temporarily stored
│   └── demo.pdf
│
├── db/
│   └── faiss/             # Vector DB stored here
│
├── pyproject.toml         # Poetry config
├── poetry.lock
└── README.md
```

🚀 Features
* Upload any PDF
* Auto-chunk + embed with nomic-embed-text
* Query answers using phi3-mini
* Vector DB resets automatically on new uploads
* Fully local (privacy-safe)
* Clean modern UI

🛠 1. **Requirements**
* macOS / Linux / Windows
* Python 3.10+
* Ollama installed
* Poetry installed

🧠 2. **Install Ollama**
```
https://ollama.com/download
```

📥 3. **Pull Required Models**

**Embedding model**
```
ollama pull nomic-embed-text
```

**LLM**
```
ollama pull phi3:mini
```

📦 4. **Project Setup Using Poetry**

1. Install dependencies
```
poetry install
```

2. Enter virtual environment
```
poetry shell
```
**Note**: If you get a virtual environment created in this step, but not activated, run below command
```
. /path to virtual environment/bin/activate
```

3. Run Ollama
   
Make sure Ollama Application is running or run below command
```
ollama serve
```

4. Run Streamlit Application
```
streamlit run app.py
```

▶️ 5. **Running the Application**

Once running, open:
```
http://localhost:8501
```

Workflow:
* Upload a PDF
* Vector DB is built
* Ask questions
* Answers come from the PDF only





