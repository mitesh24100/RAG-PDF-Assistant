import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama


FAISS_DIR = "./db/faiss"
os.environ["CHROMA_TELEMETRY"] = "false"


# ------------------------------------------------------------
# 1. Load + Chunk PDF
# ------------------------------------------------------------
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 2. Create or Load Vector Database
# ------------------------------------------------------------
def create_vector_db(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    os.makedirs(os.path.dirname(FAISS_DIR), exist_ok=True)
    vectordb.save_local(FAISS_DIR)

    return vectordb


def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


# ------------------------------------------------------------
# 3. RAG Query + Response
# ------------------------------------------------------------
def ask_question(vectordb, query):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOllama(model="phi3:mini")   # <-- Updated model

    prompt = f"""
    You are a helpful assistant. Use ONLY the context below to answer.
    
    CONTEXT:
    {context}

    QUESTION:
    {query}

    If the answer is not in the context, say: "The document does not contain that information."
    """

    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    pass