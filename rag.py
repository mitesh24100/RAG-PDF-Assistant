import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama


PDF_PATH = "./pdfs/demo.pdf"
CHROMA_DIR = "./db/chroma"
os.environ["CHROMA_TELEMETRY"] = "false"


# ------------------------------------------------------------
# 1. Load + Chunk PDF
# ------------------------------------------------------------
def load_and_split_pdf():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        # smaller chunks for phi3-mini
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


# ------------------------------------------------------------
# 2. Create or Load Vector Database
# ------------------------------------------------------------
def create_vector_db(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectordb


def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


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


# ------------------------------------------------------------
# 4. Main Loop
# ------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR) or len(os.listdir(CHROMA_DIR)) == 0:
        print("Building vector database...")
        chunks = load_and_split_pdf()
        vectordb = create_vector_db(chunks)
        print("Vector DB created.")
    else:
        print("Loading existing vector DB...")
        vectordb = load_vector_db()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_question(vectordb, query)
        print("\nANSWER:\n", answer)
