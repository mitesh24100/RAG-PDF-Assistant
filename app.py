import os
import shutil
import streamlit as st


from rag import (
    load_and_split_pdf,
    create_vector_db,
    load_vector_db,
    ask_question,
    FAISS_DIR,
)

os.environ["CHROMA_TELEMETRY"] = "false"


# ----------------------------
# Clear vector DB
# ----------------------------
def clear_vector_db():
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
    os.makedirs(FAISS_DIR, exist_ok=True)


# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("📄 PDF RAG Assistant")
    st.markdown("Ask questions **based on your uploaded PDF**, powered by Ollama & FAISS.")

    st.markdown("---")

    st.info("Upload your PDF on the main screen →")


# ----------------------------
# Main Layout
# ----------------------------
st.title("✨ Your Intelligent PDF Assistant")

st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 10px;
    }
    .uploaded-pdf {
        border-radius: 10px;
        padding: 12px;
        background-color: rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

# Maintain chat history
if "chat" not in st.session_state:
    st.session_state.chat = []   # Stores {"role": "user"/"assistant", "msg": "text"}


# ----------------------------
# File Uploaded
# ----------------------------
if uploaded_file:
    
    pdf_path = f"./pdfs/temp_{uploaded_file.name}"
    os.makedirs("./pdfs", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Clear old DB
    clear_vector_db()

    st.info("📚 Processing your PDF… extracting text + splitting into chunks…")
    chunks = load_and_split_pdf(pdf_path)

    st.info("🔍 Building vector database…")
    vectordb = create_vector_db(chunks)

    st.success("✅ Vector DB ready! Ask your question below.")

    query = st.chat_input("Ask anything about your PDF...")

    # ----------------------------
    # Chat Interaction
    # ----------------------------
    if query:
        st.session_state.chat.append({"role": "user", "msg": query})

        with st.spinner("Thinking..."):
            answer = ask_question(vectordb, query)

        st.session_state.chat.append({"role": "assistant", "msg": answer})

    # ----------------------------
    # Display Chat Messages
    # ----------------------------
    for c in st.session_state.chat:
        if c["role"] == "user":
            st.markdown(f"**🧑 You:** {c['msg']}")
        else:
            st.markdown(f"**🤖 Assistant:** {c['msg']}")

    # Cleanup temp PDF
    try:
        os.remove(pdf_path)
    except:
        pass
