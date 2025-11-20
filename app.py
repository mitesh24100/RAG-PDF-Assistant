import streamlit as st
import os
import tempfile
from rag import process_document, get_vectorstore, get_rag_chain

# Page Config
st.set_page_config(page_title="Gemini RAG PDF Chat", layout="wide")

def get_api_key():
    """Helper to get API key from environment or user input"""
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    return st.sidebar.text_input("Google API Key", type="password")

st.title("📄 Chat with PDF using Gemini")

# --- Sidebar & Configuration ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = get_api_key()
    
    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    process_button = st.button("Process PDF")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- PDF Processing Logic ---
if process_button and uploaded_file and google_api_key:
    with st.spinner("Processing PDF..."):
        # Save uploaded file to a temporary file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Call functions from rag.py
        splits = process_document(temp_file_path)
        
        # Cleanup temp file
        os.remove(temp_file_path)

        if splits:
            st.session_state.vectorstore = get_vectorstore(splits, google_api_key)
            st.session_state.rag_chain = get_rag_chain(st.session_state.vectorstore, google_api_key)
            st.success("PDF Processed! You can now ask questions.")
        else:
            st.error("Could not process the PDF. Please try another file.")

# --- Chat Interface ---
if not google_api_key:
    st.warning("Please enter your Google API Key to continue.")
elif not st.session_state.rag_chain:
    st.info("Please upload and process a PDF to start chatting.")
else:
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if user_input := st.chat_input("Ask a question about the PDF..."):
        # Display user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")