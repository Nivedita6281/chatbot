import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DOCS_DIR = "stored_documents"  # Folder containing the documents
VECTOR_STORE_PATH = "faiss_index"  # Path to save/load FAISS vector store

def preprocess_and_store():
    """Reads documents, splits into chunks, and stores in FAISS."""
    # Load documents
    documents = []
    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        if os.path.isfile(filepath) and filepath.endswith(".txt"):
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings and create FAISS index
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save FAISS index
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Vector store created and saved successfully!")

def load_vector_store():
    """Loads an existing FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
        return vector_store
    else:
        raise FileNotFoundError("Vector store not found. Please run `preprocess_and_store` first.")

if __name__ == "__main__":
    preprocess_and_store()
