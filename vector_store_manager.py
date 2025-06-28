import os
import shutil
from typing import List

# LangChain components
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import from our previous script
from document_processor import process_pdf, get_embedding_model
# In vector_store_manager.py

# NOTE: We have removed DB_DIRECTORY and the load_vector_store function.

def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str
) -> Chroma:
    """
    Creates an IN-MEMORY ChromaDB vector store.
    It does not save to disk.
    """
    print(f"Creating IN-MEMORY vector store with collection name: {collection_name}")
    
    # Filter out chunks with empty page_content, which can cause errors
    filtered_chunks = [chunk for chunk in chunks if chunk.page_content]
    if not filtered_chunks:
        print("Warning: All document chunks were empty. No vector store will be created.")
        return None

    # Create the vector store without a persist_directory to keep it in memory
    vector_store = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
    print("In-memory vector store created successfully.")
    return vector_store
