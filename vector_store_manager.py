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

# Define a constant for the database directory
DB_DIRECTORY = "db"


def create_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: str = DB_DIRECTORY
) -> Chroma:
    """
    Creates a ChromaDB vector store from a list of document chunks.

    Args:
        chunks (List[Document]): The list of document chunks to be stored.
        embeddings (HuggingFaceEmbeddings): The embedding model to use.
        collection_name (str): The name of the collection to create in ChromaDB.
        persist_directory (str): The directory to save the database to.

    Returns:
        Chroma: The created Chroma vector store object.
    """
    print(f"Creating vector store with collection name: {collection_name}")

    # Filter out complex metadata from the documents
    filtered_chunks = filter_complex_metadata(chunks)

    # Use Chroma.from_documents to create the vector store in one go.
    # This will process the chunks, create embeddings, and store them.
    vector_store = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    # The .persist() call is implicitly handled by from_documents with a persist_directory
    print(f"Vector store created and persisted to: {persist_directory}")
    return vector_store


def load_vector_store(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: str = DB_DIRECTORY
) -> Chroma:
    """
    Loads an existing ChromaDB vector store from disk.

    Args:
        embeddings (HuggingFaceEmbeddings): The embedding model to use.
        collection_name (str): The name of the collection to load.
        persist_directory (str): The directory where the database is stored.

    Returns:
        Chroma: The loaded Chroma vector store object.
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Persistence directory not found: {persist_directory}")
    
    print(f"Loading vector store from: {persist_directory} with collection: {collection_name}")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vector_store


