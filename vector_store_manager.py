import os
import shutil
import uuid
from typing import List
import streamlit as st

# LangChain components
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings.base import Embeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from our previous script
from document_processor import process_pdf, get_embedding_model

# Define a constant for the database directory
DB_DIRECTORY = "db"

def get_secrets():
    """Get secrets from Streamlit secrets or fallback to environment variables."""
    try:
        # Try to get from Streamlit secrets first
        qdrant_url = st.secrets.get("QDRANT_URL")
        qdrant_api_key = st.secrets.get("QDRANT_API_KEY")
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        
        # Fallback to environment variables if secrets not available
        if not qdrant_url:
            qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_api_key:
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not google_api_key:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            
        return {
            "QDRANT_URL": qdrant_url,
            "QDRANT_API_KEY": qdrant_api_key,
            "GOOGLE_API_KEY": google_api_key
        }
    except Exception as e:
        logger.warning(f"Could not access Streamlit secrets: {e}. Using environment variables.")
        return {
            "QDRANT_URL": os.getenv("QDRANT_URL"),
            "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
        }

def get_qdrant_client():
    """
    Returns a Qdrant client - either cloud or local based on secrets/environment variables.
    """
    secrets = get_secrets()
    qdrant_url = secrets["QDRANT_URL"]
    qdrant_api_key = secrets["QDRANT_API_KEY"]
    
    if qdrant_url and qdrant_api_key:
        logger.info(f"🌐 Connecting to Qdrant Cloud at: {qdrant_url}")
        logger.info(f"🔑 Using API Key: {'*' * (len(qdrant_api_key) - 8) + qdrant_api_key[-8:]}")
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Verify cloud connection
        try:
            info = client.get_collections()
            logger.info(f"✅ Successfully connected to Qdrant Cloud! Collections: {len(info.collections)}")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant Cloud: {e}")
            raise
    else:
        logger.warning("🏠 No cloud credentials found. Using local Qdrant (not recommended for production)")
        return QdrantClient(host="localhost", port=6333)

def verify_connection_type():
    """
    Verify if you're connected to cloud or local Qdrant and return connection info.
    """
    try:
        client = get_qdrant_client()
        
        # Try to get cluster info (only available in cloud)
        try:
            # This is a cloud-specific endpoint
            collections = client.get_collections()
            
            secrets = get_secrets()
            qdrant_url = secrets["QDRANT_URL"] or ""
            if "cloud.qdrant.io" in qdrant_url:
                return {
                    "type": "cloud",
                    "url": qdrant_url,
                    "collections_count": len(collections.collections),
                    "status": "connected"
                }
            else:
                return {
                    "type": "local",
                    "url": "localhost:6333",
                    "collections_count": len(collections.collections),
                    "status": "connected"
                }
        except Exception as e:
            return {
                "type": "unknown",
                "error": str(e),
                "status": "error"
            }
    except Exception as e:
        return {
            "type": "error",
            "error": str(e),
            "status": "connection_failed"
        }

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


def create_qdrant_vector_store(
    documents: List[Document], 
    embedding_model: Embeddings, 
    collection_name: str
) -> QdrantVectorStore:
    """
    Creates a Qdrant vector store from a list of documents.
    
    Args:
        documents: List of LangChain Document objects
        embedding_model: The embedding model to use
        collection_name: Name for the Qdrant collection
    
    Returns:
        QdrantVectorStore: The created vector store
    """
    client = get_qdrant_client()
    
    # Log where data will be stored
    connection_info = verify_connection_type()
    logger.info(f"📊 Creating collection '{collection_name}' in {connection_info['type']} Qdrant")
    
    try:
        secrets = get_secrets()
        vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embedding_model,
            url=secrets["QDRANT_URL"],
            api_key=secrets["QDRANT_API_KEY"],
            collection_name=collection_name,
            vector_params=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        logger.info(f"✅ Successfully created vector store with {len(documents)} documents")
        return vector_store
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        raise

def create_empty_vector_store(
    embedding_model: Embeddings, 
    collection_name: str
) -> QdrantVectorStore:
    """
    Creates an empty Qdrant vector store.
    
    Args:
        embedding_model: The embedding model to use
        collection_name: Name for the Qdrant collection
    
    Returns:
        QdrantVectorStore: The created empty vector store
    """
    secrets = get_secrets()
    qdrant_url = secrets["QDRANT_URL"]
    qdrant_api_key = secrets["QDRANT_API_KEY"]
    
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in Streamlit secrets or environment variables")
    
    # Create the empty vector store
    vector_store = QdrantVectorStore(
        client=QdrantClient(url=qdrant_url, api_key=qdrant_api_key),
        collection_name=collection_name,
        embedding=embedding_model,
        distance=Distance.COSINE,
    )
    
    return vector_store

def delete_collection(collection_name: str) -> bool:
    """
    Deletes a collection from Qdrant Cloud.
    
    Args:
        collection_name: Name of the collection to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        secrets = get_secrets()
        qdrant_url = secrets["QDRANT_URL"]
        qdrant_api_key = secrets["QDRANT_API_KEY"]
        
        if not qdrant_url or not qdrant_api_key:
            return False
            
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Check if collection exists before trying to delete
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            client.delete_collection(collection_name)
            return True
        return True  # Collection doesn't exist, consider it successful
        
    except Exception as e:
        print(f"Error deleting collection {collection_name}: {e}")
        return False

def cleanup_session_collections(session_id: str) -> bool:
    """
    Cleans up all collections associated with a session.
    
    Args:
        session_id: The session ID to clean up
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        collection_name = f"policies-{session_id}"
        return delete_collection(collection_name)
    except Exception as e:
        print(f"Error cleaning up session {session_id}: {e}")
        return False

def list_collections_info():
    """
    Lists all collections and their info to verify data storage location.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        connection_info = verify_connection_type()
        
        info = {
            "connection_type": connection_info["type"],
            "connection_url": connection_info.get("url", "unknown"),
            "total_collections": len(collections.collections),
            "collections": []
        }
        
        for collection in collections.collections:
            collection_info = client.get_collection(collection.name)
            info["collections"].append({
                "name": collection.name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            })
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Error getting collections info: {e}")
        return {"error": str(e)}

def cleanup_session_collections(session_id: str):
    """
    Cleans up collections related to a specific session.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        
        for collection in collections.collections:
            if session_id in collection.name:
                logger.info(f"🗑️ Deleting collection: {collection.name}")
                client.delete_collection(collection.name)
                
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")


