import os
import json
from typing import List, Optional

# LangChain components for document loading, splitting, and embedding
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Define the recommended embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model(device: str = "cpu") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the sentence-transformer embedding model.

    Args:
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        HuggingFaceEmbeddings: The initialized embedding model.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on device: {device}")
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded successfully.")
    return embeddings

def save_extracted_text(text: str, pdf_path: str, format: str = "md") -> str:
    """
    Saves the extracted text from a PDF to a file.
    
    Args:
        text (str): The extracted text content
        pdf_path (str): The path to the original PDF file
        format (str): The output format ("md" or "txt")
        
    Returns:
        str: Path to the saved file
    """
    # Get the directory and base filename without extension
    directory = os.path.dirname(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create the output filename
    output_path = os.path.join(directory, f"{base_name}_extracted.{format}")
    
    # Save the text to the file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Extracted text saved to: {output_path}")
    return output_path


def save_chunks_to_json(chunks: List[Document], pdf_path: str) -> str:
    """
    Saves all chunks to a single JSON file.
    
    Args:
        chunks (List[Document]): The document chunks
        pdf_path (str): The path to the original PDF file
        
    Returns:
        str: Path to the saved JSON file
    """
    # Get the directory and base filename without extension
    directory = os.path.dirname(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create the output filename
    output_path = os.path.join(directory, f"{base_name}_chunks.json")
    
    # Convert chunks to serializable format
    serializable_chunks = []
    for i, chunk in enumerate(chunks):
        serializable_chunks.append({
            "id": i+1,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, indent=2)
    
    print(f"All chunks saved to JSON file: {output_path}")
    return output_path

def process_pdf(pdf_path: str, save_text: bool = True, save_chunks: bool = True, 
               chunk_format: str = "json") -> List[Document]:
    """
    Processes a single PDF document by extracting text, chunking it,
    and preparing it for embedding. Optionally saves the extracted text
    and chunks to files.

    The multi-stagestrategy is ordered from fastest/most structured to slowest/most robust:
    1. Unstructured 'elements' mode: Best for structured, digitally-born PDFs.
    2. PyMuPDFLoader: A very fast and reliable fallback for text-based PDFs that confuse Unstructured.
    3. Unstructured 'hi_res' mode: The final, OCR-based fallback for scanned/image-based PDFs.

    Args:
        pdf_path (str): The file path to the PDF document.
        save_text (bool): Whether to save extracted text as a file.
        save_chunks (bool): Whether to save chunks as files.
        chunk_format (str): Format to save chunks ("txt" for individual files or "json" for a single JSON file).

    Returns:
        List[Document]: A list of processed and chunked Document objects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print(f"--- Processing PDF: {os.path.basename(pdf_path)} ---")

    try:
        
    # 1. EXTRACT text using UnstructuredFileLoader
    # Using "hi_res" strategy for better accuracy with complex layouts and scanned PDFs.
    # This may require additional dependencies like 'unstructured-inference' and 'detectron2'.
    # If you face issues, you might need to install them:
    # pip install "unstructured[all-docs]"
    
        loader = UnstructuredFileLoader(pdf_path, mode="elements")
        docs = loader.load()
    except Exception as e:
        print(f"Unstructured 'elements' mode failed: {e}. Trying next strategy.")
        docs = []
    
    if not docs:
        try:
            # 2. Secondary Fallback: PyMuPDFLoader
            print("Strategy 2: Trying PyMuPDFLoader...")
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            if not docs or sum(len(doc.page_content) for doc in docs) < 100:
                 print("PyMuPDFLoader extracted little text. Trying final strategy.")
                 docs = []
        except Exception as e:
            print(f"PyMuPDFLoader failed: {e}. Trying final strategy.")
            docs = []

    if not docs:
        try:
            # 3. Final Fallback: Unstructured 'hi_res' OCR mode
            # This is slow and requires Tesseract/Poppler, but is a powerful last resort.
            print("Strategy 3: Trying Unstructured 'hi_res' mode (OCR)...")
            # First, ensure dependencies are met. You MUST fix the Poppler error for this to work.
            loader = UnstructuredFileLoader(pdf_path, mode="hi_res")
            docs = loader.load()
        except Exception as e:
            print(f"All extraction strategies failed for {os.path.basename(pdf_path)}. Final error: {e}")
            return [] # Return empty if all methods fail

    if not docs:
        print(f"Warning: Could not extract any content from {os.path.basename(pdf_path)} after all strategies.")
        return []

    print(f"Successfully extracted content using one of the strategies.")

    # --- TEMPORARY DEBUGGING STEP ---
    print(f"\n--- DEBUGGING {os.path.basename(pdf_path)} ---")
    if not docs:
        print("UnstructuredFileLoader returned NO documents.")
    else:
        print(f"Extracted {len(docs)} elements.")
        # Check if the content is empty
        total_content_length = sum(len(doc.page_content) for doc in docs)
        print(f"Total characters extracted: {total_content_length}")
        if total_content_length < 100: # If very little content was found
            print("WARNING: Very little or no text content was extracted. This may be a scanned/image-based PDF.")
    print("--- END DEBUG ---\n")
    # --- END TEMPORARY DEBUGGING STEP ---


    # We join the elements back into a single string for the splitter
    # Unstructured 'elements' mode gives us a list of Text elements
    full_text = "\n\n".join([doc.page_content for doc in docs])
    
    if not full_text.strip():
        print(f"Warning: No text could be extracted from {os.path.basename(pdf_path)}. It might be an image-only PDF or corrupted. Skipping chunking.")
        return []

    # Save the extracted text if requested
    if save_text:
        save_extracted_text(full_text, pdf_path, format="md")

    # 2. CHUNK the text smartly
    # We will use a combination of Markdown and Recursive splitting for robustness.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(full_text)

    # Now, recursively split the text from each markdown section
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""] # Standard separators
    )
    
    chunks = []
    # If markdown splitting was effective, split within those sections
    if len(md_header_splits) > 1:
        print(f"Document structure found with {len(md_header_splits)} markdown sections. Applying recursive splitting to each section.")
        for md_split in md_header_splits:
            chunks.extend(r_splitter.split_documents([md_split]))
    else:
        # Otherwise, split the whole document recursively
        print("No significant markdown structure found. Applying recursive splitting to the entire document.")
        chunks = r_splitter.split_text(full_text)
        # Convert text chunks to Document objects
        chunks = [Document(page_content=chunk) for chunk in chunks]

    if not chunks:
        print(f"Warning: Text was extracted but chunking resulted in 0 chunks for {os.path.basename(pdf_path)}. Creating a single chunk with the full text.")
        chunks = [Document(page_content=full_text)]

    print(f"Successfully split document into {len(chunks)} chunks.")

    # 3. Augment chunks with metadata
    # Add the source file name to each chunk's metadata for later reference.
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf_path)

    # 4. Save the chunks if requested
    if save_chunks:
        save_chunks_to_json(chunks, pdf_path)

    print(f"--- Finished processing {os.path.basename(pdf_path)} ---")
    return chunks

