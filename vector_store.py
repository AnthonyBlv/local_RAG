import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DATA_PATH = "data"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Wrapper to handle text file encoding issues
def safe_text_loader(path):
    return TextLoader(path, encoding='utf-8')

# Mapping for file extensions
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": safe_text_loader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx": UnstructuredWordDocumentLoader,
}

# Initialize Embeddings once
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_all_files_in_directory(directory):
    """Scans the directory and returns a set of full file paths."""
    file_paths = set()
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in LOADER_MAPPING:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.add(full_path)
    return file_paths

def load_specific_files(file_paths):
    """Loads only the specific list of files provided."""
    documents = []
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1]
        if ext in LOADER_MAPPING:
            try:
                # print(f"Loading: {os.path.basename(file_path)}")
                loader_fn = LOADER_MAPPING[ext]
                loader = loader_fn(file_path)
                docs = loader.load()
                
                # Standardize Metadata
                for doc in docs:
                    doc.metadata["source"] = file_path
                    doc.metadata["filename"] = os.path.basename(file_path)
                
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents

def get_vector_store():
    """Returns the singleton vector store instance."""
    if not os.path.exists(PERSIST_DIR):
        print("Creating new vector store...")
        # Initial load if DB doesn't exist
        all_files = get_all_files_in_directory(DATA_PATH)
        docs = load_specific_files(all_files)
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            return Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=PERSIST_DIR,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            # Return empty store if no files found
            return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def sync_vector_store(vector_store):
    """Synchronizes the Vector Store with the actual files on disk."""
    print("--- Starting Synchronization ---")
    
    # 1. Get DB State
    db_data = vector_store.get()
    existing_sources = set()
    if db_data['metadatas']:
        for meta in db_data['metadatas']:
            if meta and 'source' in meta:
                existing_sources.add(os.path.abspath(meta['source']))
    
    # 2. Get Disk State
    current_files = get_all_files_in_directory(DATA_PATH)
    
    # 3. Calculate Diff
    new_files = current_files - existing_sources
    deleted_files = existing_sources - current_files
    
    # 4. Handle Deletions
    if deleted_files:
        ids_to_delete = []
        for idx, meta in enumerate(db_data['metadatas']):
            if meta and os.path.abspath(meta.get('source', '')) in deleted_files:
                ids_to_delete.append(db_data['ids'][idx])
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks.")
            
    # 5. Handle Additions
    if new_files:
        new_docs = load_specific_files(new_files)
        if new_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            new_chunks = splitter.split_documents(new_docs)
            vector_store.add_documents(new_chunks)
            print(f"Added {len(new_chunks)} new chunks.")
            
    print("--- Sync Complete ---")
    return vector_store
