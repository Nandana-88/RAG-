import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from src.config import INFO_DIR, PG_DIR, UG_DIR


def load_text_files(path):
    return DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=lambda p: TextLoader(p, autodetect_encoding=True)
    ).load()


def load_pdf_files(path):
    return DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    ).load()


def load_documents(data_path=None):
    """
    Load documents from specified path or default directories.
    
    Args:
        data_path: Optional path to load documents from. If None, loads from default directories.
    
    Returns:
        List of loaded documents
    """
    if data_path:
        # Load from specified path (supports both txt and pdf)
        all_docs = []
        if os.path.exists(data_path):
            try:
                # Try loading text files
                txt_docs = load_text_files(data_path)
                all_docs.extend(txt_docs)
            except Exception as e:
                print(f"No text files found in {data_path}: {e}")
            
            try:
                # Try loading PDF files
                pdf_docs = load_pdf_files(data_path)
                all_docs.extend(pdf_docs)
            except Exception as e:
                print(f"No PDF files found in {data_path}: {e}")
        else:
            print(f"Warning: Path {data_path} does not exist")
        
        return all_docs
    else:
        # Load from default directories
        info_docs = load_text_files(INFO_DIR)
        pg_docs = load_pdf_files(PG_DIR)
        ug_docs = load_pdf_files(UG_DIR)
        
        return info_docs + pg_docs + ug_docs
