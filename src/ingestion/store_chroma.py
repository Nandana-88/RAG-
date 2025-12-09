import os
from langchain_chroma import Chroma

def store_to_chroma(chunks, persist_directory, embedding_model, collection_name=None):
    """
    Initialize (or load) a Chroma vector store and persist the given chunks.
    Minimal, no typing or path logic — expects strings/objects passed in from caller.
    """
    print(f"DEBUG: store_to_chroma called with collection_name={collection_name}")
    
    # Fix for TypeError: argument 'name': 'NoneType' object cannot be converted to 'PyString'
    # ChromaDB requires a string name, cannot be None.
    if collection_name is None:
        collection_name = "langchain"
        
    chroma = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    chunk_list = list(chunks)
    if chunk_list:
        chroma.add_documents(chunk_list)
        # chroma.persist() # New Chroma automatically persists, but we can verify
        
    print(f"DEBUG: Successfully stored {len(chunk_list)} chunks to {persist_directory}")
    return chroma
