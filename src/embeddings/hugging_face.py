# src/embeddings/hugging_face.py

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
    except ImportError:
        # Fallback for older environments
        print("Warning: langchain_huggingface not found, using legacy langchain.embeddings")
        from langchain.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
