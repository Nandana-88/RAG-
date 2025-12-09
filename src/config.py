# Storage
PERSIST_DIR = "chroma_db"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "models/gemini-2.5-flash"  # Fast, stable, and supports 1M tokens

# API Key env variable
GOOGLE_API_KEY_ENV = "GEMINI_API_KEY"

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# Retriever
TOP_K = 3

# config.py
INFO_DIR = "data/info"
PG_DIR = "data/pg"
UG_DIR = "data/ug"
