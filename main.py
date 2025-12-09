import os
import argparse
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env

from src.config import (
    PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    GOOGLE_API_KEY_ENV,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)

from src.ingestion.load_docs import load_documents
from src.ingestion.split_docs import split_documents
from src.ingestion.store_chroma import store_to_chroma

from src.embeddings.hugging_face import get_embeddings
from src.utils.helpers import ensure_dir

# NOTE: do NOT import build_chain or langchain-related modules at top-level.
# They will be imported lazily inside chat() to avoid import-time failures during ingest.

def ingest(data_path):
    """Ingest documents from the specified path into ChromaDB."""
    ensure_dir(PERSIST_DIR)

    print(f"Loading documents from: {data_path}")
    docs = load_documents(data_path)
    if not docs:
        print("❌ No documents found to ingest.")
        return

    print(f"✅ Loaded {len(docs)} documents")
    chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"✅ Split into {len(chunks)} chunks")
    
    embeddings = get_embeddings(EMBEDDING_MODEL)
    store_to_chroma(chunks, PERSIST_DIR, embeddings)

    print(f"✅ Successfully ingested {len(docs)} documents into ChromaDB!")


def chat(question, context=None, context_file=None):
    # Lazy import of chain so ingest doesn't require langchain/llm packages
    from src.rag.chain import build_chain, ask  # <<-- lazy import
    from langchain_community.vectorstores import Chroma
    from src.embeddings.hugging_face import get_embeddings

    # 1. Resolve Retrieval/Context
    if context_file:
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                context = f.read()
        except Exception as e:
            print("Failed to read context file:", e)
            return

    # If no manual context provided, try to retrieve from Vector DB
    if not context:
        print("Retrieving context from vector store...")
        try:
            embeddings = get_embeddings(EMBEDDING_MODEL)
            # Explicitly match the collection name used in ingest (defaults to "langchain")
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name="langchain")
            
            # DEBUG: Check if collection has documents
            try:
                count = db._collection.count()
                print(f"DEBUG: ChromaDB collection has {count} documents")
            except Exception as e:
                print(f"DEBUG: Could not get collection count: {e}")

            # Retrieve top k documents
            docs = db.similarity_search(question, k=TOP_K)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                print(f"Retrieved {len(docs)} documents.")
            else:
                print("No relevant documents found in vector store.")
                # Debugging: maybe print all docs?
                # print("DEBUG: All docs count:", db._collection.count()) 
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # fall through to check (if not context)

    if not context:
        print("No context available to answer the question.")
        return

    # 2. Generate Answer
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        print(f"❌ Error: {GOOGLE_API_KEY_ENV} not found in environment variables.")
        print("Please set your API key in the .env file.")
        return
    
    try:
        chain = build_chain(LLM_MODEL, api_key)
        res = ask(chain, context, question)
        print("\n--- Result ---")
        print(res.content if hasattr(res, "content") else res) # Handle both str and AIMessage return types
        print("--------------\n")
    except Exception as e:
        print(f"❌ Generative Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple passthrough LLM CLI")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--path", required=True)

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("--q", required=True, help="Question to ask")
    p_ask.add_argument("--context", required=False, help="Context text to pass to the LLM")
    p_ask.add_argument("--context-file", required=False, help="Path to a text file containing context")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.path)
    elif args.cmd == "ask":
        chat(args.q, context=args.context, context_file=args.context_file)
