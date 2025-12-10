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
    TOP_K,
    RERANKER_MODEL,
    INITIAL_RETRIEVAL_K,
    FINAL_TOP_K
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

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Loading documents from: {data_path}")

    docs = load_documents(data_path)
    if not docs:
        logger.error("❌ No documents found to ingest.")
        return

    logger.info(f"✅ Loaded {len(docs)} documents")
    chunks = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"✅ Split into {len(chunks)} chunks")
    
    embeddings = get_embeddings(EMBEDDING_MODEL)
    store_to_chroma(chunks, PERSIST_DIR, embeddings)

    logger.info(f"✅ Successfully ingested {len(docs)} documents into ChromaDB!")


def chat(question, context=None, context_file=None):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Lazy import of chain so ingest doesn't require langchain/llm packages
    from src.rag.chain import build_chain, ask  # <<-- lazy import
    from langchain_community.vectorstores import Chroma
    from src.embeddings.hugging_face import get_embeddings
    from src.embeddings.reranker import rerank_documents  # Import reranker

    # 1. Resolve Retrieval/Context
    if context_file:
        try:
            with open(context_file, "r", encoding="utf-8") as f:
                context = f.read()
        except Exception as e:
            logger.error("Failed to read context file:", e)
            return

    # If no manual context provided, try to retrieve from Vector DB
    if not context:
        logger.info("Retrieving context from vector store...")
        try:
            embeddings = get_embeddings(EMBEDDING_MODEL)
            # Explicitly match the collection name used in ingest (defaults to "langchain")
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name="langchain")
            
            # DEBUG: Check if collection has documents
            try:
                count = db._collection.count()
                logger.info(f"DEBUG: ChromaDB collection has {count} documents")
            except Exception as e:
                logger.error(f"DEBUG: Could not get collection count: {e}")

            # Step 1: Retrieve more documents initially for reranking
            logger.info(f"Retrieving top {INITIAL_RETRIEVAL_K} documents for reranking...")
            initial_docs = db.similarity_search(question, k=INITIAL_RETRIEVAL_K)
            
            if initial_docs:
                logger.info(f"Retrieved {len(initial_docs)} documents, now reranking...")
                
                # Step 2: Rerank the documents
                docs = rerank_documents(
                    query=question,
                    documents=initial_docs,
                    top_k=FINAL_TOP_K,
                    model_name=RERANKER_MODEL
                )
                
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    logger.info(f"Using top {len(docs)} reranked documents for context.")
                else:
                    logger.info("No documents after reranking.")
            else:
                logger.info("No relevant documents found in vector store.")
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            # fall through to check (if not context)

    if not context:
        logger.error("No context available to answer the question.")
        return

    # 2. Generate Answer
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        logger.error(f"❌ Error: {GOOGLE_API_KEY_ENV} not found in environment variables.")
        logger.error("Please set your API key in the .env file.")
        return
    
    try:
        chain = build_chain(LLM_MODEL, api_key)
        res = ask(chain, context, question)
        logger.info("\n--- Result ---")
        logger.info(res.content if hasattr(res, "content") else res) # Handle both str and AIMessage return types
        logger.info("--------------\n")
    except Exception as e:
        logger.error(f"❌ Generative Error: {e}")
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
