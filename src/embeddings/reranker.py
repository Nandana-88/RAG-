# src/embeddings/reranker.py
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

# Global cache for the reranker model
_reranker_model = None

def get_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Load and cache the cross-encoder reranker model.
    
    Args:
        model_name: Name of the cross-encoder model to use
        
    Returns:
        CrossEncoder model instance
    """
    global _reranker_model
    
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {model_name}")
        _reranker_model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully")
    
    return _reranker_model


def rerank_documents(query, documents, top_k=5, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Rerank documents based on their relevance to the query using a cross-encoder.
    
    Args:
        query: The search query string
        documents: List of document objects (must have .page_content attribute)
        top_k: Number of top documents to return after reranking
        model_name: Name of the cross-encoder model to use
        
    Returns:
        List of top_k most relevant documents, sorted by relevance score
    """
    if not documents:
        logger.warning("No documents to rerank")
        return []
    
    # Get the reranker model
    reranker = get_reranker(model_name)
    
    # Prepare query-document pairs for scoring
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Get relevance scores
    logger.info(f"Reranking {len(documents)} documents...")
    scores = reranker.predict(pairs)
    
    # Combine documents with their scores
    doc_score_pairs = list(zip(documents, scores))
    
    # Sort by score (descending) and take top_k
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
    
    # Log the scores for debugging
    logger.info(f"Top {top_k} reranked document scores:")
    for i, (doc, score) in enumerate(doc_score_pairs[:top_k], 1):
        preview = doc.page_content[:100].replace('\n', ' ')
        logger.info(f"  {i}. Score: {score:.4f} | Preview: {preview}...")
    
    return reranked_docs
