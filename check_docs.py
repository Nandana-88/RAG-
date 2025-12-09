import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from src.embeddings.hugging_face import get_embeddings
from src.config import PERSIST_DIR, EMBEDDING_MODEL

load_dotenv()

def check_source_files(data_path="./data"):
    """Count all PDF files in the data directory."""
    data_dir = Path(data_path)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_path}")
        return []
    
    pdf_files = list(data_dir.rglob("*.pdf"))
    print(f"\n📁 Source Files in {data_path}:")
    print(f"   Total PDF files: {len(pdf_files)}")
    
    for pdf in sorted(pdf_files):
        size_kb = pdf.stat().st_size / 1024
        print(f"   - {pdf.relative_to(data_dir)} ({size_kb:.1f} KB)")
    
    return pdf_files

def check_chroma_db():
    """Check ChromaDB collection and stored documents."""
    if not os.path.exists(PERSIST_DIR):
        print(f"\n❌ ChromaDB directory not found: {PERSIST_DIR}")
        print("   Run: python main.py ingest --path ./data")
        return None
    
    print(f"\n💾 ChromaDB Status:")
    try:
        embeddings = get_embeddings(EMBEDDING_MODEL)
        db = Chroma(
            persist_directory=PERSIST_DIR, 
            embedding_function=embeddings, 
            collection_name="langchain"
        )
        
        # Get collection count
        count = db._collection.count()
        print(f"   Total chunks stored: {count}")
        
        # Get all documents with metadata
        all_docs = db.get()
        
        # Extract unique source files
        sources = set()
        if 'metadatas' in all_docs and all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                if metadata and 'source' in metadata:
                    sources.add(metadata['source'])
        
        print(f"   Unique source documents: {len(sources)}")
        
        if sources:
            print(f"\n📄 Stored Documents:")
            for source in sorted(sources):
                # Count chunks per source
                chunk_count = sum(1 for m in all_docs['metadatas'] 
                                if m and m.get('source') == source)
                print(f"   - {Path(source).name} ({chunk_count} chunks)")
        
        return db, all_docs, sources
    
    except Exception as e:
        print(f"   ❌ Error accessing ChromaDB: {e}")
        return None

def compare_files_vs_db(pdf_files, stored_sources):
    """Compare source files with stored documents."""
    if not pdf_files or not stored_sources:
        return
    
    print(f"\n🔍 Comparison:")
    
    # Convert stored sources to just filenames for comparison
    stored_filenames = {Path(s).name for s in stored_sources}
    source_filenames = {pdf.name for pdf in pdf_files}
    
    missing_in_db = source_filenames - stored_filenames
    extra_in_db = stored_filenames - source_filenames
    
    if not missing_in_db and not extra_in_db:
        print(f"   ✅ All {len(pdf_files)} PDF files are loaded in ChromaDB!")
    else:
        if missing_in_db:
            print(f"   ⚠️  Files NOT in database ({len(missing_in_db)}):")
            for filename in sorted(missing_in_db):
                print(f"      - {filename}")
        
        if extra_in_db:
            print(f"   ⚠️  Files in database but not in data folder ({len(extra_in_db)}):")
            for filename in sorted(extra_in_db):
                print(f"      - {filename}")

def test_retrieval(db, test_query="fee structure"):
    """Test if retrieval is working."""
    if not db:
        return
    
    print(f"\n🧪 Test Retrieval (query: '{test_query}'):")
    try:
        docs = db.similarity_search(test_query, k=3)
        print(f"   Retrieved {len(docs)} documents")
        
        if docs:
            print(f"\n   Top result preview:")
            preview = docs[0].page_content[:200].replace('\n', ' ')
            print(f"   {preview}...")
            print(f"   Source: {docs[0].metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Retrieval error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("📊 RAG SYSTEM DOCUMENT CHECK")
    print("=" * 60)
    
    # Step 1: Check source files
    pdf_files = check_source_files("./data")
    
    # Step 2: Check ChromaDB
    result = check_chroma_db()
    
    if result:
        db, all_docs, sources = result
        
        # Step 3: Compare
        compare_files_vs_db(pdf_files, sources)
        
        # Step 4: Test retrieval
        test_retrieval(db, "fee structure")
    
    print("\n" + "=" * 60)
    print("✅ Check complete!")
    print("=" * 60)
