import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the chat function from main
from src.config import PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL, GOOGLE_API_KEY_ENV, TOP_K

# Page configuration
st.set_page_config(
    page_title="University of Kerala Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ChatGPT-style styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #ffffff;
        padding: 0;
    }
    
    /* Hide default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1.5rem 0;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background-color: #f7f7f8;
        border-radius: 12px;
        padding: 12px 16px;
        max-width: 800px;
    }
    
    /* Assistant message - different background */
    .stChatMessage:has([data-testid="stChatMessageContent"]:first-child) {
        background-color: #ffffff;
    }
    
    /* Input container at bottom */
    .stChatInputContainer {
        border-top: 1px solid #e5e5e5;
        padding: 1rem 0;
        background-color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background-color: white;
        color: #374151;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #f9fafb;
        border-color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# Cache the embeddings model (loads only once)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    from src.embeddings.hugging_face import get_embeddings as get_emb
    return get_emb(EMBEDDING_MODEL)

# Cache the vector database connection (loads only once)
@st.cache_resource(show_spinner=False)
def get_vector_db():
    from langchain_community.vectorstores import Chroma
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=embeddings, 
        collection_name="langchain"
    )

# Cache the RAG chain (loads only once)
@st.cache_resource(show_spinner=False)
def get_rag_chain():
    from src.rag.chain import build_chain
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{GOOGLE_API_KEY_ENV} not found in environment")
    return build_chain(LLM_MODEL, api_key)

# Conversational RAG function with chat history
def chat_with_rag(question, chat_history):
    """Fast conversational chat function with cached models"""
    try:
        # Use cached database and chain
        db = get_vector_db()
        chain = get_rag_chain()
        
        # Retrieve context
        docs = db.similarity_search(question, k=TOP_K)
        
        if not docs:
            return "I couldn't find relevant information in the knowledge base."
        
        context = "\n\n".join([d.page_content for d in docs])
        
        # Build conversation history for context
        conversation_context = ""
        if chat_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in chat_history[-4:]:  # Last 2 exchanges (4 messages)
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Enhanced prompt with conversation history
        full_context = f"{context}\n{conversation_context}"
        
        # Generate response
        from src.rag.chain import ask
        result = ask(chain, full_context, question)
        
        return result.content if hasattr(result, "content") else str(result)
        
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    # Logo and title in sidebar
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)
    
    st.title("University of Kerala")
    st.caption("AI Assistant")
    
    st.divider()
    
    st.subheader("About")
    st.markdown("""
    Ask me about:
    - 📚 Courses & Programs
    - 💰 Fee Structure
    - 📅 Academic Calendar
    - 🎓 Admissions
    - 📍 Campus Info
    """)
    
    st.divider()
    
    # Database status
    if os.path.exists(PERSIST_DIR):
        st.success("✅ Knowledge base ready")
        # Show number of messages in conversation
        if st.session_state.messages:
            msg_count = len(st.session_state.messages) // 2
            st.info(f"💬 {msg_count} messages in conversation")
    else:
        st.error("❌ Knowledge base not found")
        st.code("python main.py ingest --path ./data", language="bash")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.markdown("### 💬 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message University of Kerala Assistant..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass chat history for conversational context
            result = chat_with_rag(prompt, st.session_state.messages[:-1])
            st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
