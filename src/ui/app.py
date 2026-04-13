"""
Streamlit UI for Fin-Context RAG Engine
Interactive interface for financial document analysis
"""

import os
import streamlit as st
from dotenv import load_dotenv
import sys
import gc
import time
import shutil

# Bridge Streamlit Secrets to Environment Variables
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "WANDB_API_KEY" in st.secrets:
    os.environ["WANDB_API_KEY"] = st.secrets["WANDB_API_KEY"]

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.core import PDFProcessor, VectorStoreManager, RAGEngine
    from src.eval import RAGEvaluator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Fin-Context AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism styling
st.markdown("""
<style>
    body {
        background-color: #F8F9FA;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-citation {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .stChatMessage {
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 0.25rem 0;
    }
    .stChatMessage[data-testid="chat-message-container-user"] {
        background-color: #e3f2fd;
    }
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background-color: #ffffff;
    }
        padding: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ["GROQ_API_KEY", "GEMINI_API_KEY", "WANDB_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please set these variables in your .env file and restart the app.")
        return False
    return True

def initialize_session_state():
    """Initialize session state variables."""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    if 'processor' not in st.session_state:
        st.session_state.processor = PDFProcessor()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'rag_engine' not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.session_state.rag_engine = RAGEngine(groq_api_key)
    
    if 'evaluator' not in st.session_state:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        wandb_project = os.getenv("WANDB_PROJECT", "fin-context-rag")
        if wandb_api_key and gemini_api_key:
            st.session_state.evaluator = RAGEvaluator(wandb_project, wandb_api_key, gemini_api_key)
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_session_processed' not in st.session_state:
        st.session_state.current_session_processed = False
    
    if 'current_chunks_count' not in st.session_state:
        st.session_state.current_chunks_count = 0
    
    if 'current_docs_count' not in st.session_state:
        st.session_state.current_docs_count = 0
    
    if 'eval_history' not in st.session_state:
        st.session_state.eval_history = []

def evaluate_response(query, response, context):
    """Evaluate response using Gemini model as a Judge."""
    try:
        import google.generativeai as genai
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        
        # Create Judge prompt
        judge_prompt = f"""
You are an expert judge evaluating AI responses for faithfulness to context.

Query: {query}
Context: {context}
Response: {response}

Your task: Act as a judge and score how well this answer matches the provided context on a scale of 1-10.

Scoring criteria:
1 = Completely unfaithful (makes up information not in context)
10 = Completely faithful (only uses information from context)

Respond with ONLY a single number (1-10). No explanation needed.
"""
        
        result = model.generate_content(judge_prompt)
        score = int(result.text.strip())
        return max(1, min(10, score))  # Ensure score is between 1-10
    
    except Exception as e:
        print(f"Judge evaluation error: {e}")
        return 5  # Default score if evaluation fails

def sidebar_info():
    """Display sidebar information and controls."""
    with st.sidebar:
        # Branding at very top
        st.sidebar.markdown("### Context AI")
        st.sidebar.markdown("---")
        
        # File Upload Section
        st.header("📁 File Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        if uploaded_file:
            # Save the uploaded file
            file_path = os.path.join("data/raw", uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Saved {uploaded_file.name} to data/raw/")
            
            # Clear vector store to force reprocessing
            if 'processed' in st.session_state:
                st.session_state.processed = False
        
        if st.button("🔄 Process Documents", type="primary", key="sidebar_process_btn"):
            process_documents()
            st.session_state.processed = True
            st.session_state.current_session_processed = True
            # Store current session counts
            if st.session_state.get('vector_store') is not None:
                st.session_state.current_chunks_count = st.session_state.vector_store.get_document_count()
                # For now, use a default docs count - this could be enhanced
                st.session_state.current_docs_count = 3
            st.rerun()
        
        if st.button(" Clear Vector Store"):
            clear_vector_store()
            st.session_state.current_session_processed = False
            st.session_state.current_chunks_count = 0
            st.session_state.current_docs_count = 0
        
        # Green confirmation when processed in current session
        if st.session_state.get('current_session_processed', False):
            st.success(f"Processed {st.session_state.get('current_chunks_count', 0)} chunks from {st.session_state.get('current_docs_count', 0)} documents", icon="✅")
        
        st.divider()
        
        # Configuration Section
        st.header(" Configuration")
        st.session_state.chunk_size = st.slider(
            "Chunk Size", min_value=500, max_value=2000, value=1000, step=100, key='sidebar_chunk_size'
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap", min_value=0, max_value=500, value=200, step=50, key='sidebar_chunk_overlap'
        )
        st.session_state.retrieval_k = st.slider(
            "Retrieval K", min_value=3, max_value=10, value=5, step=1, key='sidebar_retrieval_k'
        )
        
        st.divider()
        
        # System Status Section
        st.header("📊 System Status")
        
        # Environment status
        env_status = check_environment()
        if env_status:
            st.success("✅ All environment variables set")
        
        # Vector DB Status indicator
        st.subheader("📊 Vector DB Status")
        if st.session_state.vector_db_online:
            st.success("🟢 Online")
            if st.session_state.vector_store:
                doc_count = st.session_state.vector_store.get_document_count()
                st.metric("Documents", doc_count)
        else:
            st.error("🔴 Offline")
        
                
        # Save uploaded file
        if uploaded_file is not None:
            # Ensure data/raw directory exists
            os.makedirs("data/raw", exist_ok=True)
            
            # Save the uploaded file
            file_path = os.path.join("data/raw", uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Saved {uploaded_file.name} to data/raw/")
            
            # Clear vector store to force reprocessing
            if 'processed' in st.session_state:
                st.session_state.processed = False
        
        st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #004C94;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
    font-weight: 600;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #0052CC;
}
</style>
""", unsafe_allow_html=True)
        
                
        
def process_documents():
    """Process PDF documents from data/raw directory."""
    try:
        with st.status("🔄 Processing documents...", expanded=True) as status:
            # Update processor with current settings
            status.write("📝 Initializing PDF processor...")
            st.session_state.processor = PDFProcessor(
                chunk_size=st.session_state.get('chunk_size', 1000),
                chunk_overlap=st.session_state.get('chunk_overlap', 200)
            )
            
            # Process documents
            status.write("📄 Reading PDF files from data/raw...")
            documents = st.session_state.processor.process_directory("data/raw")
            
            if documents:
                # Create vector store
                status.write("🔍 Creating vector embeddings...")
                vector_store_manager = VectorStoreManager()
                vector_store_manager.create_vectorstore(documents)
                st.session_state.vector_store = vector_store_manager
                st.session_state.processed = True
                
                # Get unique documents for better reporting
                unique_docs = len(set(doc.metadata.get('source', doc.metadata.get('filename', 'Unknown')) for doc in documents))
                status.update(label="", state="complete")
                
                # Store actual counts in session state
                st.session_state.current_chunks_count = len(documents)
                st.session_state.current_docs_count = unique_docs
                
                # Success toast - stays visible while UI updates
                st.toast('Documents processed successfully!', icon='')
                
                # Automatic rerun to update sidebar status
                st.rerun()
            else:
                status.update(label="", state="error")
                st.warning("")
                st.warning("⚠️ No PDF files found in data/raw directory. Please upload a PDF first.")
                
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")

def clear_vector_store():
    """Clear the vector store."""
    try:
        if st.session_state.vector_store:
            # Close the connection properly
            try:
                if hasattr(st.session_state.vector_store, '_client'):
                    st.session_state.vector_store._client.reset()
            except:
                pass
            
            # Set to None to release references
            st.session_state.vector_store = None
            
            # Force garbage collection
            gc.collect()
            
            # Wait for Windows to release file locks
            time.sleep(0.5)
            
            # Remove the chroma folder if it exists
            if os.path.exists('data/chroma'):
                shutil.rmtree('data/chroma', ignore_errors=True)
        
        # Reset session states
        st.session_state.processed = False
        st.session_state.vector_db_online = False
        st.session_state.current_session_processed = False
        st.session_state.current_chunks_count = 0
        st.session_state.current_docs_count = 0
        
        st.success("**Vector store cleared**")
    except Exception as e:
        st.error(f"**Error clearing vector store: {str(e)}**")

def display_chat_interface():
    """Display the main chat interface."""
    st.header("💬 Context Document Q&A")
    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                st.write(message["content"])
                
                # Display sources if available
                if "sources" in message:
                    st.subheader("📚 Sources")
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-citation">
                            <strong>Source:</strong> {source['filename']}<br>
                            <strong>Page:</strong> {source['page']}<br>
                            <strong>Content:</strong> {source['content'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input - enabled when processed is True or chroma exists
    if prompt := st.chat_input("Ask about your context documents..."):
        # Bypass stale processed flag if data/chroma folder exists
        if not st.session_state.processed and not os.path.exists('data/chroma'):
            st.warning("?? Please process documents first!")
            return
        
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Process query
        start_time = time.time()
        with st.spinner("Analyzing documents..."):
            try:
                # Verify database connection
                if st.session_state.vector_store is None:
                    st.error("?? Vector store not initialized. Please process documents first.")
                    return
                
                # Get RAG response
                result = st.session_state.rag_engine.query(
                    prompt, 
                    k=st.session_state.get('retrieval_k', 5)
                )
                
                # Debug print statement
                docs = result.get("retrieved_documents", [])
                print(f'DEBUG: Found {len(docs)} relevant chunks for this query')
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Add assistant response
                answer = result["answer"]
                docs = result.get("retrieved_documents", [])
                
                # Strict system prompt handling
                if len(docs) == 0:
                    answer = "I searched the database but found no matches for your query. Please try different keywords or ensure the documents contain relevant information."
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": [
                        {
                            "filename": doc.metadata.get('filename', 'Unknown'),
                            "page": doc.metadata.get('page', 'Unknown'),
                            "content": doc.page_content
                        }
                        for doc in docs
                    ]
                })
                
                # Track evaluation metrics
                eval_data = {
                    'timestamp': time.time(),
                    'response_time': response_time,
                    'chunks_retrieved': len(docs),
                    'query': prompt
                }
                
                # Evaluate faithfulness if enabled
                if st.session_state.get('enable_evaluation', False):
                    context_text = "\n".join([doc.page_content for doc in docs[:3]])  # Use first 3 chunks as context
                    faithfulness_score = evaluate_response(prompt, answer, context_text)
                    eval_data['faithfulness_score'] = faithfulness_score
                    
                    # Also log to W&B if evaluator exists
                    if 'evaluator' in st.session_state:
                        evaluate_query(result)
                
                # Add to evaluation history
                st.session_state.eval_history.append(eval_data)
                
                # Rerun to display new message
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

def evaluate_query(result):
    """Evaluate query and log to W&B."""
    try:
        if 'evaluator' in st.session_state:
            scores = st.session_state.evaluator.evaluate_and_log(
                result,
                experiment_name=f"streamlit_query_{len(st.session_state.chat_history)}"
            )
            
            # Display evaluation metrics
            with st.expander("📊 Evaluation Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Context Precision", f"{scores.get('context_precision', 0):.3f}")
                    st.metric("Faithfulness", f"{scores.get('faithfulness', 0):.3f}")
                with col2:
                    st.metric("Answer Relevancy", f"{scores.get('answer_relevancy', 0):.3f}")
                    st.metric("Context Recall", f"{scores.get('context_recall', 0):.3f}")
                    
    except Exception as e:
        st.warning(f" Evaluation failed: {str(e)}")

def main():
    """Main application function."""
    # Physical check for database
    if os.path.exists('data/chroma') and any(os.scandir('data/chroma')):
        st.session_state.processed = True
        st.session_state.vector_db_online = True
    else:
        st.session_state.processed = False
        st.session_state.vector_db_online = False
    
    # Professional Banking Blue CSS theme
    st.markdown("""
    <style>
    /* Primary Banking Blue theme */
    :root {
        --primary-color: #004C94;
        --primary-hover: #0052CC;
        --border-radius: 10px;
    }
    
    body {
        background-color: #F8F9FA;
    }
    
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    /* Rounded containers and buttons */
    .stButton > button {
        border-radius: var(--border-radius) !important;
    }
    
    div[data-testid="stButton"] > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--border-radius) !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 8px rgba(0, 76, 148, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: var(--primary-hover) !important;
        box-shadow: 0 6px 12px rgba(0, 76, 148, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Chat bubbles with subtle shadows */
    .stChatMessage {
        border-radius: var(--border-radius) !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1) !important;
        margin: 0.5rem 0 !important;
        border: 1px solid rgba(0, 76, 148, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        color: #333 !important;
    }
    
    .stChatMessage[data-testid="chat-message-container-user"] {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
        border-left: 4px solid var(--primary-color) !important;
        color: white !important;
    }
    
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background: linear-gradient(135deg, #1C39BB, #1A2B4C) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #1C39BB !important;
        color: #F0F2F6 !important;
        box-shadow: 0 8px 24px rgba(28, 57, 187, 0.3) !important;
        position: relative !important;
    }
    
    .stChatMessage[data-testid="chat-message-container-assistant"]::before {
        content: '' !important;
        position: absolute !important;
        top: -2px !important;
        left: -2px !important;
        right: -2px !important;
        bottom: -2px !important;
        background: linear-gradient(135deg, rgba(28, 57, 187, 0.2), rgba(26, 43, 76, 0.2)) !important;
        border-radius: var(--border-radius) !important;
        z-index: -1 !important;
    }
    
    .stChatMessage[data-testid="chat-message-container-assistant"] p,
    .stChatMessage[data-testid="chat-message-container-assistant"] div {
        color: #F0F2F6 !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stChatMessage[data-testid="chat-message-container-assistant"] strong {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Source citation styling */
    .source-citation {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        margin: 8px 0 !important;
        font-size: 0.85em !important;
        color: #A8B2D1 !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .source-citation strong {
        color: #D4D8E8 !important;
        font-weight: 600 !important;
    }
    
    /* Persian Blue styling for chat messages */
    div[data-testid="stChatMessage"] {
        background-color: #1C39BB !important;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li {
        color: #FFFFFF !important;
        font-weight: 400;
    }
    
    /* Sidebar containers */
    .css-1d391kg {
        border-radius: var(--border-radius) !important;
    }
    
    .stSelectbox > div > div {
        border-radius: var(--border-radius) !important;
    }
    
    .stSlider > div > div {
        border-radius: var(--border-radius) !important;
    }
    
    /* Status indicator */
    .status-online {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: var(--border-radius);
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 76, 148, 0.3);
    }
    
    /* Metric cards */
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 76, 148, 0.1);
    }
    
    </style>
    """, unsafe_allow_html=True)
    
        
    # Main Header with a reliable URL logo
    col1, col2 = st.columns([1, 5])
    with col1:
        # This is a clean, professional AI document icon
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=70) 
    with col2:
        st.title("Insight Context AI")
    
    # --- 3. APP LOGIC ---
    if not check_environment():
        st.stop()

    initialize_session_state()
    sidebar_info()

    # Tabs with fixed emojis
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

    with tab1:
        display_chat_interface()

    with tab2:
        st.header("📊 Analytics Dashboard")
        
        st.session_state.enable_evaluation = st.checkbox(
            "Enable Response Evaluation", 
            value=st.session_state.get('enable_evaluation', False)
        )
        
        if st.session_state.eval_history:
            avg_time = sum(item.get('response_time', 0) for item in st.session_state.eval_history) / len(st.session_state.eval_history)
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            
            if st.session_state.enable_evaluation:
                scores = [i.get('faithfulness_score', 0) for i in st.session_state.eval_history if 'faithfulness_score' in i]
                if scores:
                    st.subheader("📈 Faithfulness Score Trend")
                    st.line_chart(scores)
            
            st.subheader("🔍 Retrieval Statistics")
            counts = [i.get('chunks_retrieved', 0) for i in st.session_state.eval_history]
            if counts:
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Chunks", f"{sum(counts)/len(counts):.1f}")
                c2.metric("Min", min(counts))
                c3.metric("Max", max(counts))
        else:
            st.info("📝 No chat interactions yet.")

if __name__ == "__main__":
    main()