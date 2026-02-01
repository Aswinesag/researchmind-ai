"""
Research RAG Assistant - Main Streamlit Application
AI-powered research paper analysis with RAG
"""
import streamlit as st
import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import config
from src.document_processor import DocumentProcessor
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingGenerator, ChunkEmbedder
from src.vector_store import VectorStore
from src.rag_chain import ConversationRAG
from src.citation_handler import CitationHandler
from src.search.search_manager import SearchManager
from src.download.pdf_downloader import PDFDownloader
from src.utils import save_json, load_json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="ResearchMind AI",
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout=config.STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Dark theme text colors */
    .main .block-container {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4fc3f7;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #b0bec5;
        margin-bottom: 2rem;
    }
    
    /* Paper cards with dark theme */
    .paper-card {
        background-color: #2d2d2d;
        border-left: 4px solid #4fc3f7;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #ffffff;
    }
    
    /* Source badges */
    .source-badge {
        background-color: #1e3a5f;
        color: #ffffff;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    
    /* All text elements - white for dark theme */
    p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
        border: 1px solid #444 !important;
    }
    
    /* Buttons */
    .stButton > button {
        color: #ffffff !important;
        background-color: #1976d2 !important;
        border: 1px solid #1976d2 !important;
    }
    
    /* Sidebar elements */
    .css-1d391kg {
        background-color: #252526 !important;
    }
    
    /* Metrics and selectboxes */
    .stMetric {
        color: #ffffff !important;
    }
    
    .stSelectbox > div > div > select {
        color: #ffffff !important;
        background-color: #2d2d2d !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b0bec5 !important;
    }
    
    /* Form elements */
    .stForm {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store = None
        st.session_state.rag_chain = None
        st.session_state.embedding_generator = None
        st.session_state.papers_metadata = []
        st.session_state.chat_history = []
        st.session_state.current_paper_filter = None
        
        logger.info("Session state initialized")


def load_components():
    """Load and cache heavy components"""
    if not st.session_state.initialized:
        with st.spinner("🔧 Initializing AI components..."):
            try:
                # Load embedding model
                if st.session_state.embedding_generator is None:
                    st.session_state.embedding_generator = EmbeddingGenerator(
                        model_name=config.EMBEDDING_MODEL
                    )
                    st.session_state.embedding_generator.load_model()
                
                # Initialize vector store
                if st.session_state.vector_store is None:
                    embedding_dim = st.session_state.embedding_generator.get_embedding_dim()
                    st.session_state.vector_store = VectorStore(
                        embedding_dim=embedding_dim,
                        index_path=str(config.FAISS_INDEX_DIR)
                    )
                    
                    # Try to load existing index
                    try:
                        st.session_state.vector_store.load()
                        logger.info("Loaded existing vector store")
                    except:
                        logger.info("No existing vector store found")
                
                # Initialize RAG chain
                if st.session_state.rag_chain is None:
                    st.session_state.rag_chain = ConversationRAG(
                        vector_store=st.session_state.vector_store,
                        embedding_generator=st.session_state.embedding_generator,
                        groq_api_key=config.GROQ_API_KEY,
                        model_name=config.GROQ_MODEL,
                        top_k=config.TOP_K_RETRIEVAL
                    )
                
                st.session_state.initialized = True
                logger.info("All components initialized successfully")
                
            except Exception as e:
                st.error(f"Error initializing components: {e}")
                logger.error(f"Initialization error: {e}")
                st.stop()


def sidebar():
    """Render sidebar with improved organization"""
    with st.sidebar:
        # App header with container
        with st.container():
            st.markdown("# 🧠 ResearchMind AI")
            st.markdown("*Your Intelligent Research Companion*")
        
        st.markdown("---")
        
        # System status container
        with st.container():
            st.markdown("### 📊 System Status")
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Papers", stats['unique_papers'])
                with col2:
                    st.metric("📝 Chunks", stats['total_chunks'])
                
                # Status indicator
                if stats['total_chunks'] > 0:
                    st.success("🟢 System Ready")
                else:
                    st.warning("🟡 No Papers")
            else:
                st.error("🔴 Not Initialized")
        
        st.markdown("---")
        
        # Navigation container
        with st.container():
            st.markdown("### 🧭 Navigation")
            mode = st.selectbox(
                "Choose Mode",
                ["💬 Chat", "📤 Upload Papers", "🔍 Search Papers", "📚 Library"],
                key="mode",
                index=0
            )
        
        st.markdown("---")
        
        # Settings container with better organization
        with st.container():
            st.markdown("### ⚙️ Settings")
            
            with st.expander("🔍 Filter Options", expanded=False):
                st.session_state.current_paper_filter = st.selectbox(
                    "Filter by Paper",
                    ["All Papers"] + [p.get('title', 'Unknown')[:50] 
                                      for p in st.session_state.papers_metadata],
                    key="paper_filter"
                )
            
            with st.expander("🛠️ Actions", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
                        st.session_state.chat_history = []
                        if st.session_state.rag_chain:
                            st.session_state.rag_chain.clear_history()
                        st.success("Chat cleared!")
                        st.rerun()
                
                with col2:
                    if st.button("💾 Save", use_container_width=True, key="save_index"):
                        if st.session_state.vector_store:
                            st.session_state.vector_store.save()
                            st.success("Index saved!")
        
        st.markdown("---")
        
        # Footer
        with st.container():
            st.markdown("---")
            st.markdown("<div style='text-align: center; font-size: 0.8em; color: #666;'>Made with ❤️ as ResearchMind AI</div>", 
                       unsafe_allow_html=True)
        
        return mode


def chat_interface():
    """Chat interface with improved layout"""
    # Header container
    with st.container():
        st.markdown('<div class="main-header">💬 Chat with Papers</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Ask questions about your research papers</div>', unsafe_allow_html=True)
    
    # Status container
    with st.container():
        if st.session_state.vector_store and st.session_state.vector_store.get_stats()['total_chunks'] == 0:
            st.warning("⚠️ No papers indexed yet. Please upload or search for papers first.")
            return
        elif st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            st.info(f"📚 Ready to answer questions from {stats['unique_papers']} papers ({stats['total_chunks']} chunks)")
    
    # Chat container
    with st.container():
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 View {len(message['sources'])} sources"):
                        citation_handler = CitationHandler()
                        st.markdown(citation_handler.create_source_summary(message["sources"]))
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your papers..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing papers..."):
                    try:
                        # Apply paper filter if selected
                        filter_by = None
                        if st.session_state.current_paper_filter != "All Papers":
                            # Find paper by title
                            for paper in st.session_state.papers_metadata:
                                if paper.get('title', '')[:50] == st.session_state.current_paper_filter:
                                    filter_by = {'paper_id': paper['paper_id']}
                                    break
                        
                        # Query RAG
                        result = st.session_state.rag_chain.query_with_history(
                            question=prompt,
                            top_k=config.TOP_K_RETRIEVAL,
                            filter_by=filter_by
                        )
                        
                        answer = result['answer']
                        sources = result.get('sources', [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Display sources
                        if sources:
                            with st.expander(f"📚 View {len(sources)} sources"):
                                citation_handler = CitationHandler()
                                st.markdown(citation_handler.create_source_summary(sources))
                        
                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        logger.error(f"Chat error: {e}")


def upload_interface():
    """Upload interface with improved layout"""
    # Header container
    with st.container():
        st.markdown('<div class="main-header">📤 Upload Papers</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload PDF research papers to analyze</div>', unsafe_allow_html=True)
    
    # Upload section with tabs
    with st.container():
        tab1, tab2 = st.tabs(["📁 Upload Files", "📋 Upload Guide"])
        
        with tab1:
            # File upload with better styling
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload research papers in PDF format (Max 50MB per file)",
                key="pdf_uploader"
            )
            
            if uploaded_files:
                # File info container
                with st.expander(f"📄 Selected Files ({len(uploaded_files)})", expanded=True):
                    for i, file in enumerate(uploaded_files):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"📄 {file.name}")
                        with col2:
                            st.write(f"{file.size / 1024 / 1024:.1f} MB")
                        with col3:
                            if st.button("❌", key=f"remove_{i}", help="Remove file"):
                                uploaded_files.pop(i)
                                st.rerun()
                
                # Process button with better styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Process Papers", type="primary", use_container_width=True):
                        process_uploaded_papers(uploaded_files)
        
        with tab2:
            # Upload guide
            st.markdown("""
            ### 📋 Upload Guidelines
            
            **Supported Formats:**
            - PDF files only
            - Maximum size: 50MB per file
            - Multiple files supported
            
            **Recommended Papers:**
            - Academic research papers
            - Scientific articles
            - Conference papers
            - Technical reports
            
            **Processing Steps:**
            1. 📁 Select PDF files
            2. 🔍 Text extraction and chunking
            3. 🧠 AI-powered embedding generation
            4. 💾 Automatic indexing
            
            **Tips:**
            - Clear file names improve organization
            - Papers with abstracts work best
            - Recent papers provide better context
            """)


def process_uploaded_papers(uploaded_files):
    """Process uploaded PDF papers"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processor = DocumentProcessor()
    chunker = TextChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunk_embedder = ChunkEmbedder(st.session_state.embedding_generator)
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((idx + 1) / total_files)
            
            # Save uploaded file temporarily
            temp_path = config.UPLOADS_DIR / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process document
            result = processor.process_document(str(temp_path))
            
            if result['success']:
                # Chunk text
                chunks = chunker.chunk_with_metadata(
                    result['text'],
                    result['metadata']
                )
                
                # Embed chunks
                embedded_chunks = chunk_embedder.embed_chunks(chunks)
                
                # Add to vector store
                st.session_state.vector_store.add_chunks(embedded_chunks)
                
                # Save metadata
                st.session_state.papers_metadata.append(result['metadata'])
                
                st.success(f"✅ Processed: {result['metadata']['title'][:60]}...")
            else:
                st.error(f"❌ Failed: {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            logger.error(f"Upload processing error: {e}")
    
    # Save vector store
    st.session_state.vector_store.save()
    
    progress_bar.progress(1.0)
    status_text.text("✅ All papers processed!")
    st.balloons()


def search_interface():
    """Search interface for finding papers"""
    st.markdown('<div class="main-header">🔍 Search Papers</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search arXiv, Semantic Scholar, and PubMed</div>', unsafe_allow_html=True)
    
    # Search form
    with st.form("search_form"):
        query = st.text_input("Search Query", placeholder="e.g., transformer attention mechanisms")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_results = st.number_input("Max Results", min_value=5, max_value=50, value=15)
        with col2:
            year_start = st.number_input("From Year", min_value=2000, max_value=2024, value=2020)
        with col3:
            year_end = st.number_input("To Year", min_value=2000, max_value=2024, value=2024)
        
        sources = st.multiselect(
            "Sources",
            ["arxiv", "semantic_scholar", "pubmed"],
            default=["arxiv", "semantic_scholar"]
        )
        
        submitted = st.form_submit_button("🔍 Search", type="primary")
    
    if submitted and query:
        search_papers(query, max_results, (year_start, year_end), sources)


def search_papers(query, max_results, year_range, sources):
    """Execute paper search"""
    with st.spinner("🔍 Searching papers..."):
        try:
            manager = SearchManager(
                arxiv_enabled="arxiv" in sources,
                semantic_scholar_enabled="semantic_scholar" in sources,
                pubmed_enabled="pubmed" in sources,
                semantic_scholar_api_key=config.SEMANTIC_SCHOLAR_API_KEY,
                pubmed_email=config.PUBMED_EMAIL
            )
            
            papers = manager.search_and_aggregate(
                query=query,
                sources=sources,
                max_results=max_results,
                year_range=year_range,
                rank_by='citations'
            )
            
            if papers:
                st.success(f"Found {len(papers)} papers!")
                
                # Store in session state
                st.session_state.search_results = papers
                
                # Display results
                display_search_results(papers)
            else:
                st.warning("No papers found. Try adjusting your search criteria.")
        
        except Exception as e:
            st.error(f"Search error: {e}")
            logger.error(f"Search error: {e}")


def display_search_results(papers):
    """Display search results with download option"""
    citation_handler = CitationHandler()
    
    # Select papers to download
    selected_papers = []
    
    for i, paper in enumerate(papers):
        with st.expander(f"📄 {paper['title'][:80]}..."):
            st.markdown(citation_handler.format_paper_card(paper))
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.checkbox("Select", key=f"select_{i}"):
                    selected_papers.append(paper)
    
    # Download selected papers
    if selected_papers:
        st.markdown("---")
        if st.button(f"📥 Download & Index {len(selected_papers)} Selected Papers", type="primary"):
            download_and_index_papers(selected_papers)


def download_and_index_papers(papers):
    """Download and index papers"""
    downloader = PDFDownloader(
        download_dir=str(config.FETCHED_PDFS_DIR),
        max_retries=config.MAX_RETRY_ATTEMPTS
    )
    
    with st.spinner("📥 Downloading papers..."):
        downloaded_papers = downloader.download_batch(
            papers,
            max_concurrent=config.MAX_CONCURRENT_DOWNLOADS
        )
    
    # Process successful downloads
    successful = [p for p in downloaded_papers if p.get('download_status') == 'success']
    
    if successful:
        with st.spinner("🔄 Processing and indexing papers..."):
            processor = DocumentProcessor()
            chunker = TextChunker(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunk_embedder = ChunkEmbedder(st.session_state.embedding_generator)
            
            for paper in successful:
                try:
                    # Process PDF
                    result = processor.process_document(paper['local_path'])
                    
                    if result['success']:
                        # Update metadata with search info
                        result['metadata'].update({
                            'source': paper.get('source'),
                            'citations_count': paper.get('citations_count'),
                            'doi': paper.get('doi'),
                            'arxiv_id': paper.get('arxiv_id')
                        })
                        
                        # Chunk
                        chunks = chunker.chunk_with_metadata(
                            result['text'],
                            result['metadata']
                        )
                        
                        # Embed
                        embedded_chunks = chunk_embedder.embed_chunks(chunks)
                        
                        # Index
                        st.session_state.vector_store.add_chunks(embedded_chunks)
                        st.session_state.papers_metadata.append(result['metadata'])
                
                except Exception as e:
                    logger.error(f"Error processing downloaded paper: {e}")
        
        # Save
        st.session_state.vector_store.save()
        
        st.success(f"✅ Successfully downloaded and indexed {len(successful)} papers!")
        st.balloons()
    else:
        st.warning("No papers were successfully downloaded.")


def library_interface():
    """Library interface with improved layout"""
    # Header container
    with st.container():
        st.markdown('<div class="main-header">📚 Paper Library</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Browse and manage your indexed papers</div>', unsafe_allow_html=True)
    
    if not st.session_state.papers_metadata:
        # Empty state container
        with st.container():
            st.info("📭 No papers in library yet. Upload or search for papers to get started!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📤 Upload Papers", use_container_width=True):
                    st.session_state.mode = "📤 Upload Papers"
                    st.rerun()
            with col2:
                if st.button("🔍 Search Papers", use_container_width=True):
                    st.session_state.mode = "🔍 Search Papers"
                    st.rerun()
            with col3:
                if st.button("💬 Start Chat", use_container_width=True):
                    st.session_state.mode = "💬 Chat"
                    st.rerun()
        return
    
    # Library stats container
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Papers", len(st.session_state.papers_metadata))
        with col2:
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                st.metric("📝 Total Chunks", stats['total_chunks'])
        with col3:
            # Calculate average year if available
            years = [p.get('year', 0) for p in st.session_state.papers_metadata if p.get('year')]
            avg_year = sum(years) / len(years) if years else 0
            st.metric("📅 Avg Year", f"{avg_year:.0f}" if avg_year else "N/A")
        with col4:
            # Count unique sources
            sources = set(p.get('source', 'Unknown') for p in st.session_state.papers_metadata)
            st.metric("🔗 Sources", len(sources))
    
    # Search and filter container
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("🔍 Search papers by title or author...", placeholder="Enter keywords...")
        with col2:
            sort_by = st.selectbox("📊 Sort by", ["Title", "Year (Newest)", "Year (Oldest)", "Source"])
    
    # Papers container with tabs
    with st.container():
        citation_handler = CitationHandler()
        
        # Filter papers based on search
        filtered_papers = st.session_state.papers_metadata
        if search_term:
            filtered_papers = [
                p for p in filtered_papers 
                if search_term.lower() in p.get('title', '').lower() 
                or search_term.lower() in ' '.join(p.get('authors', [])).lower()
            ]
        
        # Sort papers
        if sort_by == "Title":
            filtered_papers.sort(key=lambda x: x.get('title', ''))
        elif sort_by == "Year (Newest)":
            filtered_papers.sort(key=lambda x: x.get('year', 0), reverse=True)
        elif sort_by == "Year (Oldest)":
            filtered_papers.sort(key=lambda x: x.get('year', 9999))
        elif sort_by == "Source":
            filtered_papers.sort(key=lambda x: x.get('source', ''))
        
        if not filtered_papers:
            st.warning("🔍 No papers match your search criteria.")
        else:
            st.info(f"📚 Showing {len(filtered_papers)} of {len(st.session_state.papers_metadata)} papers")
            
            # Display papers in a cleaner format
            for i, paper in enumerate(filtered_papers):
                with st.expander(f"📄 {paper.get('title', 'Unknown')[:80]}...", expanded=False):
                    # Paper details in columns
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(citation_handler.format_paper_card(paper))
                    
                    with col2:
                        # Action buttons
                        st.markdown("**Actions:**")
                        if st.button("📋 Copy Citation", key=f"cite_{paper['paper_id']}", use_container_width=True):
                            citation = citation_handler.format_ieee_citation(paper)
                            st.code(citation)
                        
                        if st.button("🗑️ Remove", key=f"del_{paper['paper_id']}", use_container_width=True):
                            # Remove from vector store
                            st.session_state.vector_store.delete_paper(paper['paper_id'])
                            # Remove from metadata
                            st.session_state.papers_metadata = [
                                p for p in st.session_state.papers_metadata 
                                if p['paper_id'] != paper['paper_id']
                            ]
                            st.success("Paper removed!")
                            st.rerun()


def main():
    """Main application with improved layout"""
    # Initialize
    init_session_state()
    
    # Load components
    load_components()
    
    # Get sidebar (which includes navigation)
    mode = sidebar()
    
    # Main layout with tabs
    st.markdown("---")
    
    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Upload", "🔍 Search", "📚 Library"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        upload_interface()
    
    with tab3:
        search_interface()
    
    with tab4:
        library_interface()


if __name__ == "__main__":
    main()