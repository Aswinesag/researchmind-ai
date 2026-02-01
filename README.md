# ResearchMind AI 📚🤖

ResearchMind AI is an intelligent research companion that helps you analyze academic papers using advanced AI. Upload papers or search for them from multiple sources (arXiv, Semantic Scholar, PubMed), then ask questions and get insights with proper IEEE citations.

## ✨ Key Features

### 🎯 Core Functionality
- **📤 Upload & Analyze**: Upload research papers (PDF) and ask intelligent questions
- **🔍 Multi-Source Search**: Search papers from arXiv, Semantic Scholar, and PubMed simultaneously
- **💬 Conversational AI**: Natural language Q&A with context-aware responses
- **📚 Paper Library**: Organize and manage your research collection
- **📝 IEEE Citations**: Automatic citation formatting with proper references

### 🚀 Advanced Features
- **🧠 Smart RAG Pipeline**: Advanced retrieval-augmented generation with context filtering
- **📊 Multi-Paper Analysis**: Compare and synthesize information across multiple papers
- **🎯 Semantic Search**: Find relevant papers using advanced embedding similarity
- **📈 Citation Analysis**: View citation counts and impact metrics
- **🔄 Conversation History**: Maintain context across multiple questions
- **📑 Auto-Summarization**: Generate paper summaries automatically
- **🔍 Advanced Filtering**: Filter by year, source, citation count, and custom criteria

### 🎨 Professional UI/UX
- **� Modern Interface**: Clean, responsive design with dark theme support
- **🗂️ Tabbed Navigation**: Organized workflow with dedicated sections (Chat, Upload, Search, Library)
- **� Real-time Metrics**: Live statistics and system status indicators
- **⚡ Quick Actions**: Fast access to common operations in sidebar
- **🎛️ Customizable Settings**: Flexible configuration options

## 🛠 Tech Stack

### Core Technologies
- **Frontend**: Streamlit (Modern web interface with tabbed navigation)
- **Embeddings**: Sentence Transformers (allenai/specter for scientific papers)
- **Vector Store**: FAISS (High-performance similarity search)
- **RAG Framework**: LangChain (Advanced RAG capabilities)
- **LLM**: Groq API (llama-3.3-70b-versatile - production model)

### APIs & Data Sources
- **arXiv**: Open access scientific papers
- **Semantic Scholar**: Academic paper database with citations
- **PubMed**: Biomedical literature database
- **Groq Cloud**: High-speed LLM inference

### Processing Pipeline
- **PDF Processing**: Advanced text extraction and chunking
- **Semantic Chunking**: Intelligent text segmentation
- **Embedding Generation**: High-quality vector representations
- **Similarity Search**: Fast and accurate retrieval
- **Citation Formatting**: IEEE style automatic formatting

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd research-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys:
- **GROQ_API_KEY**: Get from [Groq Console](https://console.groq.com/)
- **SEMANTIC_SCHOLAR_API_KEY**: (Optional) Get from [Semantic Scholar](https://www.semanticscholar.org/product/api)
- **PUBMED_EMAIL**: Your email for PubMed API

### 5. Run the application
```bash
streamlit run app.py
```

## 📖 Usage Guide

### 📤 Upload & Analyze Papers
1. Navigate to the **"Upload"** tab
2. Select PDF research papers (multiple files supported)
3. Review selected files with size information in the file manager
4. Click **"Process Papers"** to analyze and index
5. Switch to **"Chat"** tab to ask questions

### 🔍 Search & Fetch Papers
1. Go to the **"Search"** tab
2. Enter search query (e.g., "transformer attention mechanisms")
3. Configure search parameters:
   - Max results (5-50)
   - Year range (2000-2024)
   - Data sources (arXiv, Semantic Scholar, PubMed)
4. Click **"Search"** to find papers
5. Review results and select papers to download
6. Papers are automatically processed and indexed

### 💬 Ask Questions
1. Use the **"Chat"** tab for Q&A
2. Ask questions about your uploaded papers
3. Get detailed answers with proper IEEE citations
4. View source papers for each answer
5. Maintain conversation context across multiple questions

### 📚 Manage Library
1. **"Library"** tab shows all indexed papers
2. **Search and filter** papers by title, author, or year
3. **Sort** papers by various criteria (title, year, source)
4. **Copy citations** in IEEE format
5. **Remove papers** from your collection

### 💡 Example Queries
- "What methodology did the authors use in their experiments?"
- "Summarize the key findings of all papers about transformers"
- "Compare the approaches across different papers on attention mechanisms"
- "What are the main limitations mentioned in these papers?"
- "How do these papers build upon previous work in the field?"
- "What datasets were used for evaluation?"
- "What are the future research directions suggested?"

## ⚙️ Configuration

### Customizable Settings (`config.py`)
- **Embedding Model**: allenai/specter (scientific paper optimized)
- **Chunk Size**: 800 tokens with 150 token overlap
- **Retrieval Count**: Top 5 most relevant chunks
- **Citation Style**: IEEE (configurable to APA, MLA)
- **Similarity Threshold**: 0.7 minimum similarity score
- **Rate Limits**: Configurable API request limits
- **Storage Paths**: Customizable data directories

### Advanced Options
- **Fallback Models**: Alternative embedding models
- **Processing Parameters**: Chunking strategies
- **API Settings**: Timeouts and retry attempts
- **UI Configuration**: Theme and layout options

## Project Structure

```
research-rag-assistant/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create from .env.example)
├── src/
│   ├── document_processor.py      # PDF processing
│   ├── embeddings.py              # Embedding generation
│   ├── vector_store.py            # FAISS vector store
│   ├── rag_chain.py               # RAG pipeline
│   ├── citation_handler.py        # Citation formatting
│   ├── search/                    # Paper search modules
│   │   ├── search_manager.py
│   │   ├── arxiv_searcher.py
│   │   ├── semantic_scholar_searcher.py
│   │   └── pubmed_searcher.py
│   └── download/                  # PDF download modules
│       └── pdf_downloader.py
├── data/
│   ├── uploads/                   # User uploaded papers
│   ├── fetched/                   # Auto-fetched papers
│   ├── processed/                 # Processed chunks
│   └── faiss_index/              # Vector store index
└── prompts/
    └── research_prompts.py       # Custom prompts
```

## 📊 API Rate Limits & Performance

### API Limits
- **arXiv**: 1 request per 3 seconds (no API key required)
- **Semantic Scholar**: 100 requests per 5 minutes (with API key)
- **PubMed**: 3 requests per second (10 with API key)
- **Groq**: High-speed inference with generous limits

### Performance Features
- **GPU Acceleration**: CUDA support for embeddings
- **Batch Processing**: Parallel paper processing
- **Caching**: Intelligent caching for faster responses
- **Optimized Search**: Fast similarity search with FAISS
- **Memory Management**: Efficient memory usage for large libraries

## 🔧 Troubleshooting

### Common Issues

#### "GROQ_API_KEY not set"
- Create `.env` file from `.env.example`
- Add your Groq API key: `GROQ_API_KEY=your_key_here`
- Restart the application

#### "PDF processing failed"
- Ensure PDF is not password-protected
- Check if PDF is text-based (not scanned images)
- Verify PDF is not corrupted
- Try with a different PDF file

#### "Embedding model download slow"
- First run downloads the model (~500MB for specter)
- Subsequent runs use cached model
- Check internet connection
- Consider using alternative embedding models

#### "No papers found in search"
- Broaden search terms
- Check year range settings
- Enable all data sources
- Verify API keys for restricted sources

#### "Memory issues with large libraries"
- Reduce chunk size in config
- Use smaller embedding models
- Clear unused papers from library
- Increase system RAM if possible

### Performance Optimization
- **Use GPU**: Ensure CUDA is available for faster embeddings
- **Batch Upload**: Process multiple papers at once
- **Optimal Chunking**: Adjust chunk size for your use case
- **Regular Cleanup**: Remove unused papers to save space

## Contributing

### Development Setup
```bash
git clone <repository-url>
cd research-rag-assistant
python -m venv venv
source venv/bin/activate pip install -r requirements.txt
# Add your API keys to .env
streamlit run app.py
```

## License

MIT License

## 🙏 Acknowledgments

### Core Technologies
- **[LangChain](https://langchain.com/)**: Advanced RAG framework
- **[Groq](https://groq.com/)**: High-speed LLM inference
- **[Streamlit](https://streamlit.io/)**: Modern web app framework
- **[FAISS](https://github.com/facebookresearch/faiss)**: Efficient similarity search
- **[Sentence Transformers](https://www.sbert.net/)**: State-of-the-art embeddings

### Data Sources
- **[arXiv](https://arxiv.org/)**: Open access scientific papers
- **[Semantic Scholar](https://www.semanticscholar.org/)**: Academic paper database
- **[PubMed](https://pubmed.ncbi.nlm.nih.gov/)**: Biomedical literature

---

**Built with ❤️ as ResearchMind AI**  
*Your intelligent research companion*