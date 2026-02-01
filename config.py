"""
Central configuration file for Research RAG Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FETCHED_DIR = DATA_DIR / "fetched"
FETCHED_PDFS_DIR = FETCHED_DIR / "pdfs"
FETCHED_METADATA_DIR = FETCHED_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories if they don't exist
for directory in [UPLOADS_DIR, FETCHED_PDFS_DIR, FETCHED_METADATA_DIR, 
                  PROCESSED_DIR, FAISS_INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "user@example.com")
PUBMED_TOOL_NAME = os.getenv("PUBMED_TOOL_NAME", "ResearchRAGAssistant")

# Model Configuration
EMBEDDING_MODEL = "allenai/specter"  # Scientific paper embeddings
# Fallback models if specter doesn't work
EMBEDDING_MODEL_FALLBACK = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Production model with good performance
# Alternative: "mistral-saba-24b" for multilingual capabilities

# Chunking Configuration
CHUNK_SIZE = 800  # Tokens per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Semantic splitting

# Retrieval Configuration
TOP_K_RETRIEVAL = 5  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score

# Search Configuration
DEFAULT_MAX_RESULTS = 15  # Default papers to fetch
MAX_SEARCH_RESULTS = 50  # Maximum allowed
MIN_CITATION_COUNT = 0  # Minimum citations filter
DEFAULT_YEAR_RANGE = (2020, 2024)  # Default year range

# API Rate Limits (requests per minute)
ARXIV_RATE_LIMIT = 20  # 1 request per 3 seconds
SEMANTIC_SCHOLAR_RATE_LIMIT = 100  # Per 5 minutes
PUBMED_RATE_LIMIT = 180  # 3 per second = 180 per minute

# Download Configuration
MAX_CONCURRENT_DOWNLOADS = 3  # Parallel downloads
DOWNLOAD_TIMEOUT = 30  # Seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # Seconds

# Citation Configuration
CITATION_STYLE = "IEEE"  # IEEE, APA, MLA
INCLUDE_PAGE_NUMBERS = True
INCLUDE_DOI = True

# UI Configuration
STREAMLIT_PAGE_TITLE = "Research RAG Assistant"
STREAMLIT_PAGE_ICON = "📚"
STREAMLIT_LAYOUT = "wide"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File Size Limits
MAX_UPLOAD_SIZE_MB = 50  # Maximum PDF upload size
MAX_TOTAL_STORAGE_GB = 5  # Maximum total storage

# Feature Flags
ENABLE_ARXIV = True
ENABLE_SEMANTIC_SCHOLAR = True
ENABLE_PUBMED = True
ENABLE_CITATION_NETWORK = True
ENABLE_AUTO_SUMMARIZATION = True

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    if not GROQ_API_KEY:
        issues.append("⚠️ GROQ_API_KEY not set in .env file")
    
    if ENABLE_SEMANTIC_SCHOLAR and not SEMANTIC_SCHOLAR_API_KEY:
        issues.append("⚠️ SEMANTIC_SCHOLAR_API_KEY not set (optional but recommended)")
    
    if ENABLE_PUBMED and PUBMED_EMAIL == "user@example.com":
        issues.append("⚠️ PUBMED_EMAIL should be set to your actual email")
    
    return issues

def get_config_summary():
    """Return a summary of current configuration"""
    return {
        "Embedding Model": EMBEDDING_MODEL,
        "LLM Model": GROQ_MODEL,
        "Chunk Size": CHUNK_SIZE,
        "Top K Retrieval": TOP_K_RETRIEVAL,
        "Citation Style": CITATION_STYLE,
        "Enabled Sources": {
            "arXiv": ENABLE_ARXIV,
            "Semantic Scholar": ENABLE_SEMANTIC_SCHOLAR,
            "PubMed": ENABLE_PUBMED
        }
    }

if __name__ == "__main__":
    # Test configuration
    print("Configuration Summary:")
    print("-" * 50)
    for key, value in get_config_summary().items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    print("Configuration Validation:")
    print("=" * 50)
    issues = validate_config()
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ All configuration settings are valid!")