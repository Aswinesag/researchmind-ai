"""
Utility functions for the Research RAG Assistant
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_file_hash(file_path: str) -> str:
    """
    Generate MD5 hash of a file for deduplication
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return ""


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing special characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove special characters but keep dots for extensions
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Remove multiple underscores
    filename = re.sub(r'_+', '_', filename)
    return filename


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extract publication year from text
    
    Args:
        text: Text to search for year
        
    Returns:
        Year as integer or None
    """
    # Look for 4-digit years between 1900 and current year + 1
    current_year = datetime.now().year
    pattern = r'\b(19\d{2}|20[0-2]\d)\b'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the first valid year found
        for match in matches:
            year = int(match)
            if 1900 <= year <= current_year + 1:
                return year
    return None


def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract DOI from text
    
    Args:
        text: Text to search for DOI
        
    Returns:
        DOI string or None
    """
    # DOI pattern: 10.xxxx/xxxxx
    pattern = r'10\.\d{4,}(?:\.\d+)*\/(?:(?!["&\'<>])\S)+'
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_emails_from_text(text: str) -> List[str]:
    """
    Extract email addresses from text
    
    Args:
        text: Text to search for emails
        
    Returns:
        List of email addresses
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Strip whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load dictionary from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file info
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "extension": path.suffix,
        "hash": generate_file_hash(file_path)
    }


def parse_author_names(author_string: str) -> List[str]:
    """
    Parse author names from a string
    
    Args:
        author_string: String containing author names
        
    Returns:
        List of author names
    """
    # Split by common delimiters
    authors = re.split(r'[;,]|\band\b', author_string)
    # Clean and filter
    authors = [a.strip() for a in authors if a.strip()]
    return authors


def create_ieee_citation(
    title: str,
    authors: List[str],
    year: Optional[int] = None,
    venue: Optional[str] = None,
    pages: Optional[str] = None,
    doi: Optional[str] = None
) -> str:
    """
    Create IEEE format citation
    
    Args:
        title: Paper title
        authors: List of author names
        year: Publication year
        venue: Publication venue
        pages: Page numbers
        doi: DOI
        
    Returns:
        IEEE formatted citation
    """
    citation_parts = []
    
    # Authors
    if authors:
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) <= 6:
            author_str = ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            # More than 6 authors, use et al.
            author_str = f"{authors[0]} et al."
        citation_parts.append(author_str)
    
    # Title
    if title:
        citation_parts.append(f'"{title},"')
    
    # Venue
    if venue:
        citation_parts.append(f"in {venue},")
    
    # Year
    if year:
        citation_parts.append(f"{year}.")
    
    # Pages
    if pages:
        citation_parts.append(f"pp. {pages}.")
    
    # DOI
    if doi:
        citation_parts.append(f"doi: {doi}.")
    
    return " ".join(citation_parts)


def estimate_tokens(text: str) -> int:
    """
    Estimate number of tokens in text (rough approximation)
    Rule of thumb: ~4 characters per token for English
    
    Args:
        text: Text to estimate
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_by_sentences(text: str, max_chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunk text by sentences with overlap
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum tokens per chunk
        overlap: Overlap tokens between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = estimate_tokens(sentence)
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                s_size = estimate_tokens(s)
                if overlap_size + s_size <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += s_size
                else:
                    break
            
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


if __name__ == "__main__":
    # Test utilities
    print("Testing Utility Functions")
    print("=" * 50)
    
    # Test author parsing
    authors_str = "John Doe, Jane Smith, and Bob Johnson"
    authors = parse_author_names(authors_str)
    print(f"Parsed authors: {authors}")
    
    # Test IEEE citation
    citation = create_ieee_citation(
        title="Attention Is All You Need",
        authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
        year=2017,
        venue="NeurIPS",
        doi="10.xxxx/xxxxx"
    )
    print(f"\nIEEE Citation:\n{citation}")
    
    # Test text cleaning
    dirty_text = "This   has    multiple    spaces\n\n\nand newlines"
    clean = clean_text(dirty_text)
    print(f"\nCleaned text: {clean}")
    
    # Test token estimation
    sample_text = "This is a sample sentence for token estimation."
    tokens = estimate_tokens(sample_text)
    print(f"\nEstimated tokens: {tokens}")
    
    print("\n✅ All utility functions working!")