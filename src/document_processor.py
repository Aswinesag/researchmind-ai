"""
Document processor for research papers
Handles PDF parsing, metadata extraction, and text preprocessing
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import PyPDF2
import pdfplumber

from .utils import (
    clean_text, extract_year_from_text, extract_doi_from_text,
    extract_emails_from_text, parse_author_names, generate_file_hash
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process research papers and extract metadata"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def is_supported(self, file_path: str) -> bool:
        """Check if file format is supported"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def extract_text_pypdf(self, file_path: str) -> str:
        """
        Extract text using PyPDF2
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {file_path}: {e}")
            return ""
    
    def extract_text_pdfplumber(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text and metadata using pdfplumber
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text, page_metadata)
        """
        try:
            text = ""
            page_metadata = []
            
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    page_metadata.append({
                        "page_number": i + 1,
                        "text_length": len(page_text),
                        "has_text": bool(page_text.strip())
                    })
            
            return text, page_metadata
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {file_path}: {e}")
            return "", []
    
    def extract_text(self, file_path: str, method: str = "pdfplumber") -> str:
        """
        Extract text from PDF using specified method
        
        Args:
            file_path: Path to PDF file
            method: Extraction method ('pdfplumber' or 'pypdf')
            
        Returns:
            Extracted text
        """
        if not self.is_supported(file_path):
            logger.error(f"Unsupported file format: {file_path}")
            return ""
        
        if method == "pdfplumber":
            text, _ = self.extract_text_pdfplumber(file_path)
            if not text:  # Fallback to PyPDF2
                logger.info("Falling back to PyPDF2")
                text = self.extract_text_pypdf(file_path)
        else:
            text = self.extract_text_pypdf(file_path)
        
        return clean_text(text)
    
    def extract_title(self, text: str, max_lines: int = 10) -> str:
        """
        Extract paper title from text (usually in first few lines)
        
        Args:
            text: Full paper text
            max_lines: Number of lines to search
            
        Returns:
            Extracted title
        """
        lines = text.split('\n')[:max_lines]
        
        # Title is usually the longest line in the first few lines
        # that's not too short and doesn't look like metadata
        title_candidates = []
        
        for line in lines:
            line = line.strip()
            if 20 <= len(line) <= 200:  # Reasonable title length
                # Skip lines that look like metadata
                if not re.search(r'^\d|^vol|^page|^abstract|^keywords|^doi', line.lower()):
                    title_candidates.append(line)
        
        if title_candidates:
            # Return the longest candidate as title
            return max(title_candidates, key=len)
        
        return "Unknown Title"
    
    def extract_authors(self, text: str, max_chars: int = 1000) -> List[str]:
        """
        Extract author names from text
        
        Args:
            text: Paper text (usually first page)
            max_chars: Characters to search
            
        Returns:
            List of author names
        """
        # Search in first portion of text
        search_text = text[:max_chars]
        
        # Common patterns for author sections
        patterns = [
            r'(?:Authors?|By):?\s*([^\n]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)',
        ]
        
        authors = []
        for pattern in patterns:
            matches = re.findall(pattern, search_text)
            if matches:
                for match in matches:
                    parsed = parse_author_names(match)
                    authors.extend(parsed)
                break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_authors = []
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)
        
        return unique_authors[:10]  # Limit to first 10 authors
    
    def extract_abstract(self, text: str) -> str:
        """
        Extract abstract from paper text
        
        Args:
            text: Full paper text
            
        Returns:
            Abstract text
        """
        # Look for "Abstract" section
        abstract_pattern = r'(?i)abstract[:\s]+(.*?)(?=\n\n|\nintroduction|\n1\.|\nkeywords)'
        match = re.search(abstract_pattern, text, re.DOTALL)
        
        if match:
            abstract = match.group(1).strip()
            # Clean up
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract[:1500]  # Limit length
        
        return ""
    
    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract paper sections (Introduction, Methods, Results, etc.)
        
        Args:
            text: Full paper text
            
        Returns:
            List of sections with titles and content
        """
        sections = []
        
        # Common section headers
        section_patterns = [
            r'\n(\d+\.?\s+[A-Z][^\n]{5,50})\n',  # Numbered sections
            r'\n([A-Z][A-Z\s]{5,50})\n',  # ALL CAPS sections
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                for i, match in enumerate(matches):
                    section_title = match.group(1).strip()
                    start_pos = match.end()
                    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section_content = text[start_pos:end_pos].strip()
                    
                    sections.append({
                        "title": section_title,
                        "content": section_content[:5000]  # Limit content length
                    })
                break
        
        return sections
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract references/bibliography
        
        Args:
            text: Full paper text
            
        Returns:
            List of references
        """
        # Look for references section
        ref_pattern = r'(?i)(?:references|bibliography)[:\s]+(.*?)(?=$|\nappendix)'
        match = re.search(ref_pattern, text, re.DOTALL)
        
        if match:
            ref_text = match.group(1)
            # Split by common reference separators
            refs = re.split(r'\n\[\d+\]|\n\d+\.', ref_text)
            refs = [ref.strip() for ref in refs if len(ref.strip()) > 20]
            return refs[:50]  # Limit to first 50 references
        
        return []
    
    def extract_metadata(self, file_path: str, text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from paper
        
        Args:
            file_path: Path to PDF file
            text: Pre-extracted text (optional)
            
        Returns:
            Dictionary with paper metadata
        """
        if text is None:
            text = self.extract_text(file_path)
        
        # Extract all metadata
        title = self.extract_title(text)
        authors = self.extract_authors(text)
        abstract = self.extract_abstract(text)
        year = extract_year_from_text(text[:2000])  # Search in first portion
        doi = extract_doi_from_text(text[:2000])
        sections = self.extract_sections(text)
        references = self.extract_references(text)
        
        # File metadata
        file_hash = generate_file_hash(file_path)
        file_path_obj = Path(file_path)
        
        metadata = {
            "paper_id": file_hash[:16],  # Use hash prefix as ID
            "file_name": file_path_obj.name,
            "file_path": str(file_path_obj.absolute()),
            "file_hash": file_hash,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "abstract": abstract,
            "num_sections": len(sections),
            "num_references": len(references),
            "text_length": len(text),
            "processed_date": datetime.now().isoformat(),
            "source": "upload"  # Will be 'arxiv', 'semantic_scholar', etc. for fetched papers
        }
        
        return metadata
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with text and metadata
        """
        logger.info(f"Processing document: {file_path}")
        
        # Extract text
        text = self.extract_text(file_path)
        
        if not text:
            logger.error(f"Failed to extract text from {file_path}")
            return {
                "success": False,
                "error": "Text extraction failed",
                "metadata": {},
                "text": ""
            }
        
        # Extract metadata
        metadata = self.extract_metadata(file_path, text)
        
        logger.info(f"Successfully processed: {metadata['title']}")
        
        return {
            "success": True,
            "metadata": metadata,
            "text": text,
            "abstract": metadata["abstract"],
            "sections": self.extract_sections(text)
        }


def test_document_processor():
    """Test the document processor"""
    print("Testing Document Processor")
    print("=" * 50)
    
    processor = DocumentProcessor()
    
    # Test text extraction methods
    sample_text = """
    Attention Is All You Need
    
    Ashish Vaswani, Noam Shazeer, Niki Parmar
    
    Abstract
    The dominant sequence transduction models are based on complex recurrent or 
    convolutional neural networks. We propose a new simple network architecture 
    based solely on attention mechanisms.
    
    1. Introduction
    Recurrent neural networks have been the dominant approach...
    
    2. Model Architecture
    The Transformer follows this overall architecture...
    
    References
    [1] First reference here
    [2] Second reference here
    """
    
    # Test metadata extraction
    print("\n1. Testing Title Extraction:")
    title = processor.extract_title(sample_text)
    print(f"   Title: {title}")
    
    print("\n2. Testing Author Extraction:")
    authors = processor.extract_authors(sample_text)
    print(f"   Authors: {authors}")
    
    print("\n3. Testing Abstract Extraction:")
    abstract = processor.extract_abstract(sample_text)
    print(f"   Abstract: {abstract[:100]}...")
    
    print("\n4. Testing Section Extraction:")
    sections = processor.extract_sections(sample_text)
    print(f"   Found {len(sections)} sections")
    for section in sections:
        print(f"   - {section['title']}")
    
    print("\n5. Testing Reference Extraction:")
    refs = processor.extract_references(sample_text)
    print(f"   Found {len(refs)} references")
    
    print("\n✅ Document Processor tests complete!")


if __name__ == "__main__":
    test_document_processor()