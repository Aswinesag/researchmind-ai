"""
Text chunking module for research papers
Implements intelligent chunking strategies with metadata preservation
"""
import logging
from typing import List, Dict, Any, Optional
import re

try:
    from .utils import estimate_tokens
except ImportError:
    from utils import estimate_tokens

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text documents for embedding and retrieval"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            separators: List of separators for splitting (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_by_separators(self, text: str) -> List[str]:
        """
        Chunk text using hierarchical separators
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        current_size = 0
        
        # Split by the most preferred separator available
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                break
        else:
            parts = [text]
        
        for part in parts:
            part_size = estimate_tokens(part)
            
            # If adding this part exceeds chunk size
            if current_size + part_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Calculate overlap
                overlap_text = ""
                if self.chunk_overlap > 0:
                    # Take last portion of current chunk as overlap
                    words = current_chunk.split()
                    overlap_size = 0
                    overlap_words = []
                    
                    for word in reversed(words):
                        word_size = estimate_tokens(word)
                        if overlap_size + word_size <= self.chunk_overlap:
                            overlap_words.insert(0, word)
                            overlap_size += word_size
                        else:
                            break
                    
                    overlap_text = " ".join(overlap_words)
                
                current_chunk = overlap_text + " " + part if overlap_text else part
                current_size = estimate_tokens(current_chunk)
            else:
                current_chunk += separator + part if current_chunk else part
                current_size += part_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_with_metadata(
        self,
        text: str,
        metadata: Dict[str, Any],
        section_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and preserve metadata
        
        Args:
            text: Text to chunk
            metadata: Paper metadata
            section_aware: Whether to maintain section boundaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info(f"Chunking document: {metadata.get('title', 'Unknown')}")
        
        if section_aware:
            chunks = self._chunk_by_sections(text, metadata)
        else:
            chunks = self._chunk_basic(text, metadata)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def _chunk_basic(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Basic chunking without section awareness
        
        Args:
            text: Text to chunk
            metadata: Paper metadata
            
        Returns:
            List of chunk dictionaries
        """
        text_chunks = self.chunk_by_separators(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{metadata['paper_id']}_chunk_{i}",
                "paper_id": metadata['paper_id'],
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "metadata": {
                    "title": metadata.get('title', 'Unknown'),
                    "authors": metadata.get('authors', []),
                    "year": metadata.get('year'),
                    "source": metadata.get('source', 'upload'),
                    "file_path": metadata.get('file_path', '')
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text while maintaining section boundaries
        
        Args:
            text: Text to chunk
            metadata: Paper metadata
            
        Returns:
            List of chunk dictionaries with section info
        """
        # Detect sections
        sections = self._detect_sections(text)
        
        chunks = []
        chunk_counter = 0
        
        for section in sections:
            section_title = section['title']
            section_text = section['content']
            
            # Chunk each section
            section_chunks = self.chunk_by_separators(section_text)
            
            for i, chunk_text in enumerate(section_chunks):
                chunk = {
                    "chunk_id": f"{metadata['paper_id']}_chunk_{chunk_counter}",
                    "paper_id": metadata['paper_id'],
                    "text": chunk_text,
                    "chunk_index": chunk_counter,
                    "section": section_title,
                    "section_chunk_index": i,
                    "metadata": {
                        "title": metadata.get('title', 'Unknown'),
                        "authors": metadata.get('authors', []),
                        "year": metadata.get('year'),
                        "source": metadata.get('source', 'upload'),
                        "file_path": metadata.get('file_path', ''),
                        "section": section_title
                    }
                }
                chunks.append(chunk)
                chunk_counter += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks
    
    def _detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Detect sections in academic paper
        
        Args:
            text: Paper text
            
        Returns:
            List of sections with titles and content
        """
        sections = []
        
        # Common section patterns
        patterns = [
            r'\n(\d+\.?\s+[A-Z][^\n]{5,80})\n',  # Numbered sections (e.g., "1. Introduction")
            r'\n([A-Z][A-Z\s]{5,80})\n',  # ALL CAPS sections
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 2:  # Need at least 2 sections to be meaningful
                for i, match in enumerate(matches):
                    section_title = match.group(1).strip()
                    start_pos = match.end()
                    end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                    section_content = text[start_pos:end_pos].strip()
                    
                    if len(section_content) > 50:  # Ignore very short sections
                        sections.append({
                            "title": section_title,
                            "content": section_content
                        })
                break
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{
                "title": "Full Text",
                "content": text
            }]
        
        return sections
    
    def chunk_abstract_separately(
        self,
        text: str,
        abstract: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create special chunk for abstract and chunk the rest
        
        Args:
            text: Full paper text
            abstract: Abstract text
            metadata: Paper metadata
            
        Returns:
            List of chunks with abstract as first chunk
        """
        chunks = []
        
        # Add abstract as first chunk if available
        if abstract and len(abstract) > 50:
            abstract_chunk = {
                "chunk_id": f"{metadata['paper_id']}_abstract",
                "paper_id": metadata['paper_id'],
                "text": abstract,
                "chunk_index": 0,
                "section": "Abstract",
                "is_abstract": True,
                "metadata": {
                    "title": metadata.get('title', 'Unknown'),
                    "authors": metadata.get('authors', []),
                    "year": metadata.get('year'),
                    "source": metadata.get('source', 'upload'),
                    "section": "Abstract"
                }
            }
            chunks.append(abstract_chunk)
        
        # Chunk the rest of the text
        remaining_chunks = self.chunk_with_metadata(text, metadata, section_aware=True)
        
        # Adjust chunk indices
        for i, chunk in enumerate(remaining_chunks, start=len(chunks)):
            chunk['chunk_index'] = i
        
        chunks.extend(remaining_chunks)
        
        # Update total chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks


def test_chunker():
    """Test text chunking"""
    print("Testing Text Chunker")
    print("=" * 50)
    
    # Sample research paper text
    sample_text = """
    1. Introduction
    
    Machine learning has revolutionized many fields of computer science. 
    Deep learning, in particular, has shown remarkable success in various 
    applications including computer vision, natural language processing, 
    and speech recognition. This paper presents a novel approach to 
    understanding attention mechanisms in transformer models.
    
    2. Related Work
    
    Previous work on attention mechanisms includes the seminal paper by 
    Vaswani et al. which introduced the Transformer architecture. Since 
    then, numerous variants have been proposed including BERT, GPT, and 
    their successors. Our work builds upon these foundations.
    
    3. Methodology
    
    We propose a new attention mechanism that improves computational 
    efficiency while maintaining model performance. The key innovation 
    is in how we compute attention weights. Our approach reduces the 
    complexity from O(n^2) to O(n log n) for sequence length n.
    """
    
    metadata = {
        "paper_id": "test_paper_001",
        "title": "Novel Attention Mechanisms for Transformers",
        "authors": ["John Doe", "Jane Smith"],
        "year": 2024,
        "source": "upload"
    }
    
    # Test basic chunking
    print("\n1. Testing basic chunking...")
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_with_metadata(sample_text, metadata, section_aware=False)
    print(f"   Created {len(chunks)} chunks")
    print(f"   First chunk preview: {chunks[0]['text'][:100]}...")
    
    # Test section-aware chunking
    print("\n2. Testing section-aware chunking...")
    chunks_sectioned = chunker.chunk_with_metadata(sample_text, metadata, section_aware=True)
    print(f"   Created {len(chunks_sectioned)} chunks")
    for i, chunk in enumerate(chunks_sectioned[:3]):
        print(f"   Chunk {i} section: {chunk.get('section', 'N/A')}")
    
    # Test with abstract
    print("\n3. Testing abstract chunking...")
    abstract = "This paper presents novel attention mechanisms for transformer models that reduce computational complexity."
    chunks_with_abstract = chunker.chunk_abstract_separately(sample_text, abstract, metadata)
    print(f"   Created {len(chunks_with_abstract)} chunks")
    print(f"   First chunk is abstract: {chunks_with_abstract[0].get('is_abstract', False)}")
    
    # Test chunk metadata
    print("\n4. Testing chunk metadata...")
    sample_chunk = chunks[0]
    print(f"   Chunk ID: {sample_chunk['chunk_id']}")
    print(f"   Paper ID: {sample_chunk['paper_id']}")
    print(f"   Chunk index: {sample_chunk['chunk_index']}/{sample_chunk['total_chunks']}")
    print(f"   Metadata keys: {list(sample_chunk['metadata'].keys())}")
    
    print("\n✅ Text chunking tests complete!")


if __name__ == "__main__":
    test_chunker()