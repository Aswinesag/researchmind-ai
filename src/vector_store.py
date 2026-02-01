"""
Vector store module using FAISS for similarity search
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document chunks"""
    
    def __init__(self, embedding_dim: int, index_path: Optional[str] = None):
        """
        Initialize vector store
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load index
        """
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.index = None
        self.chunks = []  # Store chunk metadata
        self.chunk_id_to_idx = {}  # Map chunk IDs to indices
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            # Use flat L2 index for small to medium datasets
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
            
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks]).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunks and create ID mapping
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.chunk_id_to_idx[chunk['chunk_id']] = idx
        
        logger.info(f"Vector store now contains {len(self.chunks)} chunks")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_by: Optional metadata filters
            
        Returns:
            List of similar chunks with scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Ensure query embedding is 2D and float32
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS
        # Get more results if filtering is needed
        search_k = top_k * 3 if filter_by else top_k
        distances, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))
        
        # Get results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                
                # Apply filters if specified
                if filter_by and not self._matches_filter(chunk, filter_by):
                    continue
                
                # Add similarity score (convert distance to similarity)
                # For L2 distance, smaller is better, so use 1/(1+distance)
                chunk['score'] = float(1 / (1 + dist))
                chunk['distance'] = float(dist)
                
                results.append(chunk)
                
                if len(results) >= top_k:
                    break
        
        logger.info(f"Found {len(results)} results for query")
        
        return results
    
    def _matches_filter(self, chunk: Dict[str, Any], filter_by: Dict[str, Any]) -> bool:
        """
        Check if chunk matches filter criteria
        
        Args:
            chunk: Chunk dictionary
            filter_by: Filter criteria
            
        Returns:
            True if chunk matches all filters
        """
        for key, value in filter_by.items():
            # Check in both chunk and metadata
            chunk_value = chunk.get(key)
            if chunk_value is None and 'metadata' in chunk:
                chunk_value = chunk['metadata'].get(key)
            
            if chunk_value != value:
                return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk by ID
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk dictionary or None
        """
        idx = self.chunk_id_to_idx.get(chunk_id)
        if idx is not None and idx < len(self.chunks):
            return self.chunks[idx]
        return None
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get unique papers in the vector store
        
        Returns:
            List of unique paper metadata
        """
        papers = {}
        for chunk in self.chunks:
            paper_id = chunk.get('paper_id')
            if paper_id and paper_id not in papers:
                papers[paper_id] = chunk.get('metadata', {})
                papers[paper_id]['paper_id'] = paper_id
        
        return list(papers.values())
    
    def delete_paper(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper
        Note: This doesn't remove from FAISS index, just marks as deleted
        
        Args:
            paper_id: Paper ID to delete
            
        Returns:
            Number of chunks deleted
        """
        deleted_count = 0
        new_chunks = []
        new_chunk_id_to_idx = {}
        
        for i, chunk in enumerate(self.chunks):
            if chunk.get('paper_id') != paper_id:
                new_idx = len(new_chunks)
                new_chunks.append(chunk)
                new_chunk_id_to_idx[chunk['chunk_id']] = new_idx
            else:
                deleted_count += 1
        
        self.chunks = new_chunks
        self.chunk_id_to_idx = new_chunk_id_to_idx
        
        logger.info(f"Deleted {deleted_count} chunks for paper {paper_id}")
        
        return deleted_count
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk
        
        Args:
            path: Directory path to save index
        """
        import faiss
        
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save chunks and metadata
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_id_to_idx': self.chunk_id_to_idx,
                'embedding_dim': self.embedding_dim
            }, f)
        
        logger.info(f"Saved vector store to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """
        Load vector store from disk
        
        Args:
            path: Directory path to load index from
        """
        import faiss
        
        load_path = Path(path or self.index_path)
        
        # Load FAISS index
        index_file = load_path / "faiss.index"
        if not index_file.exists():
            logger.warning(f"No index found at {index_file}")
            return
        
        self.index = faiss.read_index(str(index_file))
        
        # Load chunks and metadata
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_id_to_idx = data['chunk_id_to_idx']
            self.embedding_dim = data['embedding_dim']
        
        logger.info(f"Loaded vector store from {load_path}")
        logger.info(f"Vector store contains {len(self.chunks)} chunks")
    
    def clear(self):
        """Clear all data from vector store"""
        self._initialize_index()
        self.chunks = []
        self.chunk_id_to_idx = {}
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        unique_papers = len(set(chunk.get('paper_id') for chunk in self.chunks))
        
        return {
            'total_chunks': len(self.chunks),
            'unique_papers': unique_papers,
            'embedding_dim': self.embedding_dim,
            'index_size': self.index.ntotal
        }


def test_vector_store():
    """Test vector store operations"""
    print("Testing Vector Store")
    print("=" * 50)
    
    # Create dummy embeddings and chunks
    embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
    
    print("\n1. Initializing vector store...")
    vs = VectorStore(embedding_dim=embedding_dim)
    print(f"   Vector store initialized with dimension {embedding_dim}")
    
    # Create sample chunks
    print("\n2. Creating sample chunks...")
    chunks = []
    for i in range(5):
        chunk = {
            'chunk_id': f'chunk_{i}',
            'paper_id': 'paper_1' if i < 3 else 'paper_2',
            'text': f'This is chunk {i} about machine learning.',
            'embedding': np.random.rand(embedding_dim).astype('float32'),
            'metadata': {
                'title': f'Paper {"1" if i < 3 else "2"}',
                'year': 2024,
                'section': 'Introduction'
            }
        }
        chunks.append(chunk)
    
    # Add chunks
    print("\n3. Adding chunks to vector store...")
    vs.add_chunks(chunks)
    stats = vs.get_stats()
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Unique papers: {stats['unique_papers']}")
    
    # Search
    print("\n4. Testing search...")
    query_embedding = np.random.rand(embedding_dim).astype('float32')
    results = vs.search(query_embedding, top_k=3)
    print(f"   Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"   Result {i+1}: {result['chunk_id']}, score: {result['score']:.4f}")
    
    # Filter search
    print("\n5. Testing filtered search...")
    filtered_results = vs.search(
        query_embedding,
        top_k=3,
        filter_by={'paper_id': 'paper_1'}
    )
    print(f"   Found {len(filtered_results)} results for paper_1")
    
    # Get papers
    print("\n6. Getting all papers...")
    papers = vs.get_all_papers()
    print(f"   Found {len(papers)} unique papers")
    for paper in papers:
        print(f"   - {paper.get('title')}")
    
    # Save and load
    print("\n7. Testing save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        vs.save(tmpdir)
        print(f"   Saved to {tmpdir}")
        
        # Create new vector store and load
        vs2 = VectorStore(embedding_dim=embedding_dim)
        vs2.load(tmpdir)
        print(f"   Loaded {len(vs2.chunks)} chunks")
    
    print("\n✅ Vector store tests complete!")


if __name__ == "__main__":
    test_vector_store()