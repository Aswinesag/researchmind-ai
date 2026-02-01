"""
Embedding generation module for research papers
Uses sentence-transformers for creating vector embeddings
"""
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence-transformers"""
    
    def __init__(self, model_name: str = "allenai/specter"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self):
        """Load the embedding model (lazy loading)"""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Try fallback model
            logger.info("Attempting to load fallback model: all-mpnet-base-v2")
            try:
                from sentence_transformers import SentenceTransformer
                self.model_name = "sentence-transformers/all-mpnet-base-v2"
                self.model = SentenceTransformer(self.model_name)
                test_embedding = self.model.encode("test")
                self.embedding_dim = len(test_embedding)
                logger.info(f"Fallback model loaded. Embedding dimension: {self.embedding_dim}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            self.load_model()
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings"""
        if self.embedding_dim is None:
            self.load_model()
        return self.embedding_dim


class ChunkEmbedder:
    """Embed document chunks with metadata"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize chunk embedder
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedder = embedding_generator
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 32) -> List[dict]:
        """
        Generate embeddings for chunks with metadata
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            batch_size: Batch size for processing
            
        Returns:
            List of chunks with added 'embedding' field
        """
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedder.embed_batch(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
        
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        
        return chunks
    
    def embed_single_chunk(self, chunk: dict) -> dict:
        """
        Generate embedding for a single chunk
        
        Args:
            chunk: Chunk dictionary with 'text' and metadata
            
        Returns:
            Chunk with added 'embedding' field
        """
        embedding = self.embedder.embed_text(chunk['text'])
        chunk['embedding'] = embedding
        return chunk


def test_embeddings():
    """Test embedding generation"""
    print("Testing Embedding Generation")
    print("=" * 50)
    
    # Initialize embedding generator
    print("\n1. Initializing embedding generator...")
    generator = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")  # Smaller model for testing
    
    # Test single text embedding
    print("\n2. Testing single text embedding...")
    text = "This is a sample research paper about machine learning."
    embedding = generator.embed_text(text)
    print(f"   Text: {text}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {generator.get_embedding_dim()}")
    
    # Test batch embedding
    print("\n3. Testing batch embedding...")
    texts = [
        "Introduction to neural networks",
        "Deep learning applications",
        "Transformer architecture explained",
        "Attention mechanisms in NLP"
    ]
    embeddings = generator.embed_batch(texts, show_progress=False)
    print(f"   Number of texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Test ChunkEmbedder
    print("\n4. Testing ChunkEmbedder...")
    chunk_embedder = ChunkEmbedder(generator)
    
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": texts[0],
            "page": 1,
            "section": "Introduction"
        },
        {
            "chunk_id": "chunk_2",
            "text": texts[1],
            "page": 2,
            "section": "Methods"
        }
    ]
    
    embedded_chunks = chunk_embedder.embed_chunks(chunks)
    print(f"   Embedded {len(embedded_chunks)} chunks")
    print(f"   First chunk embedding shape: {embedded_chunks[0]['embedding'].shape}")
    
    # Test similarity (cosine similarity)
    print("\n5. Testing embedding similarity...")
    emb1 = embeddings[0]
    emb2 = embeddings[1]
    emb3 = embeddings[2]
    
    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    print(f"   Similarity between '{texts[0]}' and '{texts[1]}': {sim_1_2:.4f}")
    print(f"   Similarity between '{texts[0]}' and '{texts[2]}': {sim_1_3:.4f}")
    
    print("\n✅ Embedding generation tests complete!")


if __name__ == "__main__":
    test_embeddings()