"""
RAG Chain - Integrates retrieval with LLM generation using LangChain and Groq
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add prompts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from prompts.research_prompts import PROMPT_TEMPLATES, get_prompt, detect_question_type

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG pipeline for question answering over research papers"""
    
    def __init__(
        self,
        vector_store,
        embedding_generator,
        groq_api_key: Optional[str] = None,
        model_name: str = "mixtral-8x7b-32768",
        top_k: int = 5,
        temperature: float = 0.1
    ):
        """
        Initialize RAG chain
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            groq_api_key: Groq API key
            model_name: Groq model name
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        self.model_name = model_name
        self.top_k = top_k
        self.temperature = temperature
        self.llm = None
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        try:
            from langchain_groq import ChatGroq
            
            if not self.groq_api_key:
                raise ValueError("Groq API key not set")
            
            self.llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name=self.model_name,
                temperature=self.temperature
            )
            
            logger.info(f"Initialized Groq LLM: {self.model_name}")
            
        except ImportError:
            logger.error("langchain-groq not installed. Install with: pip install langchain-groq")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_by: Optional metadata filters
            
        Returns:
            List of retrieved chunks with metadata
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_by=filter_by
        )
        
        logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
        
        return results
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the papers."
        
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            authors = metadata.get('authors', [])
            year = metadata.get('year', 'N/A')
            section = metadata.get('section', '')
            
            # Format chunk with metadata
            chunk_text = f"[Source {i}]\n"
            chunk_text += f"Title: {title}\n"
            chunk_text += f"Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}\n"
            chunk_text += f"Year: {year}\n"
            if section:
                chunk_text += f"Section: {section}\n"
            chunk_text += f"\nContent:\n{chunk['text']}\n"
            
            context_parts.append(chunk_text)
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str, use_smart_prompts: bool = True) -> str:
        """
        Create prompt for the LLM
        
        Args:
            query: User query
            context: Retrieved context
            use_smart_prompts: Use question-type detection for specialized prompts
            
        Returns:
            Formatted prompt
        """
        if use_smart_prompts:
            # Detect question type and use appropriate prompt
            prompt_type = detect_question_type(query)
            logger.info(f"Using prompt type: {prompt_type}")
            
            try:
                if prompt_type == 'conversational':
                    # For conversational, we'll handle this separately
                    prompt = get_prompt('default', context=context, question=query)
                else:
                    prompt = get_prompt(prompt_type, context=context, question=query)
            except Exception as e:
                logger.warning(f"Error using specialized prompt: {e}, falling back to default")
                prompt = get_prompt('default', context=context, question=query)
        else:
            # Use default prompt
            prompt = get_prompt('default', context=context, question=query)
        
        return prompt
    
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            chunks: Retrieved chunks
            include_sources: Whether to include source chunks in response
            
        Returns:
            Dictionary with answer and sources
        """
        # Format context
        context = self.format_context(chunks)
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        try:
            # Generate response
            response = self.llm.invoke(prompt)
            answer = response.content
            
            logger.info("Generated answer successfully")
            
            result = {
                'answer': answer,
                'query': query,
                'num_sources': len(chunks)
            }
            
            if include_sources:
                result['sources'] = chunks
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'query': query,
                'num_sources': 0,
                'sources': [],
                'error': str(e)
            }
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_by: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve + generate
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            filter_by: Optional metadata filters
            include_sources: Include source chunks in response
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        # Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k, filter_by=filter_by)
        
        if not chunks:
            return {
                'answer': "I couldn't find relevant information in the papers to answer your question.",
                'query': question,
                'num_sources': 0,
                'sources': []
            }
        
        # Generate answer
        result = self.generate_answer(question, chunks, include_sources)
        
        return result
    
    def format_sources_for_display(self, sources: List[Dict[str, Any]]) -> str:
        """
        Format sources for display in UI
        
        Args:
            sources: List of source chunks
            
        Returns:
            Formatted string
        """
        if not sources:
            return "No sources"
        
        output = "### Sources\n\n"
        
        for i, chunk in enumerate(sources, 1):
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            authors = metadata.get('authors', [])
            year = metadata.get('year', 'N/A')
            section = metadata.get('section', '')
            
            output += f"**[{i}]** {title}\n"
            output += f"- Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}\n"
            output += f"- Year: {year}\n"
            if section:
                output += f"- Section: {section}\n"
            output += f"- Relevance Score: {chunk.get('score', 0):.3f}\n\n"
        
        return output


class ConversationRAG(RAGChain):
    """RAG with conversation history support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    def add_to_history(self, question: str, answer: str):
        """Add Q&A to conversation history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer
        })
        
        # Keep only last 5 exchanges
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
    
    def get_context_with_history(self, query: str, context: str) -> str:
        """Create prompt with conversation history"""
        prompt = f"""You are a research assistant helping analyze academic papers. You have access to relevant excerpts from research papers and the conversation history.

CONVERSATION HISTORY:
"""
        
        for i, exchange in enumerate(self.conversation_history[-3:], 1):
            prompt += f"\nQ{i}: {exchange['question']}\n"
            prompt += f"A{i}: {exchange['answer'][:200]}...\n"
        
        prompt += f"""

CONTEXT FROM PAPERS:
{context}

USER QUESTION:
{query}

ANSWER (Use IEEE citation format [1], [2] etc.):"""
        
        return prompt
    
    def query_with_history(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query with conversation history"""
        # Retrieve chunks
        chunks = self.retrieve(question, top_k=top_k, filter_by=filter_by)
        
        if not chunks:
            answer = "I couldn't find relevant information in the papers."
            self.add_to_history(question, answer)
            return {
                'answer': answer,
                'query': question,
                'num_sources': 0,
                'sources': []
            }
        
        # Format context
        context = self.format_context(chunks)
        
        # Create prompt with history
        prompt = self.get_context_with_history(question, context)
        
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Add to history
            self.add_to_history(question, answer)
            
            return {
                'answer': answer,
                'query': question,
                'num_sources': len(chunks),
                'sources': chunks
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'query': question,
                'num_sources': 0,
                'sources': []
            }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Cleared conversation history")


def test_rag_chain():
    """Test RAG chain (requires full setup)"""
    print("Testing RAG Chain")
    print("=" * 50)
    
    print("\nNote: This test requires:")
    print("1. Groq API key set in .env")
    print("2. Vector store with indexed papers")
    print("3. Embedding model loaded")
    
    print("\nRAG chain module created successfully!")
    print("Integration test should be done through the Streamlit app.")


if __name__ == "__main__":
    test_rag_chain()