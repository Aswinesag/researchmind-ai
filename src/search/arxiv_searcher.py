"""
arXiv API integration for searching and fetching research papers
"""
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ArxivSearcher:
    """Search and fetch papers from arXiv"""
    
    def __init__(self, rate_limit_delay: float = 3.0):
        """
        Initialize arXiv searcher
        
        Args:
            rate_limit_delay: Delay between requests in seconds (arXiv recommends 3s)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        year_range: Optional[tuple] = None,
        sort_by: str = "relevance"
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Optional (start_year, end_year) tuple
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            import arxiv
        except ImportError:
            logger.error("arxiv package not installed. Install with: pip install arxiv")
            return []
        
        logger.info(f"Searching arXiv for: {query}")
        
        # Respect rate limit
        self._rate_limit()
        
        # Map sort options
        sort_options = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
            'submittedDate': arxiv.SortCriterion.SubmittedDate
        }
        
        sort_criterion = sort_options.get(sort_by, arxiv.SortCriterion.Relevance)
        
        try:
            # Create search
            search = arxiv.Search(
                query=query,
                max_results=max_results * 2,  # Get extra to filter by year
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending
            )
            
            papers = []
            for result in search.results():
                # Extract year from publication date
                pub_year = result.published.year
                
                # Filter by year range if specified
                if year_range:
                    start_year, end_year = year_range
                    if not (start_year <= pub_year <= end_year):
                        continue
                
                # Extract paper metadata
                paper = self._extract_paper_metadata(result)
                papers.append(paper)
                
                # Stop if we have enough results
                if len(papers) >= max_results:
                    break
            
            logger.info(f"Found {len(papers)} papers on arXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _extract_paper_metadata(self, result) -> Dict[str, Any]:
        """
        Extract metadata from arXiv result
        
        Args:
            result: arXiv search result object
            
        Returns:
            Paper metadata dictionary
        """
        # Extract authors
        authors = [author.name for author in result.authors]
        
        # Extract categories
        categories = result.categories
        primary_category = result.primary_category
        
        # Get PDF URL
        pdf_url = result.pdf_url
        
        # Create unique paper ID from arXiv ID
        arxiv_id = result.entry_id.split('/')[-1]  # Extract ID from URL
        
        metadata = {
            'paper_id': f"arxiv_{arxiv_id.replace('.', '_')}",
            'source': 'arxiv',
            'arxiv_id': arxiv_id,
            'title': result.title,
            'authors': authors,
            'author_string': ', '.join(authors),
            'abstract': result.summary.replace('\n', ' ').strip(),
            'year': result.published.year,
            'publication_date': result.published.isoformat(),
            'updated_date': result.updated.isoformat(),
            'categories': categories,
            'primary_category': primary_category,
            'pdf_url': pdf_url,
            'arxiv_url': result.entry_id,
            'doi': result.doi,
            'comment': result.comment,
            'journal_ref': result.journal_ref,
            'citations_count': None,  # arXiv doesn't provide citation counts
            'venue': journal_ref if (journal_ref := result.journal_ref) else 'arXiv',
            'full_text_available': True,  # arXiv always has full text
            'download_status': 'pending'
        }
        
        return metadata
    
    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by arXiv ID
        
        Args:
            arxiv_id: arXiv ID (e.g., '1706.03762' or 'arxiv:1706.03762')
            
        Returns:
            Paper metadata or None
        """
        try:
            import arxiv
        except ImportError:
            logger.error("arxiv package not installed")
            return None
        
        # Clean arXiv ID
        if arxiv_id.startswith('arxiv:'):
            arxiv_id = arxiv_id[6:]
        
        logger.info(f"Fetching arXiv paper: {arxiv_id}")
        
        self._rate_limit()
        
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            return self._extract_paper_metadata(result)
        except Exception as e:
            logger.error(f"Error fetching arXiv paper {arxiv_id}: {e}")
            return None
    
    def search_by_category(
        self,
        category: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search papers by arXiv category
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG', 'physics.gen-ph')
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        query = f"cat:{category}"
        return self.search(query, max_results=max_results, sort_by='submittedDate')
    
    def search_by_author(
        self,
        author_name: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search papers by author name
        
        Args:
            author_name: Author name
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        query = f"au:{author_name}"
        return self.search(query, max_results=max_results, sort_by='submittedDate')


def test_arxiv_searcher():
    """Test arXiv searcher"""
    print("Testing arXiv Searcher")
    print("=" * 50)
    
    searcher = ArxivSearcher(rate_limit_delay=3.0)
    
    # Test basic search
    print("\n1. Testing basic search...")
    print("   Query: 'attention mechanism transformer'")
    results = searcher.search(
        query="attention mechanism transformer",
        max_results=3,
        sort_by="relevance"
    )
    
    print(f"   Found {len(results)} papers")
    for i, paper in enumerate(results, 1):
        print(f"\n   Paper {i}:")
        print(f"   Title: {paper['title'][:80]}...")
        print(f"   Authors: {paper['author_string'][:60]}...")
        print(f"   Year: {paper['year']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Categories: {', '.join(paper['categories'][:3])}")
        print(f"   PDF URL: {paper['pdf_url']}")
    
    # Test year filtering
    print("\n2. Testing year range filter...")
    results_filtered = searcher.search(
        query="machine learning",
        max_results=3,
        year_range=(2023, 2024),
        sort_by="submittedDate"
    )
    print(f"   Found {len(results_filtered)} papers from 2023-2024")
    
    # Test category search
    print("\n3. Testing category search...")
    cat_results = searcher.search_by_category("cs.AI", max_results=2)
    print(f"   Found {len(cat_results)} papers in cs.AI category")
    
    print("\n✅ arXiv searcher tests complete!")
    print("\nNote: Actual tests require internet connection.")
    print("Run this file directly to test with real arXiv API.")


if __name__ == "__main__":
    test_arxiv_searcher()