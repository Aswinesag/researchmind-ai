"""
PubMed API integration for searching biomedical/life sciences papers
"""
import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PubMedSearcher:
    """Search and fetch papers from PubMed"""
    
    def __init__(self, email: str = "user@example.com", tool_name: str = "ResearchRAGAssistant", rate_limit_delay: float = 0.34):
        """
        Initialize PubMed searcher
        
        Args:
            email: Email for PubMed API identification (required)
            tool_name: Tool name for API identification
            rate_limit_delay: Delay between requests (0.34s = ~3 requests/second)
        """
        self.email = email
        self.tool_name = tool_name
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
        article_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Optional (start_year, end_year) tuple
            article_type: Optional article type filter (e.g., 'Review', 'Clinical Trial')
            
        Returns:
            List of paper metadata dictionaries
        """
        try:
            from pymed import PubMed
        except ImportError:
            logger.error("pymed package not installed. Install with: pip install pymed")
            return []
        
        logger.info(f"Searching PubMed for: {query}")
        
        # Initialize PubMed
        pubmed = PubMed(tool=self.tool_name, email=self.email)
        
        # Build query with filters
        search_query = query
        
        if year_range:
            start_year, end_year = year_range
            search_query += f" AND ({start_year}[PDAT]:{end_year}[PDAT])"
        
        if article_type:
            search_query += f" AND {article_type}[PT]"
        
        self._rate_limit()
        
        try:
            # Execute search
            results = pubmed.query(search_query, max_results=max_results)
            
            papers = []
            for article in results:
                paper = self._extract_paper_metadata(article)
                if paper:  # Only add if metadata extraction succeeded
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers on PubMed")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _extract_paper_metadata(self, article) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from PubMed article
        
        Args:
            article: PubMed article object
            
        Returns:
            Paper metadata dictionary or None
        """
        try:
            # Extract basic info
            pubmed_id = article.pubmed_id.strip() if article.pubmed_id else None
            if not pubmed_id:
                return None
            
            title = article.title if article.title else "Unknown Title"
            
            # Extract authors
            authors = []
            if article.authors:
                for author in article.authors:
                    # Handle different author formats
                    if hasattr(author, 'lastname') and hasattr(author, 'firstname'):
                        if author.lastname and author.firstname:
                            name = f"{author.firstname} {author.lastname}"
                        elif author.lastname:
                            name = author.lastname
                        else:
                            continue
                    else:
                        name = str(author)
                    authors.append(name)
            
            # Extract publication date
            pub_date = None
            year = None
            if article.publication_date:
                try:
                    pub_date = article.publication_date.isoformat()
                    year = article.publication_date.year
                except:
                    pass
            
            # Extract abstract
            abstract = article.abstract if article.abstract else ""
            
            # Extract journal info
            journal = article.journal if article.journal else ""
            
            # Extract DOI
            doi = article.doi if hasattr(article, 'doi') and article.doi else None
            
            # Check for PMC ID (indicates free full text availability)
            pmc_id = None
            if hasattr(article, 'pmc_id') and article.pmc_id:
                pmc_id = article.pmc_id.strip()
            
            # Construct URLs
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"
            pdf_url = None
            if pmc_id:
                pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/"
            
            metadata = {
                'paper_id': f"pubmed_{pubmed_id}",
                'source': 'pubmed',
                'pubmed_id': pubmed_id,
                'pmc_id': pmc_id,
                'title': title,
                'authors': authors,
                'author_string': ', '.join(authors) if authors else 'Unknown Authors',
                'abstract': abstract,
                'year': year,
                'publication_date': pub_date,
                'journal': journal,
                'venue': journal,
                'doi': doi,
                'pubmed_url': pubmed_url,
                'pdf_url': pdf_url,
                'citations_count': None,  # PubMed doesn't provide citation counts
                'full_text_available': pmc_id is not None,
                'download_status': 'pending' if pmc_id else 'unavailable'
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PubMed metadata: {e}")
            return None
    
    def get_paper_by_id(self, pubmed_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by PubMed ID
        
        Args:
            pubmed_id: PubMed ID (PMID)
            
        Returns:
            Paper metadata or None
        """
        try:
            from pymed import PubMed
        except ImportError:
            logger.error("pymed package not installed")
            return None
        
        logger.info(f"Fetching PubMed paper: {pubmed_id}")
        
        pubmed = PubMed(tool=self.tool_name, email=self.email)
        
        self._rate_limit()
        
        try:
            # Search by PMID
            results = pubmed.query(pubmed_id, max_results=1)
            
            for article in results:
                return self._extract_paper_metadata(article)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching PubMed paper {pubmed_id}: {e}")
            return None
    
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
        query = f"{author_name}[Author]"
        return self.search(query, max_results=max_results)
    
    def search_reviews(
        self,
        topic: str,
        max_results: int = 10,
        year_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for review articles on a topic
        
        Args:
            topic: Topic to search
            max_results: Maximum number of results
            year_range: Optional year range
            
        Returns:
            List of review papers
        """
        return self.search(
            topic,
            max_results=max_results,
            year_range=year_range,
            article_type="Review"
        )


def test_pubmed_searcher():
    """Test PubMed searcher"""
    print("Testing PubMed Searcher")
    print("=" * 50)
    
    searcher = PubMedSearcher(email="test@example.com")
    
    print("\n1. Testing basic search...")
    print("   Query: 'CRISPR gene editing'")
    results = searcher.search(
        query="CRISPR gene editing",
        max_results=3
    )
    
    print(f"   Found {len(results)} papers")
    for i, paper in enumerate(results, 1):
        print(f"\n   Paper {i}:")
        print(f"   Title: {paper['title'][:80]}...")
        print(f"   Authors: {paper['author_string'][:60]}...")
        print(f"   Year: {paper['year']}")
        print(f"   PubMed ID: {paper['pubmed_id']}")
        print(f"   Journal: {paper['journal'][:50] if paper['journal'] else 'N/A'}...")
        print(f"   Full text available: {paper['full_text_available']}")
    
    print("\n2. Testing year filter...")
    results_filtered = searcher.search(
        query="COVID-19 vaccine",
        max_results=3,
        year_range=(2023, 2024)
    )
    print(f"   Found {len(results_filtered)} papers from 2023-2024")
    
    print("\n3. Testing review article search...")
    reviews = searcher.search_reviews(
        topic="artificial intelligence healthcare",
        max_results=2
    )
    print(f"   Found {len(reviews)} review articles")
    
    print("\n✅ PubMed searcher tests complete!")
    print("\nNote: Actual tests require internet connection.")


if __name__ == "__main__":
    test_pubmed_searcher()