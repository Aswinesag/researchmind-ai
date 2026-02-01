"""
Semantic Scholar API integration for searching research papers
"""
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class SemanticScholarSearcher:
    """Search and fetch papers from Semantic Scholar"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_delay: float = 1.0):
        """
        Initialize Semantic Scholar searcher
        
        Args:
            api_key: Optional API key for higher rate limits
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {'User-Agent': 'ResearchRAGAssistant/1.0'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        return headers
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        year_range: Optional[tuple] = None,
        min_citations: int = 0,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            year_range: Optional (start_year, end_year) tuple
            min_citations: Minimum citation count
            fields: Optional list of fields to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        logger.info(f"Searching Semantic Scholar for: {query}")
        
        # Default fields to retrieve
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors',
                'citationCount', 'influentialCitationCount', 'venue',
                'publicationDate', 'externalIds', 'openAccessPdf', 'url'
            ]
        
        # Build query parameters
        params = {
            'query': query,
            'limit': min(max_results, 100),  # API limit is 100
            'fields': ','.join(fields)
        }
        
        # Add year filter if specified
        if year_range:
            params['year'] = f"{year_range[0]}-{year_range[1]}"
        
        # Add minimum citation filter
        if min_citations > 0:
            params['minCitationCount'] = min_citations
        
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}/paper/search",
                params=params,
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('data', []):
                paper = self._extract_paper_metadata(item)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers on Semantic Scholar")
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _extract_paper_metadata(self, item: Dict) -> Dict[str, Any]:
        """
        Extract metadata from Semantic Scholar result
        
        Args:
            item: Semantic Scholar paper object
            
        Returns:
            Paper metadata dictionary
        """
        # Extract authors
        authors = []
        if item.get('authors'):
            authors = [author.get('name', '') for author in item['authors']]
        
        # Get external IDs
        external_ids = item.get('externalIds', {})
        arxiv_id = external_ids.get('ArXiv')
        doi = external_ids.get('DOI')
        
        # Get PDF URL
        pdf_url = None
        open_access = item.get('openAccessPdf')
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get('url')
        
        # Create paper ID
        paper_id = item.get('paperId', '')
        
        metadata = {
            'paper_id': f"s2_{paper_id}",
            'source': 'semantic_scholar',
            'semantic_scholar_id': paper_id,
            'title': item.get('title', 'Unknown Title'),
            'authors': authors,
            'author_string': ', '.join(authors),
            'abstract': item.get('abstract', ''),
            'year': item.get('year'),
            'publication_date': item.get('publicationDate'),
            'citations_count': item.get('citationCount', 0),
            'influential_citations': item.get('influentialCitationCount', 0),
            'venue': item.get('venue', ''),
            'doi': doi,
            'arxiv_id': arxiv_id,
            'pdf_url': pdf_url,
            's2_url': item.get('url', ''),
            'external_ids': external_ids,
            'full_text_available': pdf_url is not None,
            'download_status': 'pending' if pdf_url else 'unavailable'
        }
        
        return metadata
    
    def get_paper_by_id(
        self,
        paper_id: str,
        id_type: str = 's2'
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific paper by ID
        
        Args:
            paper_id: Paper ID
            id_type: Type of ID ('s2', 'doi', 'arxiv', 'pmid')
            
        Returns:
            Paper metadata or None
        """
        logger.info(f"Fetching Semantic Scholar paper: {paper_id}")
        
        # Map ID types to API format
        id_prefix_map = {
            's2': '',
            'doi': 'DOI:',
            'arxiv': 'ARXIV:',
            'pmid': 'PMID:'
        }
        
        prefix = id_prefix_map.get(id_type, '')
        full_id = f"{prefix}{paper_id}"
        
        fields = [
            'paperId', 'title', 'abstract', 'year', 'authors',
            'citationCount', 'influentialCitationCount', 'venue',
            'publicationDate', 'externalIds', 'openAccessPdf', 'url'
        ]
        
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}/paper/{full_id}",
                params={'fields': ','.join(fields)},
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            item = response.json()
            return self._extract_paper_metadata(item)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None
    
    def get_recommendations(
        self,
        paper_id: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recommended papers similar to a given paper
        
        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended papers
        """
        logger.info(f"Getting recommendations for paper: {paper_id}")
        
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}/paper/{paper_id}/recommendations",
                params={'limit': max_results},
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('recommendedPapers', []):
                paper = self._extract_paper_metadata(item)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} recommendations")
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
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
        # First, search for the author
        self._rate_limit()
        
        try:
            response = requests.get(
                f"{self.base_url}/author/search",
                params={'query': author_name, 'limit': 1},
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if not data.get('data'):
                logger.warning(f"Author not found: {author_name}")
                return []
            
            author_id = data['data'][0]['authorId']
            
            # Get author's papers
            self._rate_limit()
            
            response = requests.get(
                f"{self.base_url}/author/{author_id}/papers",
                params={'limit': max_results, 'fields': 'paperId,title,year,authors,citationCount,venue'},
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            papers = []
            
            for item in data.get('data', []):
                paper = self._extract_paper_metadata(item)
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers by {author_name}")
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching by author: {e}")
            return []


def test_semantic_scholar():
    """Test Semantic Scholar searcher"""
    print("Testing Semantic Scholar Searcher")
    print("=" * 50)
    
    searcher = SemanticScholarSearcher()
    
    print("\n1. Testing basic search...")
    print("   Query: 'transformer neural networks'")
    results = searcher.search(
        query="transformer neural networks",
        max_results=3
    )
    
    print(f"   Found {len(results)} papers")
    for i, paper in enumerate(results, 1):
        print(f"\n   Paper {i}:")
        print(f"   Title: {paper['title'][:80]}...")
        print(f"   Authors: {paper['author_string'][:60]}...")
        print(f"   Year: {paper['year']}")
        print(f"   Citations: {paper['citations_count']}")
        print(f"   Venue: {paper['venue']}")
        print(f"   PDF Available: {paper['full_text_available']}")
    
    print("\n2. Testing year and citation filters...")
    results_filtered = searcher.search(
        query="deep learning",
        max_results=3,
        year_range=(2020, 2024),
        min_citations=100
    )
    print(f"   Found {len(results_filtered)} highly-cited recent papers")
    
    print("\n✅ Semantic Scholar searcher tests complete!")
    print("\nNote: Actual tests require internet connection.")


if __name__ == "__main__":
    test_semantic_scholar()