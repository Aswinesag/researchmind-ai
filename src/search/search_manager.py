"""
Search Manager - Orchestrates searches across multiple sources
"""
import logging
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .arxiv_searcher import ArxivSearcher
from .semantic_scholar_searcher import SemanticScholarSearcher
from .pubmed_searcher import PubMedSearcher

logger = logging.getLogger(__name__)


class SearchManager:
    """Manage searches across multiple paper sources"""
    
    def __init__(
        self,
        arxiv_enabled: bool = True,
        semantic_scholar_enabled: bool = True,
        pubmed_enabled: bool = True,
        semantic_scholar_api_key: Optional[str] = None,
        pubmed_email: str = "user@example.com"
    ):
        """
        Initialize search manager
        
        Args:
            arxiv_enabled: Enable arXiv search
            semantic_scholar_enabled: Enable Semantic Scholar search
            pubmed_enabled: Enable PubMed search
            semantic_scholar_api_key: API key for Semantic Scholar
            pubmed_email: Email for PubMed API
        """
        self.searchers = {}
        
        if arxiv_enabled:
            self.searchers['arxiv'] = ArxivSearcher()
            logger.info("Initialized arXiv searcher")
        
        if semantic_scholar_enabled:
            self.searchers['semantic_scholar'] = SemanticScholarSearcher(api_key=semantic_scholar_api_key)
            logger.info("Initialized Semantic Scholar searcher")
        
        if pubmed_enabled:
            self.searchers['pubmed'] = PubMedSearcher(email=pubmed_email)
            logger.info("Initialized PubMed searcher")
    
    def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results_per_source: int = 10,
        year_range: Optional[tuple] = None,
        min_citations: int = 0,
        parallel: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple sources
        
        Args:
            query: Search query
            sources: List of sources to search (None = all enabled)
            max_results_per_source: Maximum results per source
            year_range: Optional (start_year, end_year) tuple
            min_citations: Minimum citation count (Semantic Scholar only)
            parallel: Whether to search sources in parallel
            
        Returns:
            Dictionary mapping source names to lists of papers
        """
        if sources is None:
            sources = list(self.searchers.keys())
        
        # Filter to only enabled sources
        sources = [s for s in sources if s in self.searchers]
        
        if not sources:
            logger.warning("No search sources enabled")
            return {}
        
        logger.info(f"Searching {len(sources)} sources for: {query}")
        
        results = {}
        
        if parallel and len(sources) > 1:
            # Parallel search
            with ThreadPoolExecutor(max_workers=len(sources)) as executor:
                future_to_source = {}
                
                for source in sources:
                    future = executor.submit(
                        self._search_source,
                        source,
                        query,
                        max_results_per_source,
                        year_range,
                        min_citations
                    )
                    future_to_source[future] = source
                
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        papers = future.result()
                        results[source] = papers
                    except Exception as e:
                        logger.error(f"Error searching {source}: {e}")
                        results[source] = []
        else:
            # Sequential search
            for source in sources:
                try:
                    papers = self._search_source(
                        source,
                        query,
                        max_results_per_source,
                        year_range,
                        min_citations
                    )
                    results[source] = papers
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
                    results[source] = []
        
        # Log results
        total_papers = sum(len(papers) for papers in results.values())
        logger.info(f"Found {total_papers} total papers across {len(sources)} sources")
        for source, papers in results.items():
            logger.info(f"  {source}: {len(papers)} papers")
        
        return results
    
    def _search_source(
        self,
        source: str,
        query: str,
        max_results: int,
        year_range: Optional[tuple],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """
        Search a single source
        
        Args:
            source: Source name
            query: Search query
            max_results: Maximum results
            year_range: Optional year range
            min_citations: Minimum citations
            
        Returns:
            List of papers
        """
        searcher = self.searchers.get(source)
        if not searcher:
            return []
        
        try:
            if source == 'semantic_scholar':
                return searcher.search(
                    query=query,
                    max_results=max_results,
                    year_range=year_range,
                    min_citations=min_citations
                )
            else:
                return searcher.search(
                    query=query,
                    max_results=max_results,
                    year_range=year_range
                )
        except Exception as e:
            logger.error(f"Error in {source} search: {e}")
            return []
    
    def aggregate_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        deduplicate: bool = True,
        rank_by: str = 'citations'
    ) -> List[Dict[str, Any]]:
        """
        Aggregate and optionally deduplicate results from multiple sources
        
        Args:
            results: Dictionary mapping sources to paper lists
            deduplicate: Whether to remove duplicates
            rank_by: How to rank papers ('citations', 'year', 'relevance')
            
        Returns:
            Aggregated and ranked list of papers
        """
        all_papers = []
        
        # Combine all papers
        for source, papers in results.items():
            all_papers.extend(papers)
        
        if not all_papers:
            return []
        
        logger.info(f"Aggregating {len(all_papers)} papers")
        
        # Deduplicate if requested
        if deduplicate:
            all_papers = self._deduplicate_papers(all_papers)
            logger.info(f"After deduplication: {len(all_papers)} papers")
        
        # Rank papers
        all_papers = self._rank_papers(all_papers, rank_by)
        
        return all_papers
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate papers based on title and DOI/arXiv ID
        
        Args:
            papers: List of papers
            
        Returns:
            Deduplicated list
        """
        seen_titles: Set[str] = set()
        seen_ids: Set[str] = set()
        unique_papers = []
        
        for paper in papers:
            # Normalize title for comparison
            title = paper.get('title', '').lower().strip()
            
            # Check for unique identifiers
            doi = paper.get('doi', '').lower().strip() if paper.get('doi') else None
            arxiv_id = paper.get('arxiv_id', '').lower().strip() if paper.get('arxiv_id') else None
            pubmed_id = paper.get('pubmed_id', '').strip() if paper.get('pubmed_id') else None
            
            # Create unique key
            unique_key = doi or arxiv_id or pubmed_id or title
            
            if unique_key and unique_key not in seen_ids and title not in seen_titles:
                unique_papers.append(paper)
                seen_ids.add(unique_key)
                seen_titles.add(title)
        
        return unique_papers
    
    def _rank_papers(
        self,
        papers: List[Dict[str, Any]],
        rank_by: str = 'citations'
    ) -> List[Dict[str, Any]]:
        """
        Rank papers by specified criterion
        
        Args:
            papers: List of papers
            rank_by: Ranking criterion
            
        Returns:
            Sorted list of papers
        """
        if rank_by == 'citations':
            # Sort by citation count (highest first)
            papers.sort(
                key=lambda p: (p.get('citations_count') or 0, p.get('year') or 0),
                reverse=True
            )
        elif rank_by == 'year':
            # Sort by year (newest first)
            papers.sort(
                key=lambda p: (p.get('year') or 0, p.get('citations_count') or 0),
                reverse=True
            )
        elif rank_by == 'relevance':
            # Keep original order (assumed to be relevance-ranked by source)
            pass
        
        return papers
    
    def search_and_aggregate(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 15,
        year_range: Optional[tuple] = None,
        min_citations: int = 0,
        rank_by: str = 'citations'
    ) -> List[Dict[str, Any]]:
        """
        Search and return aggregated, deduplicated results
        
        Args:
            query: Search query
            sources: Sources to search
            max_results: Total maximum results (distributed across sources)
            year_range: Optional year range
            min_citations: Minimum citations
            rank_by: Ranking criterion
            
        Returns:
            Aggregated list of papers
        """
        # Distribute max_results across sources
        if sources is None:
            sources = list(self.searchers.keys())
        
        num_sources = len([s for s in sources if s in self.searchers])
        if num_sources == 0:
            return []
        
        per_source = max(1, max_results // num_sources)
        
        # Search all sources
        results = self.search(
            query=query,
            sources=sources,
            max_results_per_source=per_source,
            year_range=year_range,
            min_citations=min_citations,
            parallel=True
        )
        
        # Aggregate and return
        aggregated = self.aggregate_results(
            results,
            deduplicate=True,
            rank_by=rank_by
        )
        
        # Limit to max_results
        return aggregated[:max_results]
    
    def get_available_sources(self) -> List[str]:
        """Get list of available search sources"""
        return list(self.searchers.keys())


def test_search_manager():
    """Test search manager"""
    print("Testing Search Manager")
    print("=" * 50)
    
    # Initialize manager with all sources
    manager = SearchManager(
        arxiv_enabled=True,
        semantic_scholar_enabled=True,
        pubmed_enabled=False  # Disable for quick test
    )
    
    print(f"\nAvailable sources: {manager.get_available_sources()}")
    
    print("\n1. Testing multi-source search...")
    print("   Query: 'attention mechanisms'")
    results = manager.search(
        query="attention mechanisms",
        max_results_per_source=2,
        parallel=True
    )
    
    for source, papers in results.items():
        print(f"\n   {source}: {len(papers)} papers")
        for i, paper in enumerate(papers[:2], 1):
            print(f"   {i}. {paper['title'][:60]}...")
    
    print("\n2. Testing aggregation and deduplication...")
    aggregated = manager.aggregate_results(results, deduplicate=True, rank_by='year')
    print(f"   Total unique papers: {len(aggregated)}")
    
    print("\n3. Testing search and aggregate (one-step)...")
    final_results = manager.search_and_aggregate(
        query="machine learning",
        max_results=5,
        year_range=(2023, 2024),
        rank_by='citations'
    )
    print(f"   Found {len(final_results)} papers")
    for i, paper in enumerate(final_results[:3], 1):
        print(f"   {i}. {paper['title'][:60]}... ({paper['year']}, {paper.get('citations_count', 'N/A')} citations)")
    
    print("\n✅ Search manager tests complete!")
    print("\nNote: Actual tests require internet connection.")


if __name__ == "__main__":
    test_search_manager()