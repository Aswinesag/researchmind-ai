"""
Citation handler for formatting paper citations in IEEE style
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CitationHandler:
    """Handle citation formatting for research papers"""
    
    def __init__(self, style: str = "IEEE"):
        """
        Initialize citation handler
        
        Args:
            style: Citation style (currently only IEEE supported)
        """
        self.style = style
    
    def format_ieee_citation(
        self,
        paper: Dict[str, Any],
        include_doi: bool = True,
        include_url: bool = False
    ) -> str:
        """
        Format paper in IEEE citation style
        
        IEEE Format:
        [1] A. Author, B. Author, and C. Author, "Title of paper," 
        in Venue, Year, pp. pages. doi: XXX
        
        Args:
            paper: Paper metadata dictionary
            include_doi: Include DOI if available
            include_url: Include URL if available
            
        Returns:
            Formatted citation string
        """
        citation_parts = []
        
        # Authors
        authors = paper.get('authors', [])
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
        title = paper.get('title', 'Unknown Title')
        citation_parts.append(f'"{title},"')
        
        # Venue/Journal
        venue = paper.get('venue') or paper.get('journal')
        if venue:
            citation_parts.append(f"in {venue},")
        
        # Year
        year = paper.get('year')
        if year:
            citation_parts.append(f"{year}.")
        
        # DOI
        if include_doi:
            doi = paper.get('doi')
            if doi:
                citation_parts.append(f"doi: {doi}.")
        
        # URL (arXiv, PubMed, etc.)
        if include_url:
            url = None
            if paper.get('arxiv_id'):
                url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
            elif paper.get('pubmed_url'):
                url = paper['pubmed_url']
            elif paper.get('s2_url'):
                url = paper['s2_url']
            
            if url:
                citation_parts.append(f"Available: {url}")
        
        citation = " ".join(citation_parts)
        return citation
    
    def create_bibliography(
        self,
        papers: List[Dict[str, Any]],
        numbered: bool = True
    ) -> str:
        """
        Create a bibliography from list of papers
        
        Args:
            papers: List of paper metadata
            numbered: Include reference numbers
            
        Returns:
            Formatted bibliography string
        """
        bibliography = "## References\n\n"
        
        for i, paper in enumerate(papers, 1):
            citation = self.format_ieee_citation(paper)
            
            if numbered:
                bibliography += f"[{i}] {citation}\n\n"
            else:
                bibliography += f"{citation}\n\n"
        
        return bibliography
    
    def format_inline_citation(self, source_number: int) -> str:
        """
        Format inline citation
        
        Args:
            source_number: Source number
            
        Returns:
            Formatted inline citation
        """
        return f"[{source_number}]"
    
    def extract_citations_from_text(self, text: str) -> List[int]:
        """
        Extract citation numbers from text
        
        Args:
            text: Text with citations
            
        Returns:
            List of citation numbers
        """
        import re
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]
    
    def format_paper_card(self, paper: Dict[str, Any]) -> str:
        """
        Format paper information for display
        
        Args:
            paper: Paper metadata
            
        Returns:
            Formatted card string
        """
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', [])
        author_str = ', '.join(authors[:3])
        if len(authors) > 3:
            author_str += ' et al.'
        
        year = paper.get('year', 'N/A')
        venue = paper.get('venue', 'N/A')
        citations = paper.get('citations_count')
        
        card = f"**{title}**\n\n"
        card += f"*{author_str}*\n\n"
        card += f"📅 {year} | 📍 {venue}"
        
        if citations is not None:
            card += f" | 📊 {citations} citations"
        
        # Abstract
        abstract = paper.get('abstract', '')
        if abstract:
            # Truncate long abstracts
            if len(abstract) > 300:
                abstract = abstract[:297] + "..."
            card += f"\n\n{abstract}"
        
        # Links
        links = []
        if paper.get('pdf_url'):
            links.append("📄 [PDF]({})".format(paper['pdf_url']))
        if paper.get('arxiv_id'):
            links.append("🔗 [arXiv](https://arxiv.org/abs/{})".format(paper['arxiv_id']))
        if paper.get('doi'):
            links.append("🔗 [DOI](https://doi.org/{})".format(paper['doi']))
        
        if links:
            card += "\n\n" + " | ".join(links)
        
        return card
    
    def create_source_summary(
        self,
        chunks: List[Dict[str, Any]],
        include_text: bool = False
    ) -> str:
        """
        Create a summary of sources used
        
        Args:
            chunks: List of retrieved chunks
            include_text: Include chunk text
            
        Returns:
            Formatted summary
        """
        if not chunks:
            return "No sources available."
        
        # Group by paper
        papers_dict = {}
        for chunk in chunks:
            paper_id = chunk.get('paper_id')
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'metadata': chunk.get('metadata', {}),
                    'chunks': []
                }
            papers_dict[paper_id]['chunks'].append(chunk)
        
        summary = f"### 📚 Sources ({len(papers_dict)} papers, {len(chunks)} excerpts)\n\n"
        
        for i, (paper_id, data) in enumerate(papers_dict.items(), 1):
            metadata = data['metadata']
            title = metadata.get('title', 'Unknown')
            authors = metadata.get('authors', [])
            year = metadata.get('year', 'N/A')
            
            summary += f"**[{i}] {title}**\n"
            summary += f"  - Authors: {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}\n"
            summary += f"  - Year: {year}\n"
            summary += f"  - Excerpts used: {len(data['chunks'])}\n"
            
            if include_text:
                for j, chunk in enumerate(data['chunks'][:2], 1):  # Show max 2 chunks
                    section = chunk.get('section', 'N/A')
                    summary += f"\n  *Excerpt {j} ({section}):*\n"
                    summary += f"  > {chunk['text'][:200]}...\n"
            
            summary += "\n"
        
        return summary


def test_citation_handler():
    """Test citation formatting"""
    print("Testing Citation Handler")
    print("=" * 50)
    
    handler = CitationHandler(style="IEEE")
    
    # Test paper
    paper = {
        'title': 'Attention Is All You Need',
        'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit'],
        'year': 2017,
        'venue': 'NeurIPS',
        'doi': '10.48550/arXiv.1706.03762',
        'arxiv_id': '1706.03762',
        'citations_count': 85000
    }
    
    print("\n1. Testing IEEE citation format:")
    citation = handler.format_ieee_citation(paper, include_doi=True)
    print(f"   {citation}")
    
    print("\n2. Testing paper card format:")
    card = handler.format_paper_card(paper)
    print(card)
    
    print("\n3. Testing bibliography:")
    papers = [paper, {
        'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
        'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
        'year': 2019,
        'venue': 'NAACL',
        'doi': '10.18653/v1/N19-1423'
    }]
    
    bib = handler.create_bibliography(papers)
    print(bib)
    
    print("\n✅ Citation handler tests complete!")


if __name__ == "__main__":
    test_citation_handler()