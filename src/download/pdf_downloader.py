"""
PDF Downloader - Downloads research papers from various sources
"""
import logging
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)


class PDFDownloader:
    """Download PDFs from research paper sources"""
    
    def __init__(
        self,
        download_dir: str = "data/fetched/pdfs",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize PDF downloader
        
        Args:
            download_dir: Directory to save PDFs
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"PDF Downloader initialized. Download directory: {self.download_dir}")
    
    def download_paper(
        self,
        paper: Dict[str, Any],
        custom_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a single paper
        
        Args:
            paper: Paper metadata dictionary with 'pdf_url'
            custom_filename: Optional custom filename
            
        Returns:
            Updated paper metadata with download status
        """
        pdf_url = paper.get('pdf_url')
        
        if not pdf_url:
            logger.warning(f"No PDF URL for paper: {paper.get('title', 'Unknown')}")
            paper['download_status'] = 'unavailable'
            paper['local_path'] = None
            return paper
        
        paper_id = paper.get('paper_id', 'unknown')
        title = paper.get('title', 'Unknown Title')
        
        logger.info(f"Downloading: {title[:60]}...")
        
        # Generate filename
        if custom_filename:
            filename = custom_filename
        else:
            filename = self._generate_filename(paper)
        
        filepath = self.download_dir / filename
        
        # Check if already downloaded
        if filepath.exists():
            logger.info(f"Already exists: {filename}")
            paper['download_status'] = 'success'
            paper['local_path'] = str(filepath)
            return paper
        
        # Download with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    pdf_url,
                    timeout=self.timeout,
                    headers={'User-Agent': 'ResearchRAGAssistant/1.0'},
                    stream=True
                )
                response.raise_for_status()
                
                # Check if response is actually a PDF
                content_type = response.headers.get('Content-Type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    logger.warning(f"URL doesn't return PDF: {content_type}")
                    paper['download_status'] = 'invalid_format'
                    paper['local_path'] = None
                    return paper
                
                # Save PDF
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify the file was written and has content
                if filepath.exists() and filepath.stat().st_size > 0:
                    logger.info(f"Successfully downloaded: {filename}")
                    paper['download_status'] = 'success'
                    paper['local_path'] = str(filepath)
                    return paper
                else:
                    logger.error(f"Downloaded file is empty or doesn't exist")
                    if filepath.exists():
                        filepath.unlink()
                    raise Exception("Empty file downloaded")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to download after {self.max_retries} attempts")
                    paper['download_status'] = 'failed'
                    paper['local_path'] = None
                    paper['error'] = str(e)
            except Exception as e:
                logger.error(f"Unexpected error downloading PDF: {e}")
                paper['download_status'] = 'failed'
                paper['local_path'] = None
                paper['error'] = str(e)
                if filepath.exists():
                    filepath.unlink()
                break
        
        return paper
    
    def download_batch(
        self,
        papers: List[Dict[str, Any]],
        max_concurrent: int = 3,
        skip_unavailable: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Download multiple papers concurrently
        
        Args:
            papers: List of paper metadata dictionaries
            max_concurrent: Maximum concurrent downloads
            skip_unavailable: Skip papers without PDF URLs
            
        Returns:
            List of updated paper metadata
        """
        # Filter papers
        if skip_unavailable:
            downloadable = [p for p in papers if p.get('pdf_url')]
            logger.info(f"Found {len(downloadable)} papers with PDF URLs out of {len(papers)}")
        else:
            downloadable = papers
        
        if not downloadable:
            logger.warning("No papers to download")
            return papers
        
        logger.info(f"Starting batch download of {len(downloadable)} papers...")
        
        downloaded_papers = []
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_paper = {
                executor.submit(self.download_paper, paper): paper
                for paper in downloadable
            }
            
            for future in as_completed(future_to_paper):
                try:
                    updated_paper = future.result()
                    downloaded_papers.append(updated_paper)
                except Exception as e:
                    paper = future_to_paper[future]
                    logger.error(f"Error downloading {paper.get('title', 'Unknown')}: {e}")
                    paper['download_status'] = 'failed'
                    paper['error'] = str(e)
                    downloaded_papers.append(paper)
        
        # Update original papers list with download status
        paper_id_to_updated = {p['paper_id']: p for p in downloaded_papers}
        
        for paper in papers:
            paper_id = paper.get('paper_id')
            if paper_id in paper_id_to_updated:
                paper.update(paper_id_to_updated[paper_id])
        
        # Log summary
        successful = sum(1 for p in papers if p.get('download_status') == 'success')
        failed = sum(1 for p in papers if p.get('download_status') == 'failed')
        unavailable = sum(1 for p in papers if p.get('download_status') == 'unavailable')
        
        logger.info(f"Download complete: {successful} successful, {failed} failed, {unavailable} unavailable")
        
        return papers
    
    def _generate_filename(self, paper: Dict[str, Any]) -> str:
        """
        Generate a safe filename for a paper
        
        Args:
            paper: Paper metadata
            
        Returns:
            Safe filename string
        """
        # Try to use paper ID
        paper_id = paper.get('paper_id', '')
        
        # Clean up paper ID for filename
        safe_id = paper_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Add source prefix if not already there
        source = paper.get('source', 'unknown')
        if not safe_id.startswith(source):
            safe_id = f"{source}_{safe_id}"
        
        filename = f"{safe_id}.pdf"
        
        # Ensure filename isn't too long (max 255 characters)
        if len(filename) > 255:
            # Use hash of paper ID instead
            hash_id = hashlib.md5(paper_id.encode()).hexdigest()
            filename = f"{source}_{hash_id}.pdf"
        
        return filename
    
    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get download statistics
        
        Returns:
            Dictionary with download stats
        """
        pdf_files = list(self.download_dir.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files)
        
        return {
            'total_files': len(pdf_files),
            'total_size_mb': total_size / (1024 * 1024),
            'download_dir': str(self.download_dir)
        }
    
    def clear_downloads(self, confirm: bool = False):
        """
        Clear all downloaded PDFs
        
        Args:
            confirm: Must be True to actually delete files
        """
        if not confirm:
            logger.warning("clear_downloads called without confirmation")
            return
        
        pdf_files = list(self.download_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            pdf_file.unlink()
        
        logger.info(f"Deleted {len(pdf_files)} PDF files")


def test_pdf_downloader():
    """Test PDF downloader"""
    print("Testing PDF Downloader")
    print("=" * 50)
    
    import tempfile
    
    # Create temporary download directory
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = PDFDownloader(download_dir=tmpdir, max_retries=2)
        
        print(f"\n1. Download directory: {tmpdir}")
        
        # Create test paper metadata (using a known arXiv paper)
        test_paper = {
            'paper_id': 'arxiv_1706.03762',
            'title': 'Attention Is All You Need',
            'source': 'arxiv',
            'pdf_url': 'https://arxiv.org/pdf/1706.03762.pdf',
            'download_status': 'pending'
        }
        
        print("\n2. Testing single download...")
        print(f"   Paper: {test_paper['title']}")
        print("   Note: Skipping actual download (no network)")
        
        # Simulate download
        test_paper_no_url = {
            'paper_id': 'test_001',
            'title': 'Test Paper Without URL',
            'source': 'test'
        }
        
        result = downloader.download_paper(test_paper_no_url)
        print(f"   Paper without URL - Status: {result['download_status']}")
        
        print("\n3. Testing batch download...")
        test_papers = [
            {
                'paper_id': 'arxiv_1',
                'title': 'Paper 1',
                'pdf_url': 'https://example.com/paper1.pdf',
                'source': 'arxiv'
            },
            {
                'paper_id': 'arxiv_2',
                'title': 'Paper 2',
                # No PDF URL
                'source': 'arxiv'
            }
        ]
        
        print(f"   Simulating batch download of {len(test_papers)} papers...")
        # Would call: downloader.download_batch(test_papers)
        
        print("\n4. Testing download stats...")
        stats = downloader.get_download_stats()
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        
    print("\n✅ PDF downloader tests complete!")
    print("\nNote: Actual downloads require network connection.")
    print("To test with real downloads, run with internet access.")


if __name__ == "__main__":
    test_pdf_downloader()