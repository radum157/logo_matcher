import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import io
from PIL import Image
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

class WebsiteLogoExtractor:
    def __init__(self, timeout=0.1, logger=None):
        """
        Initialize logo extractor with optional logger

        Args:
            logger (logging.Logger, optional): Logger for tracking extraction process
        """
        self.logger = logger or logging.getLogger(__name__)
        self.timeout = timeout

    def _find_logo_url(self, soup, base_url):
        """
        Find logo URL from BeautifulSoup parsed HTML

        Args:
            soup (BeautifulSoup): Parsed HTML
            base_url (str): Base website URL

        Returns:
            str or None: Logo URL
        """
        logo_selectors = [
            'link[rel="icon"]',
            'link[rel="shortcut icon"]',
            'meta[property="og:image"]',
            'img[alt*="logo"]',
            'img.logo'
        ]

        for selector in logo_selectors:
            logo_element = soup.select_one(selector)
            if logo_element:
                logo_url = logo_element.get('href') or logo_element.get('content') or logo_element.get('src')
                return urljoin(base_url, logo_url)

        return None

    @retry(
        stop=stop_after_attempt(5),  # Retry up to 5 times
        wait=wait_exponential_jitter(initial=0.1, max=1),  # Exponential backoff with jitter (0.1s → 1s max)
        retry=retry_if_exception_type(requests.exceptions.RequestException),  # Retry only on request exceptions
    )
    def extract_logo(self, url, output_dir='logos'):
        """
        Extract logo from website

        Args:
            url (str): Website URL
            output_dir (str, optional): Directory to save logos

        Returns:
            str or None: Path to saved logo
        """
        try:
            # Generate output filename
            filename = f"{url.replace('https://', '').replace('http://', '').replace('.', '_')}_logo.jpg"
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                self.logger.info(f"Already downloaded logo for {url}")
                return

            # Fetch website HTML
            response = requests.get(url, timeout=self.timeout)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find logo URL
            logo_url = self._find_logo_url(soup, url)

            if not logo_url:
                self.logger.warning(f"No logo found for {url}")
                return None

            # Download logo
            logo_response = requests.get(logo_url)
            logo_image = Image.open(io.BytesIO(logo_response.content))

            # Compress and save logo
            return self.compress_logo(logo_image, output_path)

        except Exception as e:
            self.logger.error(f"Error extracting logo from {url}: {e}")
            return None

    def compress_logo(self, logo, output_path, max_size=(200, 200), quality=85):
        """
        Compress and save logo

        Args:
            logo (PIL.Image.Image): Logo image
            output_path (str): Path to save compressed logo
            max_size (tuple): Maximum logo dimensions
            quality (int): JPEG compression quality

        Returns:
            str: Path of compressed logo
        """
        # Resize while maintaining aspect ratio
        logo.thumbnail(max_size, Image.LANCZOS)

        # Convert to RGB if needed
        if logo.mode != 'RGB':
            logo = logo.convert('RGB')

        # Save with optimized compression
        logo.save(output_path, 'JPEG', optimize=True, quality=quality)

        self.logger.info(f"Logo saved at: {output_path}")
        return output_path
