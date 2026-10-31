from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SearchService(ABC):
    """
    Interface for search engines or tools to retrieve online evidence.
    Conforms to OCP & LSP - now supporting async operations.
    """
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform an asynchronous web search and return a list of normalized dictionary results.
        """
        pass


class ImageGeneratorService(ABC):
    """
    Interface for image generation models.
    Conforms to DIP - now supporting async operations.
    """
    @abstractmethod
    async def generate_image_bytes(self, prompt: str) -> bytes:
        """
        Perform an asynchronous image generation and return raw image bytes.
        """
        pass
