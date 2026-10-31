from blog_generator.config import IMAGE_PROVIDER, SEARCH_PROVIDER
from blog_generator.interfaces import SearchService, ImageGeneratorService
from blog_generator.services.search import TavilySearchService
from blog_generator.services.image import GeminiImageGeneratorService, OpenAIImageGeneratorService

class ServiceFactory:
    """
    Factory class to instantiate and manage services.
    Conforms to the Factory design pattern, decoupling graph nodes from 
    concrete implementation choices and environment settings.
    """

    @staticmethod
    def get_search_service(provider: str = None) -> SearchService:
        """
        Returns a concrete SearchService instance based on config or provided provider name.
        """
        provider_name = (provider or SEARCH_PROVIDER).lower()
        if provider_name == "tavily":
            return TavilySearchService()
        else:
            raise ValueError(f"Unknown search provider: {provider_name}")

    @staticmethod
    def get_image_generator(provider: str = None) -> ImageGeneratorService:
        """
        Returns a concrete ImageGeneratorService instance based on config or provided provider name.
        """
        provider_name = (provider or IMAGE_PROVIDER).lower()
        if provider_name == "gemini":
            return GeminiImageGeneratorService()
        elif provider_name in ("openai", "dall-e", "dalle"):
            return OpenAIImageGeneratorService()
        else:
            raise ValueError(f"Unknown image generator provider: {provider_name}")