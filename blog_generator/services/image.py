import os
import httpx
from blog_generator.interfaces import ImageGeneratorService

class GeminiImageGeneratorService(ImageGeneratorService):
    """
    Gemini-based image generator service using async client.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def generate_image_bytes(self, prompt: str) -> bytes:
        from google import genai
        from google.genai import types

        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        # Create the standard client
        client = genai.Client(api_key=self.api_key)

        # Call asynchronous generate_content via the .aio namespace
        resp = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    )
                ],
            ),
        )

        parts = getattr(resp, "parts", None)
        if not parts and getattr(resp, "candidates", None):
            try:
                parts = resp.candidates[0].content.parts
            except Exception:
                parts = None

        if not parts:
            raise RuntimeError("No image content returned (safety/quota/SDK change).")

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data

        raise RuntimeError("No inline image bytes found in response.")


class OpenAIImageGeneratorService(ImageGeneratorService):
    """
    OpenAI DALL-E based image generator service using AsyncOpenAI.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def generate_image_bytes(self, prompt: str) -> bytes:
        from openai import AsyncOpenAI
        from blog_generator.config import OPENAI_IMAGE_MODEL

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        # Instantiate the async client
        client = AsyncOpenAI(api_key=self.api_key)

        # Call OpenAI DALL-E image generation dynamically based on chosen model
        params = {
            "model": OPENAI_IMAGE_MODEL,
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1
        }
        
        # dall-e-2 does not support quality parameter, others (like dall-e-3, gpt-image-2) do
        if OPENAI_IMAGE_MODEL != "dall-e-2":
            params["quality"] = "auto"

        response = await client.images.generate(**params)

        # Handle URL output (standard for DALL-E)
        image_url = getattr(response.data[0], "url", None)
        if image_url:
            async with httpx.AsyncClient() as http_client:
                image_response = await http_client.get(image_url)
                image_response.raise_for_status()
                return image_response.content

        # Handle base64 output (standard for gpt-image-2)
        b64_json = getattr(response.data[0], "b64_json", None)
        if b64_json:
            import base64
            return base64.b64decode(b64_json)

        raise RuntimeError("No image URL or base64 data returned from OpenAI.")
