import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Model configurations
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini")

# Shared LLM Instance
llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL)

# Provider configurations
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "gemini").lower()
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "tavily").lower()
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "dall-e-3").lower()

# API Keys validation helpers
def get_tavily_api_key() -> str:
    return os.environ.get("TAVILY_API_KEY", "")

def get_google_api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "")
