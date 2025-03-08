import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")


async def get_model_history():
    """Fetches model history from the API and prints it as formatted JSON."""
    headers = {"X-API-Key": API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/model_history", headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            model_history = response.json()
            print(json.dumps(model_history, indent=2))  # Pretty print JSON
        except httpx.HTTPStatusError as e:
            print(f"API Error: {e}")
        except httpx.RequestError as e:
            print(f"Request Error: {e}")


if __name__ == "__main__":
    asyncio.run(get_model_history())
