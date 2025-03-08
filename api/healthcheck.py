import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

async def check_health():
    """Checks the health of the API and prints the result."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/healthcheck")
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            print(response.json())  # Print the response from the API
        except httpx.HTTPStatusError as e:
            print(f"API Error: {e}")
        except httpx.RequestError as e:
            print(f"Request Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_health())
