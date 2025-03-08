import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

async def train_model():
    """Sends a request to the /train endpoint to train the model."""
    headers = {"X-API-Key": API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{API_URL}/train", headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            print(response.json())  # Print the response from the API (e.g., {"message": "Model trained successfully"})
        except httpx.HTTPStatusError as e:
            print(f"API Error: {e}")
        except httpx.RequestError as e:
            print(f"Request Error: {e}")

if __name__ == "__main__":
    asyncio.run(train_model())
