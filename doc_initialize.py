from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")

async def get_preset_list():
    client = AsyncIOMotorClient(MONGO_URL)
    try:
        collection_preset = client["bossworks"]["projects"]
        preset_list = await collection_preset.find(
            {"status": 1}, {"_id": 0, "name": 1, "scope": 1}
        ).to_list(length=None)
        return preset_list
    finally:
        client.close()


agents = [
    {
        "name": "market research",
        "id": 1,
        "scope": "competitor analysis, demographic research",
    },
    {
        "name": "business plan",
        "id": 2,
        "scope": "project reports, pitch decks"
    },
    {
        "name": "financial",
        "id": 3,
        "scope": "pricing strategy, financial projections"
    },
    {
        "name": "marketing",
        "id": 4,
        "scope": "logo creation, tagline suggestions, social media captions"
    },
    {
        "name": "legal",
        "id": 5,
        "scope": "compliance, business registration, funding documentation"
    },
    {
        "name": "procurement",
        "id": 6,
        "scope": "materials sourcing, equipment procurement"
    }
]