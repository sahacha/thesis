import asyncio
from crawl4ai import AsyncWebCrawler
async def main():
    async with AsyncWebCrawler() as crawler:
        documents = await crawler.crawl_wongnai_places()
        return documents