import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://g1.globo.com/rj/rio-de-janeiro/noticia/2026/05/10/governo-do-rj-desistiu-de-comprar-black-hawk-de-r-617-milhoes-ao-desconfiar-que-helicoptero-de-guerra-era-usado.ghtml")
        print(result.markdown)  # Print first 300 chars

if __name__ == "__main__":
    asyncio.run(main())
