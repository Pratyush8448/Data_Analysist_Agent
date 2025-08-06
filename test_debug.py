import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.scraper import scrape_table_from_url
from api.analyzer import analyze_task

async def test_scraping():
    print("🔍 Testing scraping...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        df, summary = scrape_table_from_url(url)
        print(f"✅ Scraping successful: {summary}")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        print(f"🔍 First 5 rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return None

async def test_analysis():
    print("\n🔍 Testing analysis...")
    try:
        task = "Scrape the list of highest grossing films from Wikipedia. It is at the URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_films\n\nAnswer the following questions and respond with a JSON array of strings containing the answer.\n\n1. How many $2 bn movies were released before 2020?\n2. Which is the earliest film that grossed over $1.5 bn?\n3. What is the correlation between the Rank and Peak?"
        results = await analyze_task(task)
        print(f"✅ Analysis successful: {len(results)} results")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result}")
        return results
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(test_scraping())
    asyncio.run(test_analysis()) 