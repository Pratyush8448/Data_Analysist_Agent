import asyncio
from api.scraper import scrape_table_from_url

async def test_scraper():
    """Test the scraper function directly"""
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    try:
        print("Testing scraper...")
        result = scrape_table_from_url(url)
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result) if hasattr(result, '__len__') else 'no length'}")
        
        if isinstance(result, tuple) and len(result) == 2:
            df, summary = result
            print(f"Success! DataFrame shape: {df.shape}")
            print(f"Summary: {summary}")
        else:
            print(f"Unexpected result: {result}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scraper()) 