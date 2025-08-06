import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def test_wikipedia_scraping():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    print("🔍 Testing Wikipedia scraping...")
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table")
        
        print(f"Found {len(tables)} tables")
        
        for i, table in enumerate(tables):
            rows = table.find_all("tr")
            print(f"Table {i}: {len(rows)} rows")
            
            if len(rows) > 5:  # Only look at substantial tables
                # Try to read the table
                try:
                    df = pd.read_html(StringIO(str(table)), flavor="bs4")[0]
                    print(f"  Table {i} shape: {df.shape}")
                    print(f"  Table {i} columns: {df.columns.tolist()}")
                    print(f"  Table {i} first row: {df.iloc[0].tolist()}")
                    print("---")
                except Exception as e:
                    print(f"  Table {i} failed to parse: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_wikipedia_scraping() 