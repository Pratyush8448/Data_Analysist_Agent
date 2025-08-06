import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Tuple
import re
from io import StringIO


def extract_numeric_from_gross(value: str) -> float:
    """Extract numeric value from gross revenue string like '$2,797,501,328' or '₹1,810.60 crore'"""
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    # Handle different currency formats
    if '₹' in value and 'crore' in value:
        # Indian rupees in crore format
        # Remove ₹, commas, and 'crore', then convert to float
        cleaned = re.sub(r'[₹,\s]', '', value)
        cleaned = cleaned.replace('crore', '')
        # Handle ranges like "1,968.03–2,200" - take the lower bound
        if '–' in cleaned:
            parts = cleaned.split('–')
            try:
                val1 = float(parts[0])
                return val1  # Return lower bound of range
            except ValueError:
                return value
        else:
            try:
                return float(cleaned)
            except ValueError:
                return value
    else:
        # Handle dollar format
        cleaned = re.sub(r'[$,]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return value


def scrape_data(question: str) -> Tuple[pd.DataFrame, str]:
    """
    Extracts a URL from the question and scrapes the most relevant table.
    Returns the cleaned DataFrame and a summary string.
    """
    url = infer_url_from_question(question)
    if not url:
        raise ValueError("❌ Could not infer a URL from the question.")

    df, summary = scrape_table_from_url(url)

    # Append column info for context in analyzer
    col_summary = f" | Cleaned columns: {df.columns.tolist()}"
    preview_summary = f"\n\n🔎 Sample rows:\n{df.head(2).to_markdown(index=False)}"
    return df, summary + col_summary + preview_summary


def infer_url_from_question(question: str) -> str:
    """
    Extracts the first URL from the question.
    Handles punctuation and parenthesis properly.
    """
    url_match = re.search(r'https?://[^\s,)]+', question)
    return url_match.group(0) if url_match else ""


def scrape_table_from_url(url: str) -> Tuple[pd.DataFrame, str]:
    """
    Fetches the page, selects the largest HTML table, cleans it, and returns it as a DataFrame.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"🔌 Failed to connect to {url}: {e}")

    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        raise ValueError(f"❌ No tables found at {url}")

    # For Wikipedia highest-grossing films page, look for the specific table
    best_table = None
    
    for i, table in enumerate(tables):
        # Look for the table with the highest-grossing films data
        table_text = table.get_text().lower()
        
        # Check if this table contains the data we want
        if ('highest-grossing films' in table_text or 
            'rank' in table_text and 'title' in table_text and 'gross' in table_text):
            
            rows = table.find_all("tr")
            if len(rows) > 10:  # Substantial table
                best_table = table
                print(f"DEBUG: Found target table with {len(rows)} rows")
                break
    
    # If we still don't have a good table, try the first substantial table
    if not best_table:
        for table in tables:
            if len(table.find_all("tr")) >= 20:  # Substantial table
                best_table = table
                print(f"DEBUG: Using fallback table with {len(table.find_all('tr'))} rows")
                break
    
    if not best_table:
        raise ValueError(f"❌ No suitable table found at {url}")

    try:
        # Try to parse the table manually first
        print(f"DEBUG: Attempting to parse table with {len(best_table.find_all('tr'))} rows")
        
        # Use pandas read_html with better options
        print(f"DEBUG: Using pd.read_html with better parsing")
        raw_dfs = pd.read_html(StringIO(str(best_table)), flavor="bs4", header=0)
        print(f"DEBUG: Found {len(raw_dfs)} DataFrames")
        
        if not raw_dfs:
            raise ValueError("No DataFrames found in the HTML table")
        
        raw_df = raw_dfs[0]
        print(f"DEBUG: Raw DataFrame shape: {raw_df.shape}")
        print(f"DEBUG: Raw DataFrame columns: {raw_df.columns.tolist()}")
        print(f"DEBUG: Raw DataFrame head:")
        print(raw_df.head())
        
        print(f"DEBUG: Selected DataFrame with shape {raw_df.shape}")
        print(f"DEBUG: DataFrame head:")
        print(raw_df.head())
        
    except Exception as e:
        print(f"DEBUG: Error parsing table: {e}")
        raise ValueError(f"⚠️ Failed to parse table: {e}")

    df = clean_table(raw_df)
    summary = f"✅ Scraped from {url} → {df.shape[0]} rows, {df.shape[1]} columns."
    
    # Ensure we return exactly a tuple with 2 elements
    return (df, summary)


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the scraped DataFrame:
    - Detects and removes multi-row headers
    - Drops unnamed or empty columns
    - Standardizes column names (lowercase, snake_case)
    - Removes rows with junk content (citations, edits)
    - Attempts to convert columns to numeric with NaN fallback
    """
    # Remove completely empty rows and columns
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # If we have very few columns, try to find better structure
    if df.shape[1] <= 2:
        # Try to split columns that might contain multiple pieces of data
        for col in df.columns:
            if isinstance(df[col].iloc[0], str) and '|' in str(df[col].iloc[0]):
                # Split on pipe character
                split_data = df[col].str.split('|', expand=True)
                if split_data.shape[1] > 1:
                    df = pd.concat([df.drop(columns=[col]), split_data], axis=1)
    
    # Detect and flatten multi-row headers
    if df.shape[0] > 1 and df.iloc[0].nunique() > 1:
        # Check if first row looks like headers
        first_row = df.iloc[0].astype(str)
        if any('rank' in val.lower() or 'title' in val.lower() or 'gross' in val.lower() for val in first_row):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)

    # Drop unnamed or empty columns
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.contains("unnamed")]
    df = df.dropna(how="all", axis=1)

    # Standardize column names but preserve meaning
    new_columns = []
    for col in df.columns:
        col_str = str(col).lower().strip()
        # Map common column names to standard names
        if any(keyword in col_str for keyword in ['rank', 'position', 'no', 'number']):
            new_columns.append('rank')
        elif any(keyword in col_str for keyword in ['title', 'film', 'movie', 'name']):
            new_columns.append('title')
        elif any(keyword in col_str for keyword in ['gross', 'revenue', 'earnings', 'box_office']):
            new_columns.append('gross')
        elif any(keyword in col_str for keyword in ['year', 'release', 'date']):
            new_columns.append('year')
        elif any(keyword in col_str for keyword in ['peak', 'position', 'chart']):
            new_columns.append('peak')
        else:
            # Fallback to cleaned name
            new_columns.append(re.sub(r"[^\w\s]", "", col_str).replace(" ", "_"))
    
    df.columns = new_columns

    # Remove rows containing citation/edit footnote patterns
    pattern = re.compile(r"citation needed|note|edit|ref", re.IGNORECASE)
    df = df[~df.apply(lambda row: row.astype(str).str.contains(pattern).any(), axis=1)]

    # Clean string cells and convert numerics only for appropriate columns
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip() if isinstance(x, str) else x)
        
        # Only convert to numeric for columns that should be numeric
        if col.lower() in ['rank', 'peak', 'year']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif col.lower() in ['gross', 'worldwide_gross']:
            # For gross columns, try to extract numeric values
            df[col] = df[col].apply(lambda x: extract_numeric_from_gross(x) if isinstance(x, str) else x)

    return df
