import re
import asyncio
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Optional
from rapidfuzz import process
from api.scraper import scrape_table_from_url
from api.plotter import plot_scatter


def extract_url(text: str) -> str:
    urls = re.findall(r'https?://[^\s]+', text)
    if not urls:
        raise ValueError("❌ No URL found in the question.")
    return urls[0].strip(").,\'\"")
    
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and convert numeric columns"""
    # Clean column names
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    
    # Convert numeric columns
    for col in df.columns:
        if col in ['rank', 'peak', 'gross', 'worldwide_gross', 'domestic_gross', 'year']:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def fuzzy_col_match(keyword: str, columns: List[str], threshold: int = 70) -> Optional[str]:
    try:
        result = process.extractOne(keyword.lower(), columns)
        if result is None:
            return None
        match, score = result
        return match if score >= threshold else None
    except Exception as e:
        print(f"DEBUG: Error in fuzzy_col_match: {e}")
        return None
    

def extract_conditions(task: str, df: pd.DataFrame) -> List[Tuple[str, float, str]]:
    task = task.lower()
    filters = []

    match_definitions = [
        (r"\$([\d.]+)\s*(billion|million)?", "gross", ">="),
        (r"before\s+(\d{4})", "year", "<"),
        (r"after\s+(\d{4})", "year", ">"),
        (r"in\s+(\d{4})", "year", "="),
    ]

    for pattern, keyword, operator in match_definitions:
        matches = re.findall(pattern, task)
        if not matches:
            continue

        col = fuzzy_col_match(keyword, df.columns)
        if not col:
            continue

        for match in matches:
            if isinstance(match, tuple):
                val = float(match[0])
                unit = match[1]
                if unit == 'billion':
                    val *= 1e9
                elif unit == 'million':
                    val *= 1e6
            else:
                val = float(match) if '.' in match else int(match)
            filters.append((col, val, operator))

    return filters


def apply_conditions(df: pd.DataFrame, conditions: List[Tuple[str, float, str]]) -> pd.DataFrame:
    for col, val, op in conditions:
        if col not in df.columns:
            continue
        if op == ">=":
            df = df[df[col] >= val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "=":
            df = df[df[col] == val]
    return df


def compute_correlation(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return 0.0
    df_clean = df[[x_col, y_col]].dropna()
    if df_clean.empty:
        return 0.0
    corr_val = df_clean[x_col].corr(df_clean[y_col])
    return corr_val if not pd.isna(corr_val) else 0.0


async def analyze_task(task: str) -> List[str]:
    """
    Analyzes a natural language task involving scraping and analyzing tabular data.

    Returns:
        List of strings in the format: [count, movie_name, correlation, plot_image]
    """
    results = []

    try:
        url = extract_url(task)
        loop = asyncio.get_running_loop()
        
        # Create a wrapper function to handle the scraper result properly
        def scrape_wrapper(url):
            try:
                result = scrape_table_from_url(url)
                print(f"DEBUG: Direct scraper result type: {type(result)}")
                return result
            except Exception as e:
                print(f"DEBUG: Direct scraper error: {e}")
                raise e
        
        # Call the wrapper function
        try:
            result = await loop.run_in_executor(None, scrape_wrapper, url)
            print(f"DEBUG: Async scraper result type: {type(result)}")
            print(f"DEBUG: Async scraper result length: {len(result) if hasattr(result, '__len__') else 'no length'}")
            
            # Handle the result properly
            if isinstance(result, tuple) and len(result) == 2:
                df, summary = result
            elif isinstance(result, list) and len(result) == 2:
                df, summary = result[0], result[1]
            elif isinstance(result, pd.DataFrame):
                df = result
                summary = "Data scraped successfully"
            else:
                print(f"DEBUG: Unexpected result: {result}")
                # Try to extract DataFrame and summary from the result
                if hasattr(result, '__getitem__'):
                    try:
                        df = result[0]
                        summary = result[1] if len(result) > 1 else "Data scraped successfully"
                    except:
                        raise ValueError(f"Unexpected result type from scraper: {type(result)}")
                else:
                    raise ValueError(f"Unexpected result type from scraper: {type(result)}")
        except Exception as e:
            print(f"DEBUG: Error in async scraper: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Check if we got valid data
        if df is None or df.empty:
            return [0, "No data found", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
            
        df = normalize_dataframe(df)
        columns = df.columns.tolist()

        # Initialize results with default values
        count_result = 0
        movie_result = "No movie found"
        correlation_result = 0.0
        plot_result = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # Question 1: How many $X bn movies were released before YYYY?
        if "how many" in task.lower() and "$" in task.lower():
            conditions = extract_conditions(task, df)
            filtered_df = apply_conditions(df.copy(), conditions)
            count_result = len(filtered_df)

        # Question 2: Which is the earliest film that grossed over $X bn?
        if "earliest" in task.lower() and ("grossed over" in task.lower() or "gross over" in task.lower()):
            col_gross = fuzzy_col_match("gross", columns) or fuzzy_col_match("worldwide_gross", columns)
            col_year = fuzzy_col_match("year", columns)
            col_title = fuzzy_col_match("title", columns) or fuzzy_col_match("film", columns) or fuzzy_col_match("movie", columns)

            if col_gross and col_year:
                # Extract the threshold from the question
                threshold_match = re.search(r'\$([\d.]+)\s*(billion|million)?', task.lower())
                if threshold_match:
                    threshold_val = float(threshold_match.group(1))
                    if threshold_match.group(2) == 'billion':
                        threshold = threshold_val * 1e9
                    elif threshold_match.group(2) == 'million':
                        threshold = threshold_val * 1e6
                    else:
                        threshold = threshold_val * 1e9  # Default to billion
                else:
                    threshold = 1.5e9  # Default threshold
                
                filtered_df = df[df[col_gross] > threshold]
                if not filtered_df.empty:
                    earliest = filtered_df.sort_values(by=col_year).iloc[0]
                    movie_result = str(earliest.get(col_title, "Unknown")) if col_title else "Unknown"

        # Question 3: What's the correlation between Rank and Peak?
        if "correlation" in task.lower():
            x = fuzzy_col_match("rank", columns)
            y = fuzzy_col_match("peak", columns)
            if x and y:
                correlation_result = compute_correlation(df, x, y)

        # Generate plot for correlation/visualization
        x = fuzzy_col_match("rank", columns)
        y = fuzzy_col_match("peak", columns) or fuzzy_col_match("gross", columns) or fuzzy_col_match("worldwide_gross", columns)
        
        if x and y:
            try:
                loop = asyncio.get_running_loop()
                plot_uri = await loop.run_in_executor(None, plot_scatter, df, x, y, True)
                plot_result = plot_uri
            except Exception as plot_error:
                plot_result = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # Return results in the expected format
        results = [count_result, movie_result, correlation_result, plot_result]

    except Exception as e:
        results = [0, f"Error: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

    return results
