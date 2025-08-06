import re
import pandas as pd
import numpy as np
from typing import List
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
    # Don't override column names if they're already meaningful
    if not any(col.isdigit() for col in df.columns):
        # Columns are already meaningful, just convert to lowercase
        df.columns = [col.lower().strip() for col in df.columns]
    else:
        # Columns are generic numbers, try to map them
        print(f"DEBUG: Original columns: {df.columns.tolist()}")
        print(f"DEBUG: First few rows to analyze:")
        print(df.head(5))
        
        # Try to infer column meanings from the first few rows
        new_columns = []
        for i, col in enumerate(df.columns):
            # Look at the first few values to guess the column type
            sample_values = df[col].head(5).astype(str).str.lower()
            sample_text = ' '.join(sample_values)
            
            if any('rank' in val or 'position' in val or 'no' in val for val in sample_values):
                new_columns.append('rank')
            elif any('title' in val or 'film' in val or 'movie' in val for val in sample_values):
                new_columns.append('title')
            elif any('gross' in val or 'revenue' in val or '$' in val or 'billion' in val for val in sample_values):
                new_columns.append('gross')
            elif any('year' in val or 'release' in val or '19' in val or '20' in val for val in sample_values):
                new_columns.append('year')
            elif any('peak' in val or 'chart' in val for val in sample_values):
                new_columns.append('peak')
            else:
                # Try to guess based on position and content
                if i == 0:  # First column is usually rank
                    new_columns.append('rank')
                elif i == 1:  # Second column is usually title
                    new_columns.append('title')
                elif i == 2:  # Third column might be gross
                    new_columns.append('gross')
                elif i == 3:  # Fourth column might be year
                    new_columns.append('year')
                else:
                    new_columns.append(f'col_{i}')
        df.columns = new_columns
        print(f"DEBUG: Mapped columns: {df.columns.tolist()}")
    
    # Convert numeric columns
    for col in df.columns:
        if col in ['rank', 'peak', 'gross', 'worldwide_gross', 'domestic_gross', 'year']:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def fuzzy_col_match(keyword: str, columns: List[str], threshold: int = 70) -> str:
    """Simple column matching without rapidfuzz"""
    try:
        # First try exact match
        for col in columns:
            if keyword.lower() in col.lower():
                return col
        
        # Then try partial match
        for col in columns:
            if any(word in col.lower() for word in keyword.lower().split()):
                return col
        
        # Try common variations
        keyword_variations = {
            'rank': ['rank', 'position', 'no', 'number', 'rank'],
            'title': ['title', 'film', 'movie', 'name', 'title'],
            'gross': ['gross', 'revenue', 'earnings', 'box_office', 'worldwide_gross'],
            'year': ['year', 'release', 'date', 'year'],
            'peak': ['peak', 'position', 'chart', 'peak']
        }
        
        if keyword.lower() in keyword_variations:
            for variation in keyword_variations[keyword.lower()]:
                for col in columns:
                    if variation in col.lower():
                        return col
        
        return None
    except Exception as e:
        print(f"DEBUG: Error in fuzzy_col_match: {e}")
        return None
    

def extract_conditions(task: str, df: pd.DataFrame) -> List[tuple]:
    task = task.lower()
    filters = []

    match_definitions = [
        (r"\$([\d.]+)\s*(billion|million)?", "gross", ">="),
        (r"(\d+)\s*crore", "gross", ">="),
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
            print(f"DEBUG: Processing match: {match}")
            if isinstance(match, tuple):
                val = float(match[0])
                unit = match[1]
                print(f"DEBUG: Tuple match - val: {val}, unit: {unit}")
                if unit == 'billion':
                    val *= 1e9
                elif unit == 'million':
                    val *= 1e6
            else:
                # Handle crore values (1 crore = 10 million)
                if 'crore' in task.lower():
                    val = float(match) * 1e7  # 1 crore = 10 million = 10,000,000
                    print(f"DEBUG: Crore match - val: {val}")
                else:
                    val = float(match) if '.' in match else int(match)
                    print(f"DEBUG: Regular match - val: {val}")
            filters.append((col, val, operator))

    return filters


def apply_conditions(df: pd.DataFrame, conditions: List[tuple]) -> pd.DataFrame:
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


def analyze_task_simple(task: str) -> List[str]:
    """
    Simplified version without async for testing
    """
    results = []

    try:
        url = extract_url(task)
        print(f"DEBUG: Extracted URL: {url}")
        
        # Call scraper directly (no async)
        result = scrape_table_from_url(url)
        print(f"DEBUG: Scraper result type: {type(result)}")
        
        if isinstance(result, tuple) and len(result) == 2:
            df, summary = result
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        print(f"DEBUG: Got DataFrame with shape: {df.shape}")
        
        # Check if we got valid data
        if df is None or df.empty:
            return [0, "No data found", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
            
        df = normalize_dataframe(df)
        columns = df.columns.tolist()
        print(f"DEBUG: Normalized columns: {columns}")

        # Initialize results with default values
        count_result = 0
        movie_result = "No movie found"
        correlation_result = 0.0
        plot_result = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        # Question 1: How many $X bn movies were released before YYYY?
        if "how many" in task.lower() and "$" in task.lower():
            print(f"DEBUG: Processing 'how many' question")
            conditions = extract_conditions(task, df)
            print(f"DEBUG: Extracted conditions: {conditions}")
            filtered_df = apply_conditions(df.copy(), conditions)
            count_result = len(filtered_df)
            print(f"DEBUG: Count result: {count_result}")
            print(f"DEBUG: Original df shape: {df.shape}, Filtered df shape: {filtered_df.shape}")

        # Question 2: Which is the earliest film that grossed over $X bn?
        if "earliest" in task.lower() and ("grossed over" in task.lower() or "gross over" in task.lower()):
            print(f"DEBUG: Processing 'earliest' question")
            col_gross = fuzzy_col_match("gross", columns) or fuzzy_col_match("worldwide_gross", columns)
            col_year = fuzzy_col_match("year", columns)
            col_title = fuzzy_col_match("title", columns) or fuzzy_col_match("film", columns) or fuzzy_col_match("movie", columns)
            
            print(f"DEBUG: Found columns - gross: {col_gross}, year: {col_year}, title: {col_title}")

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
                
                print(f"DEBUG: Using threshold: {threshold}")
                filtered_df = df[df[col_gross] > threshold]
                print(f"DEBUG: Filtered df shape: {filtered_df.shape}")
                
                if not filtered_df.empty:
                    earliest = filtered_df.sort_values(by=col_year).iloc[0]
                    movie_result = str(earliest.get(col_title, "Unknown")) if col_title else "Unknown"
                    print(f"DEBUG: Movie result: {movie_result}")
                else:
                    print(f"DEBUG: No movies found above threshold")
            else:
                print(f"DEBUG: Missing required columns - gross: {col_gross}, year: {col_year}")

        # Question 3: What's the correlation between Rank and Peak?
        if "correlation" in task.lower():
            print(f"DEBUG: Processing 'correlation' question")
            x = fuzzy_col_match("rank", columns)
            y = fuzzy_col_match("peak", columns)
            print(f"DEBUG: Found columns for correlation - x: {x}, y: {y}")
            if x and y:
                correlation_result = compute_correlation(df, x, y)
                print(f"DEBUG: Correlation result: {correlation_result}")
            else:
                print(f"DEBUG: Missing columns for correlation")

        # Generate plot for correlation/visualization
        print(f"DEBUG: Generating plot")
        x = fuzzy_col_match("rank", columns)
        y = fuzzy_col_match("peak", columns) or fuzzy_col_match("gross", columns) or fuzzy_col_match("worldwide_gross", columns)
        print(f"DEBUG: Plot columns - x: {x}, y: {y}")
        
        if x and y:
            try:
                plot_uri = plot_scatter(df, x, y, True)
                plot_result = plot_uri
                print(f"DEBUG: Plot generated successfully")
            except Exception as plot_error:
                print(f"DEBUG: Plot error: {plot_error}")
                plot_result = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        else:
            print(f"DEBUG: Missing columns for plot")

        # Return results in the expected format
        results = [count_result, movie_result, correlation_result, plot_result]
        print(f"DEBUG: Final results: {results}")

    except Exception as e:
        print(f"DEBUG: Error in simple analyzer: {e}")
        import traceback
        traceback.print_exc()
        results = [0, f"Error: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

    return results 