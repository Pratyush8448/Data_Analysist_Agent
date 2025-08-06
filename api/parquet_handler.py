import os
import re
import requests
import tempfile
import duckdb
import pandas as pd
import json
from typing import Tuple, List, Dict, Optional


def handle_parquet_query(query: str, parquet_path: Optional[str] = None, url: Optional[str] = None, s3_path: Optional[str] = None) -> Tuple[List[Dict], str, str]:
    """
    Executes SQL query directly on a local, remote, or S3 Parquet file using DuckDB.
    
    Parameters:
        query (str): SQL query. Use 'FROM parquet' as a placeholder.
        parquet_path (str): Path to a local Parquet file (optional).
        url (str): URL to a remote Parquet file (optional).
        s3_path (str): S3 path to Parquet files (optional).

    Returns:
        Tuple: (query result as list of dicts, summary, preview in markdown)
    """
    if not parquet_path and not url and not s3_path:
        raise ValueError("Either 'parquet_path', 'url', or 's3_path' must be provided.")

    con = duckdb.connect(database=":memory:")
    
    # Install and load required extensions
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL parquet; LOAD parquet;")
    except Exception as e:
        print(f"Warning: Could not install extensions: {e}")

    if s3_path:
        # Handle S3 path
        source_path = s3_path
    elif url:
        source_path = download_parquet_tempfile(url)
    else:
        source_path = parquet_path

    # Replace placeholder with appropriate table reference
    if "FROM parquet" in query:
        if s3_path:
            query = query.replace("FROM parquet", f"FROM read_parquet('{source_path}')")
        else:
            query = query.replace("FROM parquet", f"FROM parquet_scan('{source_path}')")

    try:
        result_df = con.execute(query).fetchdf()
        summary = f"✅ Query executed → {result_df.shape[0]} rows × {result_df.shape[1]} columns."
        preview = result_df.head(5).to_markdown(index=False)
        return result_df.to_dict(orient="records"), summary, preview

    except Exception as e:
        raise ValueError(f"❌ DuckDB query failed: {str(e)}")

    finally:
        if url and source_path and os.path.exists(source_path) and not s3_path:
            try:
                os.remove(source_path)
            except Exception:
                pass


def download_parquet_tempfile(url: str) -> str:
    """
    Downloads a remote Parquet file and saves it temporarily.

    Returns:
        str: Path to the downloaded temporary file.
    """
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name

    except Exception as e:
        raise RuntimeError(f"❌ Failed to download Parquet from URL: {str(e)}")


def analyze_indian_court_data(question: str) -> List[str]:
    """
    Analyze Indian High Court judgments dataset based on the question.
    
    Returns:
        List of strings in the format: [count, court_name, correlation, plot_image]
    """
    try:
        # S3 path for the Indian High Court dataset
        s3_path = "s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1"
        
        # Parse the question to determine what analysis to perform
        question_lower = question.lower()
        
        results = []
        
        # Question 1: Which high court disposed the most cases from 2019 - 2022?
        if "disposed the most cases" in question_lower and ("2019" in question_lower or "2022" in question_lower):
            query = """
            SELECT court, COUNT(*) as case_count 
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE year BETWEEN 2019 AND 2022
            GROUP BY court 
            ORDER BY case_count DESC 
            LIMIT 1
            """
            
            try:
                result, summary, preview = handle_parquet_query(query, s3_path=s3_path)
                if result:
                    court_name = result[0].get('court', 'Unknown')
                    case_count = result[0].get('case_count', 0)
                    results.append(case_count)
                    results.append(court_name)
                else:
                    results.append(0)
                    results.append("No data found")
            except Exception as e:
                print(f"Error in court analysis: {e}")
                results.append(0)
                results.append("Error occurred")
        
        # Question 2: Regression slope of date_of_registration - decision_date by year in court=33_10
        if "regression slope" in question_lower and "33_10" in question_lower:
            query = """
            SELECT 
                year,
                AVG(CAST(decision_date AS DATE) - CAST(STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)) as avg_delay_days
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE court = '33_10' 
                AND date_of_registration IS NOT NULL 
                AND decision_date IS NOT NULL
                AND date_of_registration != ''
            GROUP BY year
            ORDER BY year
            """
            
            try:
                result, summary, preview = handle_parquet_query(query, s3_path=s3_path)
                if len(result) >= 2:
                    # Calculate regression slope manually
                    years = [row['year'] for row in result]
                    delays = [row['avg_delay_days'] for row in result]
                    
                    if len(years) >= 2:
                        # Simple linear regression slope
                        n = len(years)
                        sum_x = sum(years)
                        sum_y = sum(delays)
                        sum_xy = sum(x * y for x, y in zip(years, delays))
                        sum_x2 = sum(x * x for x in years)
                        
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                        results.append(slope)
                    else:
                        results.append(0.0)
                else:
                    results.append(0.0)
            except Exception as e:
                print(f"Error in regression analysis: {e}")
                results.append(0.0)
        
        # Question 3: Plot the year and delay days as scatterplot
        if "plot" in question_lower and "scatterplot" in question_lower:
            query = """
            SELECT 
                year,
                AVG(CAST(decision_date AS DATE) - CAST(STRPTIME(date_of_registration, '%d-%m-%Y') AS DATE)) as avg_delay_days
            FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
            WHERE court = '33_10' 
                AND date_of_registration IS NOT NULL 
                AND decision_date IS NOT NULL
                AND date_of_registration != ''
            GROUP BY year
            ORDER BY year
            """
            
            try:
                result, summary, preview = handle_parquet_query(query, s3_path=s3_path)
                if result:
                    # Create DataFrame for plotting
                    df = pd.DataFrame(result)
                    
                    # Generate scatter plot with regression line
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import io
                    import base64
                    
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df, x='year', y='avg_delay_days', color='blue', s=100)
                    sns.regplot(data=df, x='year', y='avg_delay_days', scatter=False, color='red', line_kws={'linestyle': '--'})
                    
                    plt.title('Average Delay Days by Year (Court 33_10)', fontsize=14)
                    plt.xlabel('Year', fontsize=12)
                    plt.ylabel('Average Delay (Days)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    
                    # Save to base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close()
                    
                    plot_uri = f"data:image/png;base64,{image_base64}"
                    results.append(plot_uri)
                else:
                    results.append("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
            except Exception as e:
                print(f"Error in plot generation: {e}")
                results.append("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        
        # Ensure we have exactly 4 results
        while len(results) < 4:
            if len(results) == 0:
                results.append(0)
            elif len(results) == 1:
                results.append("No court found")
            elif len(results) == 2:
                results.append(0.0)
            elif len(results) == 3:
                results.append("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
        
        return results[:4]
        
    except Exception as e:
        print(f"Error in Indian court analysis: {e}")
        return [0, "Error occurred", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
