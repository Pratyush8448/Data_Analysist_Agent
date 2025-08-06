import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import duckdb
import json
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from urllib.parse import urlparse
import tempfile
import os

class UniversalDataAnalyzer:
    """
    A universal data analyzer that can handle any data source and question type.
    """
    
    def __init__(self):
        self.supported_formats = {
            'csv': self._load_csv,
            'json': self._load_json,
            'excel': self._load_excel,
            'parquet': self._load_parquet,
            'html': self._scrape_html_table,
            'wikipedia': self._scrape_wikipedia
        }
    
    def analyze_question(self, question: str, data_source: str = None) -> List[str]:
        """
        Analyze any question with any data source.
        
        Args:
            question: Natural language question
            data_source: URL, file path, or data description
            
        Returns:
            List of 4 elements: [count, name, correlation, plot_image]
        """
        try:
            # Step 1: Parse the question to understand what's needed
            parsed_question = self._parse_question(question)
            
            # Step 2: Load data from the source
            df = self._load_data(data_source, question)
            
            if df is None or df.empty:
                return [0, "No data found", 0.0, self._get_placeholder_image()]
            
            # Step 3: Perform the requested analysis
            results = self._perform_analysis(df, parsed_question)
            
            # Step 4: Ensure we have exactly 4 results
            return self._format_results(results)
            
        except Exception as e:
            print(f"Error in universal analysis: {e}")
            return [0, f"Error: {str(e)}", 0.0, self._get_placeholder_image()]
    
    def _parse_question(self, question: str) -> Dict[str, Any]:
        """Parse natural language question into structured analysis request."""
        question_lower = question.lower()
        
        analysis = {
            'type': 'unknown',
            'conditions': [],
            'columns': [],
            'visualization': None,
            'aggregation': None
        }
        
        # Detect question types
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            analysis['type'] = 'count'
        elif any(word in question_lower for word in ['earliest', 'oldest', 'first']):
            analysis['type'] = 'min'
        elif any(word in question_lower for word in ['latest', 'newest', 'last']):
            analysis['type'] = 'max'
        elif any(word in question_lower for word in ['correlation', 'relationship']):
            analysis['type'] = 'correlation'
        elif any(word in question_lower for word in ['average', 'mean', 'median']):
            analysis['type'] = 'aggregation'
            if 'average' in question_lower or 'mean' in question_lower:
                analysis['aggregation'] = 'mean'
            elif 'median' in question_lower:
                analysis['aggregation'] = 'median'
        
        # Detect visualization requests
        if any(word in question_lower for word in ['plot', 'chart', 'graph', 'scatter', 'histogram']):
            if 'scatter' in question_lower:
                analysis['visualization'] = 'scatter'
            elif 'histogram' in question_lower:
                analysis['visualization'] = 'histogram'
            elif 'bar' in question_lower:
                analysis['visualization'] = 'bar'
            else:
                analysis['visualization'] = 'scatter'  # default
        
        # Extract column names (basic pattern matching)
        # This is a simplified version - you'd want more sophisticated NLP here
        words = question_lower.split()
        for i, word in enumerate(words):
            if word in ['of', 'between', 'and', 'vs', 'versus'] and i > 0 and i < len(words) - 1:
                analysis['columns'].extend([words[i-1], words[i+1]])
        
        # Extract conditions (basic pattern matching)
        if '$' in question:
            # Money conditions
            money_match = re.search(r'\$([\d.]+)\s*(billion|million)?', question)
            if money_match:
                amount = float(money_match.group(1))
                unit = money_match.group(2) or 'billion'
                if unit == 'billion':
                    amount *= 1e9
                elif unit == 'million':
                    amount *= 1e6
                analysis['conditions'].append(('amount', '>=', amount))
        
        if 'before' in question_lower or 'after' in question_lower:
            year_match = re.search(r'(before|after)\s+(\d{4})', question_lower)
            if year_match:
                operator = '<' if year_match.group(1) == 'before' else '>'
                year = int(year_match.group(2))
                analysis['conditions'].append(('year', operator, year))
        
        return analysis
    
    def _load_data(self, data_source: str, question: str) -> Optional[pd.DataFrame]:
        """Load data from various sources."""
        if not data_source:
            # Try to extract URL from question
            url_match = re.search(r'https?://[^\s]+', question)
            if url_match:
                data_source = url_match.group(0)
            else:
                return None
        
        # Determine data format
        data_format = self._detect_format(data_source)
        
        # Load data using appropriate method
        if data_format in self.supported_formats:
            return self.supported_formats[data_format](data_source)
        
        return None
    
    def _detect_format(self, data_source: str) -> str:
        """Detect the format of the data source."""
        if data_source.startswith('http'):
            if 'wikipedia.org' in data_source:
                return 'wikipedia'
            elif data_source.endswith('.csv'):
                return 'csv'
            elif data_source.endswith('.json'):
                return 'json'
            elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                return 'excel'
            elif data_source.endswith('.parquet'):
                return 'parquet'
            else:
                return 'html'  # Try to scrape as HTML table
        else:
            # Local file
            if data_source.endswith('.csv'):
                return 'csv'
            elif data_source.endswith('.json'):
                return 'json'
            elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                return 'excel'
            elif data_source.endswith('.parquet'):
                return 'parquet'
        
        return 'unknown'
    
    def _load_csv(self, source: str) -> pd.DataFrame:
        """Load CSV data."""
        try:
            if source.startswith('http'):
                response = requests.get(source)
                response.raise_for_status()
                return pd.read_csv(io.StringIO(response.text))
            else:
                return pd.read_csv(source)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    def _load_json(self, source: str) -> pd.DataFrame:
        """Load JSON data."""
        try:
            if source.startswith('http'):
                response = requests.get(source)
                response.raise_for_status()
                data = response.json()
            else:
                with open(source, 'r') as f:
                    data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    return pd.DataFrame(data['data'])
                else:
                    return pd.DataFrame([data])
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return pd.DataFrame()
    
    def _load_excel(self, source: str) -> pd.DataFrame:
        """Load Excel data."""
        try:
            if source.startswith('http'):
                response = requests.get(source)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                df = pd.read_excel(tmp_path)
                os.unlink(tmp_path)
                return df
            else:
                return pd.read_excel(source)
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return pd.DataFrame()
    
    def _load_parquet(self, source: str) -> pd.DataFrame:
        """Load Parquet data."""
        try:
            con = duckdb.connect(database=":memory:")
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute("INSTALL parquet; LOAD parquet;")
            
            if source.startswith('s3://'):
                df = con.execute(f"SELECT * FROM read_parquet('{source}')").fetchdf()
            else:
                df = con.execute(f"SELECT * FROM read_parquet('{source}')").fetchdf()
            
            return df
        except Exception as e:
            print(f"Error loading Parquet: {e}")
            return pd.DataFrame()
    
    def _scrape_html_table(self, url: str) -> pd.DataFrame:
        """Scrape HTML table from any website."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if not tables:
                return pd.DataFrame()
            
            # Find the largest table
            largest_table = max(tables, key=lambda t: len(t.find_all('tr')))
            
            # Parse table
            df = pd.read_html(str(largest_table))[0]
            return df
        except Exception as e:
            print(f"Error scraping HTML table: {e}")
            return pd.DataFrame()
    
    def _scrape_wikipedia(self, url: str) -> pd.DataFrame:
        """Scrape Wikipedia table (enhanced version)."""
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table', class_='wikitable')
            
            if not tables:
                # Try any table
                tables = soup.find_all('table')
            
            if not tables:
                return pd.DataFrame()
            
            # Find the most relevant table
            best_table = None
            for table in tables:
                table_text = table.get_text().lower()
                if len(table.find_all('tr')) > 5:  # Substantial table
                    best_table = table
                    break
            
            if best_table:
                df = pd.read_html(str(best_table))[0]
                return self._clean_dataframe(df)
            
            return pd.DataFrame()
        except Exception as e:
            print(f"Error scraping Wikipedia: {e}")
            return pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize DataFrame."""
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Clean column names
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        
        # Try to convert numeric columns
        for col in df.columns:
            if any(keyword in col for keyword in ['rank', 'count', 'number', 'year', 'amount', 'gross', 'revenue']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _perform_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[Any]:
        """Perform the requested analysis on the DataFrame."""
        results = []
        
        try:
            if analysis['type'] == 'count':
                # Apply conditions and count
                filtered_df = self._apply_conditions(df, analysis['conditions'])
                results.append(len(filtered_df))
                
                # Get name of first item (if available)
                name_col = self._find_name_column(df.columns)
                if name_col and not filtered_df.empty:
                    results.append(str(filtered_df.iloc[0][name_col]))
                else:
                    results.append("No items found")
            
            elif analysis['type'] in ['min', 'max']:
                # Find earliest/latest
                date_col = self._find_date_column(df.columns)
                if date_col:
                    if analysis['type'] == 'min':
                        earliest = df.loc[df[date_col].idxmin()]
                    else:
                        earliest = df.loc[df[date_col].idxmax()]
                    
                    name_col = self._find_name_column(df.columns)
                    results.append(1)  # count
                    results.append(str(earliest[name_col]) if name_col else "Unknown")
                else:
                    results.extend([0, "No date column found"])
            
            elif analysis['type'] == 'correlation':
                # Calculate correlation
                if len(analysis['columns']) >= 2:
                    col1, col2 = analysis['columns'][:2]
                    corr = df[col1].corr(df[col2])
                    results.extend([0, "Correlation calculated", corr])
                else:
                    results.extend([0, "No columns specified", 0.0])
            
            elif analysis['type'] == 'aggregation':
                # Calculate aggregation
                if analysis['columns']:
                    col = analysis['columns'][0]
                    if analysis['aggregation'] == 'mean':
                        result = df[col].mean()
                    elif analysis['aggregation'] == 'median':
                        result = df[col].median()
                    else:
                        result = df[col].mean()
                    results.extend([0, f"{analysis['aggregation']} calculated", result])
                else:
                    results.extend([0, "No column specified", 0.0])
            
            else:
                # Default: just count
                results.extend([len(df), "Data loaded", 0.0])
            
            # Generate visualization if requested
            if analysis['visualization']:
                plot_image = self._generate_visualization(df, analysis)
                results.append(plot_image)
            else:
                results.append(self._get_placeholder_image())
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            results = [0, f"Analysis error: {str(e)}", 0.0, self._get_placeholder_image()]
        
        return results
    
    def _apply_conditions(self, df: pd.DataFrame, conditions: List[tuple]) -> pd.DataFrame:
        """Apply filtering conditions to DataFrame."""
        filtered_df = df.copy()
        
        for col, operator, value in conditions:
            if col in filtered_df.columns:
                if operator == '>=':
                    filtered_df = filtered_df[filtered_df[col] >= value]
                elif operator == '<':
                    filtered_df = filtered_df[filtered_df[col] < value]
                elif operator == '>':
                    filtered_df = filtered_df[filtered_df[col] > value]
                elif operator == '=':
                    filtered_df = filtered_df[filtered_df[col] == value]
        
        return filtered_df
    
    def _find_name_column(self, columns: List[str]) -> Optional[str]:
        """Find the most likely name/title column."""
        name_keywords = ['name', 'title', 'film', 'movie', 'court', 'company', 'item']
        for col in columns:
            if any(keyword in col.lower() for keyword in name_keywords):
                return col
        return columns[0] if columns else None
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find the most likely date column."""
        date_keywords = ['date', 'year', 'time', 'created', 'published']
        for col in columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                return col
        return None
    
    def _generate_visualization(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate visualization based on analysis type."""
        try:
            plt.figure(figsize=(10, 6))
            
            if analysis['visualization'] == 'scatter' and len(analysis['columns']) >= 2:
                x_col, y_col = analysis['columns'][:2]
                plt.scatter(df[x_col], df[y_col], alpha=0.6)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Scatter Plot: {x_col} vs {y_col}')
            
            elif analysis['visualization'] == 'histogram' and analysis['columns']:
                col = analysis['columns'][0]
                plt.hist(df[col].dropna(), bins=20, alpha=0.7)
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Histogram: {col}')
            
            elif analysis['visualization'] == 'bar' and analysis['columns']:
                col = analysis['columns'][0]
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.title(f'Top 10 {col}')
                plt.xticks(rotation=45)
            
            else:
                # Default: plot first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    plt.hist(df[numeric_cols[0]].dropna(), bins=20, alpha=0.7)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution: {numeric_cols[0]}')
                else:
                    # No numeric columns, create a simple count plot
                    plt.text(0.5, 0.5, f'Data loaded: {len(df)} rows', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Data Summary')
            
            # Save to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return self._get_placeholder_image()
    
    def _get_placeholder_image(self) -> str:
        """Return a placeholder image."""
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def _format_results(self, results: List[Any]) -> List[str]:
        """Ensure results are in the correct format."""
        # Ensure we have exactly 4 results
        while len(results) < 4:
            if len(results) == 0:
                results.append(0)
            elif len(results) == 1:
                results.append("No data found")
            elif len(results) == 2:
                results.append(0.0)
            elif len(results) == 3:
                results.append(self._get_placeholder_image())
        
        # Ensure correct types
        formatted = []
        formatted.append(int(results[0]) if isinstance(results[0], (int, float)) else 0)
        formatted.append(str(results[1]))
        formatted.append(float(results[2]) if isinstance(results[2], (int, float)) else 0.0)
        formatted.append(str(results[3]))
        
        return formatted[:4] 