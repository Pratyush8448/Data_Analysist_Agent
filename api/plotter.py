import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import re
import base64
from typing import Optional
from difflib import get_close_matches

sns.set(style="whitegrid")


def plot_distribution(df: pd.DataFrame, column: str) -> Optional[str]:
    col = _match_column(df, column)
    if col is None:
        raise ValueError(f"Column '{column}' not found in DataFrame (after fuzzy matching).")

    data = df[col].dropna()
    if data.empty:
        raise ValueError(f"No valid data in column '{col}' for plotting.")

    plt.figure(figsize=(8, 5))
    if pd.api.types.is_numeric_dtype(data):
        sns.histplot(data, kde=True, color='skyblue')
        plt.title(f"Distribution of '{col}'")
    else:
        data.value_counts().head(10).plot(kind='bar', color='orange')
        plt.title(f"Top 10 Categories in '{col}'")

    plt.xlabel(col)
    plt.ylabel("Frequency")

    return _encode_plot_to_base64()


def plot_scatter(df: pd.DataFrame, x: str, y: str, regression: bool = False) -> Optional[str]:
    x_col = _match_column(df, x)
    y_col = _match_column(df, y)
    if x_col is None or y_col is None:
        raise ValueError(f"Columns '{x}' and/or '{y}' not found in DataFrame (after fuzzy matching).")

    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        raise ValueError(f"Both columns '{x_col}' and '{y_col}' must be numeric for scatter plot.")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, color='purple')

    if regression:
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            scatter=False,
            line_kws={'color': 'red', 'linestyle': '--'}
        )

    plt.title(f"Scatter Plot: {x_col} vs {y_col}" + (" with regression" if regression else ""))
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    return _encode_plot_to_base64()


def generate_plot(df: pd.DataFrame, question: str) -> tuple[Optional[str], str]:
    q = question.lower()
    try:
        # Scatter plot with or without regression
        if "scatter" in q or "plot" in q:
            match = re.search(r"(?:scatter|plot).*?(?:of|between)?\s*(\w+)\s*(?:vs|and)\s*(\w+)", q)
            if match:
                x, y = match.group(1), match.group(2)
                use_regression = "regression" in q or "regression line" in q
                image_b64 = plot_scatter(df, x, y, regression=use_regression)
                summary = f"Scatter plot of '{x}' vs '{y}'" + (" with regression line." if use_regression else ".")
                return image_b64, summary

        # Distribution / histogram / bar chart
        if "distribution" in q or "histogram" in q or "bar chart" in q or "bar plot" in q:
            match = re.search(r"(?:distribution|histogram|bar chart|bar plot).*?(?:of)?\s+(\w+)", q)
            if match:
                col = match.group(1)
                image_b64 = plot_distribution(df, col)
                return image_b64, f"Generated plot for '{col}'."

        # Fallback — plot first numeric column
        fallback_col = next(
            (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])),
            None
        )
        if fallback_col is None:
            raise ValueError("No numeric columns found for plotting.")
        image_b64 = plot_distribution(df, fallback_col)
        return image_b64, f"No specific column found; plotted '{fallback_col}'."

    except Exception as e:
        raise ValueError(f"Plot generation failed: {str(e)}")


def _encode_plot_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close('all')  # Close all figures
    return f"data:image/png;base64,{image_base64}"


def _match_column(df: pd.DataFrame, col_name: str) -> Optional[str]:
    """
    Attempts to match a column name exactly or using fuzzy matching.
    """
    col_name = col_name.strip().lower()
    # Exact match
    for col in df.columns:
        if col.strip().lower() == col_name:
            return col

    # Fuzzy match
    potential_matches = [col for col in df.columns if isinstance(col, str)]
    close = get_close_matches(col_name, potential_matches, n=1, cutoff=0.6)
    return close[0] if close else None
