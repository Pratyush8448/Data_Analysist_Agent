import base64
import traceback
from io import BytesIO
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import duckdb


# -------------------- Plot Utilities --------------------

def fig_to_base64(fig: plt.Figure, max_size_bytes: int = 100000) -> str:
    """
    Convert a Matplotlib figure to base64 string (PNG format).
    Dynamically reduce DPI to stay under max_size_bytes.
    """
    dpi = 100
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    while buf.tell() > max_size_bytes and dpi > 40:
        dpi -= 10
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -------------------- Data Loading Utilities --------------------

def safe_load_parquet(path_or_url: str) -> pd.DataFrame:
    """
    Loads Parquet from local/URL using DuckDB first, then Pandas fallback.
    """
    try:
        return duckdb.query(f"SELECT * FROM '{path_or_url}'").to_df()
    except Exception as duckdb_err:
        try:
            return pd.read_parquet(path_or_url)
        except Exception as pandas_err:
            raise RuntimeError(
                f"❌ Parquet Load Failed:\n"
                f"• DuckDB Error: {duckdb_err}\n"
                f"• Pandas Error: {pandas_err}"
            )


# -------------------- Task Detection Utilities --------------------

TASK_KEYWORDS = {
    "scrape": ["scrape", "website", "wikipedia", "extract", "web"],
    "query": ["query", "sql", "duckdb", "select", "filter"],
    "load": ["parquet", "load", "read", "open"],
    "plot": ["plot", "chart", "scatter", "bar", "line", "distribution", "regression"],
    "analyze": ["analyze", "correlation", "insight", "pattern", "relation"]
}

def validate_task_format(task_text: str) -> bool:
    """
    Return True if task_text contains at least one known keyword.
    """
    task_text = task_text.lower()
    return any(kw in task_text for kws in TASK_KEYWORDS.values() for kw in kws)


def detect_task_type(text: str) -> list[str]:
    """
    Detect all task types based on keyword matching.
    """
    text = text.lower()
    return [task for task, keywords in TASK_KEYWORDS.items() if any(kw in text for kw in keywords)] or ["unknown"]


# -------------------- Response Formatting Utilities --------------------

def format_response(success: bool, message: str, data: dict = None, error: str = None) -> dict:
    """
    Standard JSON API response structure.
    """
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "success": success,
        "message": message,
        "data": data or {},
        "error": error
    }


def format_exception_response(e: Exception, context: str = "") -> dict:
    """
    Create a structured error response from an exception.
    """
    return format_response(
        success=False,
        message=f"Exception in {context}: {str(e)}",
        error=traceback.format_exc()
    )
