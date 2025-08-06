from api.scraper import scrape_data
from api.parquet_handler import handle_parquet_query, analyze_indian_court_data
from api.plotter import generate_plot
from api.utils import detect_task_type
from api.task_parser import parse_task
from api.analyzer import analyze_task
from api.analyzer_simple import analyze_task_simple
from api.universal_analyzer import UniversalDataAnalyzer

from typing import Dict, Any, List
import traceback
import pandas as pd
import logging
import uuid
import asyncio

logging.basicConfig(level=logging.INFO)

def preview_table(df: pd.DataFrame) -> str:
    try:
        return df.head(5).to_markdown(index=False, tablefmt="github")
    except Exception as e:
        logging.warning(f"[preview_table] Error creating markdown preview: {e}")
        return ""

async def handle_request(question: str) -> List[str]:
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Incoming question: {question}")

    try:
        # Check if this is an Indian court dataset task
        if "indian high court" in question.lower() or "ecourts" in question.lower() or "judgments" in question.lower():
            try:
                answers = analyze_indian_court_data(question)
                logging.info(f"[{request_id}] Indian court analysis completed: {answers}")
                return answers
            except Exception as e:
                logging.warning(f"[{request_id}] Indian court analysis failed: {e}")
                return [0, f"Indian court analysis failed: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
        
        # Check if this is a web scraping task (most likely for our use case)
        elif "wikipedia" in question.lower() or "http" in question.lower():
            try:
                # Use the simple analyzer for now to avoid async issues
                answers = analyze_task_simple(question)
                logging.info(f"[{request_id}] Analysis completed: {answers}")
                return answers
            except Exception as e:
                logging.warning(f"[{request_id}] Analysis failed: {e}")
                return [0, f"Analysis failed: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

        # For other task types, try the general approach
        parsed = parse_task(question)
        task_type = detect_task_type(question)

        df = None
        plots = []
        summary = ""
        query_code = ""
        additional_notes = ""

        # 🌐 Web Scraping Task
        if parsed.get("data_source", {}).get("type") == "web":
            try:
                answers = await analyze_task(question)
                return answers
            except Exception as e:
                logging.warning(f"[{request_id}] Analysis failed: {e}")
                return [0, f"Analysis failed: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

        # 🦆 Parquet/DuckDB Query Task
        elif parsed.get("data_source", {}).get("type") == "duckdb":
            tables, summary, preview = await asyncio.to_thread(handle_parquet_query, question)
            return [0, "DuckDB query not implemented", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

        # 📊 Plot-Only Task
        elif any("plot" in q.get("type", "") for q in parsed.get("questions", [])):
            return [0, "Plot-only tasks not implemented", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

        # 🤷 Unknown - try universal analyzer as fallback
        else:
            try:
                universal_analyzer = UniversalDataAnalyzer()
                answers = universal_analyzer.analyze_question(question)
                logging.info(f"[{request_id}] Universal analysis completed: {answers}")
                return answers
            except Exception as e:
                logging.warning(f"[{request_id}] Universal analysis failed: {e}")
                # Final fallback to web scraping
                try:
                    answers = await analyze_task(question)
                    return answers
                except Exception as e2:
                    logging.warning(f"[{request_id}] All analysis methods failed: {e2}")
                    return [0, f"Analysis failed: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]

    except Exception as e:
        logging.exception(f"[{request_id}] Unhandled exception in handle_request")
        return [0, f"Critical error: {str(e)}", 0.0, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="]
