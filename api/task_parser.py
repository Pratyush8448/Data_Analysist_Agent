import re
from typing import Dict, Any, List, Callable

print("🚀 Running task_parser.py")

# === Entry Point ===
def parse_task(task: str) -> Dict[str, Any]:
    result = {
        "data_source": detect_data_source(task),
        "questions": []
    }
    question_blocks = extract_question_blocks(task)
    for question in question_blocks:
        parsed_question = parse_individual_question(question)
        if parsed_question:
            result["questions"].append(parsed_question)
    return result

# === Data Source Detection ===
def detect_data_source(task: str) -> Dict[str, str]:
    source = {}
    md_url_match = re.search(r'\[.*?\]\((https?://[^\s)]+)\)', task)
    if md_url_match:
        source["type"] = "web"
        source["path"] = md_url_match.group(1)
    else:
        url_match = re.search(r'(https?://[^\s]+)', task)
        if url_match:
            source["type"] = "web"
            source["path"] = url_match.group(1)
        elif ".csv" in task:
            m = re.search(r'(\S+\.csv)', task)
            source["type"] = "csv"
            source["path"] = m.group(1) if m else ""
        elif ".pdf" in task:
            m = re.search(r'(\S+\.pdf)', task)
            source["type"] = "pdf"
            source["path"] = m.group(1) if m else ""
        elif ".parquet" in task:
            m = re.search(r'(\S+\.parquet)', task)
            source["type"] = "parquet"
            source["path"] = m.group(1) if m else ""
        elif "duckdb" in task.lower():
            source["type"] = "duckdb"
            source["path"] = "duckdb"
        else:
            source["type"] = "unknown"
            source["path"] = ""
    return source

# === Question Parsing Dispatcher ===
PARSER_MAP: List[tuple[str, Callable[[str], Dict[str, Any]]]] = [
    ("how many", lambda q: parse_count_question(q)),
    ("earliest", lambda q: parse_min_question(q)),
    ("oldest", lambda q: parse_min_question(q)),
    ("correlation", lambda q: parse_correlation_question(q)),
    ("scatterplot", lambda q: parse_scatterplot_question(q)),
    ("plot", lambda q: parse_scatterplot_question(q) if "vs" in q else {"type": "unknown", "raw": q}),
    ("average", lambda q: parse_stat_summary_question(q)),
    ("mean", lambda q: parse_stat_summary_question(q)),
]

def parse_individual_question(q: str) -> Dict[str, Any]:
    q_lower = q.lower()
    for keyword, parser in PARSER_MAP:
        if keyword in q_lower:
            parsed = parser(q)
            parsed["raw_text"] = q
            return parsed
    return {"type": "unknown", "raw_text": q}

# === Question Type Parsers ===
def parse_count_question(q: str) -> Dict[str, Any]:
    condition = []
    money_match = re.search(r"\$?([\d\.]+)\s?b[n]?", q)
    year_match = re.search(r"before\s+(\d{4})", q)
    if money_match:
        amount = float(money_match.group(1)) * 1e9
        condition.append(f"gross > {int(amount)}")
    if year_match:
        condition.append(f"year < {int(year_match.group(1))}")
    return {
        "type": "count",
        "condition": " and ".join(condition) if condition else ""
    }

def parse_min_question(q: str) -> Dict[str, Any]:
    amount_match = re.search(r"\$?([\d\.]+)\s?b[n]?", q)
    amount = float(amount_match.group(1)) * 1e9 if amount_match else None
    condition = f"gross > {int(amount)}" if amount else ""
    return {
        "type": "min",
        "column": "year",
        "condition": condition
    }

def parse_correlation_question(q: str) -> Dict[str, Any]:
    x_match = re.search(r"between\s+the\s+(\w+)\s+and\s+(\w+)", q)
    if x_match:
        x, y = x_match.groups()
        return {"type": "correlation", "x": x, "y": y}
    return {"type": "correlation", "x": "", "y": ""}

def parse_scatterplot_question(q: str) -> Dict[str, Any]:
    cols = re.findall(r"\b(\w+)\b", q)
    try:
        i = cols.index("of")
        x, y = cols[i + 1], cols[i + 3]
    except:
        x, y = "", ""
    regression = "regression" in q or "line" in q
    return {
        "type": "scatterplot",
        "x": x,
        "y": y,
        "regression": regression
    }

def parse_stat_summary_question(q: str) -> Dict[str, Any]:
    match = re.search(r"(?:average|mean)\s+(\w+)", q)
    if match:
        return {"type": "mean", "column": match.group(1)}
    return {"type": "mean", "column": ""}

# === Utility ===
def extract_question_blocks(task: str) -> List[str]:
    blocks = []
    buffer = []
    lines = task.splitlines()
    for line in lines:
        num_match = re.match(r"\d+\.\s+(.*)", line.strip())
        if num_match:
            if buffer:
                blocks.append(" ".join(buffer).strip())
                buffer = []
            buffer.append(num_match.group(1))
        else:
            if buffer and line.strip():
                buffer.append(line.strip())
    if buffer:
        blocks.append(" ".join(buffer).strip())
    return [b for b in blocks if b]

# === Debug Example ===
if __name__ == "__main__":
    import json
    test_question = """
    Scrape from: [Box Office](https://en.wikipedia.org/wiki/List_of_highest-grossing_Indian_films)
    1. How many $2 bn movies were released before 2020?
    2. Which is the earliest film that grossed over $1.5 bn?
    3. What's the correlation between the Rank and Peak?
    4. Draw a scatterplot of Rank and Peak along with a red regression line.
    5. What is the average budget?
    """
    print(json.dumps(parse_task(test_question), indent=2))
