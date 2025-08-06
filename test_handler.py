# test_handler.py
import asyncio
from api.handler import handle_request

question = "Scrape the list of highest-grossing Indian films from this URL: https://en.wikipedia.org/wiki/List_of_highest-grossing_Indian_films"

if __name__ == "__main__":
    result = asyncio.run(handle_request(question))
    print("\n🔍 Summary:\n", result["summary"])
    print("\n📊 Query Code:\n", result["query_code"])
    print("\n📈 Plots:\n", result["plots"])
    print("\n🧾 Data Preview:\n", result["data_preview"])
    print("\n📝 Notes:\n", result["additional_notes"])
