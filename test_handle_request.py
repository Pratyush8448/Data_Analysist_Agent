import asyncio
from api.handler import handle_request

if __name__ == "__main__":
    with open("test_questions/question.txt", "r", encoding="utf-8") as f:
        question = f.read().strip()

    result = asyncio.run(handle_request(question))

    import json
    print(json.dumps(result, indent=2))
