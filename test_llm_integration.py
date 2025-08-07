import os
import asyncio
from dotenv import load_dotenv
from api.llm_client import llm_client
from api.handler import get_data_context, parse_llm_response

# Load environment variables
load_dotenv()

async def test_llm_integration():
    """Test LLM integration with sample questions"""
    
    print("🧪 Testing LLM Integration")
    print("=" * 50)
    
    # Test 1: Check if LLM is available
    print("\n1. Checking LLM availability...")
    if llm_client.is_available():
        print("✅ LLM is available and configured")
        print(f"   Model: {llm_client.model_name}")
        print(f"   Base URL: {llm_client.base_url}")
    else:
        print("❌ LLM is not available")
        print("   Make sure AI_PROXY_API_KEY and AI_PROXY_BASE_URL are set")
        return
    
    # Test 2: Simple question
    print("\n2. Testing simple question...")
    try:
        response = await llm_client.analyze_data("What is 2+2?")
        print(f"✅ LLM Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Data analysis question
    print("\n3. Testing data analysis question...")
    question = "How many movies grossed over $2 billion?"
    try:
        data_context = await get_data_context(question)
        response = await llm_client.analyze_data(question, data_context)
        print(f"✅ LLM Response: {response}")
        
        # Test parsing
        parsed = parse_llm_response(response, question)
        print(f"✅ Parsed Results: {parsed}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Complex question
    print("\n4. Testing complex question...")
    complex_question = """
    Scrape the list of highest grossing films from Wikipedia. It is at the URL:
    https://en.wikipedia.org/wiki/List_of_highest-grossing_films

    Answer the following questions:
    1. How many $2 bn movies were released before 2000?
    2. Which is the earliest film that grossed over $1.5 bn?
    3. What's the correlation between the Rank and Peak?
    """
    
    try:
        data_context = await get_data_context(complex_question)
        response = await llm_client.analyze_data(complex_question, data_context)
        print(f"✅ LLM Response: {response}")
        
        # Test parsing
        parsed = parse_llm_response(response, complex_question)
        print(f"✅ Parsed Results: {parsed}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_integration()) 