import requests
import json

def test_api():
    """Test the API with the sample question"""
    
    # Sample question from the test file
    question = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""

    # Create a temporary file-like object
    files = {'questions_txt': ('questions.txt', question, 'text/plain')}
    
    try:
        # Make the request
        response = requests.post('http://localhost:8000/api/', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API test successful!")
            print(f"Response: {result}")
            print(f"Response type: {type(result)}")
            print(f"Response length: {len(result)}")
            
            # Validate the response format
            if isinstance(result, list) and len(result) == 4:
                print("✅ Response format is correct (4-element array)")
                
                # Check each element type
                if isinstance(result[0], (int, float)):
                    print(f"✅ Count: {result[0]} (numeric)")
                else:
                    print(f"⚠️ Count: {result[0]} (should be numeric)")
                    
                if isinstance(result[1], str):
                    print(f"✅ Movie: {result[1]} (string)")
                else:
                    print(f"⚠️ Movie: {result[1]} (should be string)")
                    
                if isinstance(result[2], (int, float)):
                    print(f"✅ Correlation: {result[2]} (numeric)")
                else:
                    print(f"⚠️ Correlation: {result[2]} (should be numeric)")
                    
                if isinstance(result[3], str) and result[3].startswith("data:image/png;base64,"):
                    print(f"✅ Plot: {result[3][:50]}... (base64 image)")
                else:
                    print(f"⚠️ Plot: {result[3]} (should be base64 image)")
            else:
                print(f"❌ Response format incorrect: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
        else:
            print(f"❌ API test failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api() 