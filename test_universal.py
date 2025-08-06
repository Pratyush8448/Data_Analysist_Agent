import requests
import json

def test_universal_analyzer():
    """Test the universal analyzer with various question types"""
    
    test_cases = [
        {
            "name": "CSV Data Analysis",
            "question": "Analyze this CSV data from https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv and tell me how many countries have GDP over $1 trillion. Also plot the GDP vs year relationship.",
            "expected": "Should handle CSV URL and perform analysis"
        },
        {
            "name": "JSON Data Analysis", 
            "question": "Load this JSON data from https://jsonplaceholder.typicode.com/users and count how many users have email addresses ending in .com. Show me a bar chart of user IDs.",
            "expected": "Should handle JSON API and perform analysis"
        },
        {
            "name": "Generic Website Table",
            "question": "Go to https://en.wikipedia.org/wiki/List_of_countries_by_population and tell me how many countries have population over 100 million. Which country has the highest population?",
            "expected": "Should scrape any website table and analyze"
        },
        {
            "name": "Excel Data Analysis",
            "question": "I have an Excel file with sales data. Can you analyze it and tell me the average sales amount and create a histogram of sales distribution?",
            "expected": "Should handle Excel files (if provided)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print(f"Question: {test_case['question'][:100]}...")
        print(f"Expected: {test_case['expected']}")
        
        # Create a temporary file-like object
        files = {'questions_txt': ('questions.txt', test_case['question'], 'text/plain')}
        
        try:
            # Make the request
            response = requests.post('http://localhost:8000/api/', files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Test successful!")
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
                        print(f"✅ Name: {result[1]} (string)")
                    else:
                        print(f"⚠️ Name: {result[1]} (should be string)")
                        
                    if isinstance(result[2], (int, float)):
                        print(f"✅ Value: {result[2]} (numeric)")
                    else:
                        print(f"⚠️ Value: {result[2]} (should be numeric)")
                        
                    if isinstance(result[3], str) and result[3].startswith("data:image/png;base64,"):
                        print(f"✅ Plot: {result[3][:50]}... (base64 image)")
                    else:
                        print(f"⚠️ Plot: {result[3]} (should be base64 image)")
                else:
                    print(f"❌ Response format incorrect: {type(result)}, length: {len(result) if isinstance(result, list) else 'N/A'}")
            else:
                print(f"❌ Test failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error in test: {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_universal_analyzer() 