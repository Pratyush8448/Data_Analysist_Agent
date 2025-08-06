import requests
import json

def test_indian_courts_api():
    """Test the API with the Indian High Court dataset question"""
    
    # Read the question from the test file
    with open('test_indian_courts.txt', 'r', encoding='utf-8') as f:
        question = f.read()
    
    # Create a temporary file-like object
    files = {'questions_txt': ('questions.txt', question, 'text/plain')}
    
    try:
        # Make the request
        response = requests.post('http://localhost:8000/api/', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Indian Courts API test successful!")
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
                    print(f"✅ Court: {result[1]} (string)")
                else:
                    print(f"⚠️ Court: {result[1]} (should be string)")
                    
                if isinstance(result[2], (int, float)):
                    print(f"✅ Regression slope: {result[2]} (numeric)")
                else:
                    print(f"⚠️ Regression slope: {result[2]} (should be numeric)")
                    
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
        print(f"❌ Error testing Indian Courts API: {e}")

if __name__ == "__main__":
    test_indian_courts_api() 