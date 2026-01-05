import requests
import json

def test_ollama_directly():
    """Test Ollama API directly"""
    
    # Test if Ollama is running
    try:
        response = requests.get('http://localhost:11434/api/tags')
        print(f"Ollama status: {response.status_code}")
        print(f"Available models: {response.json()}")
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return

    # Test a simple completion
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": "Summarize this in one sentence: Machine learning quantization reduces model size by using lower precision numbers.",
        "stream": False
    }
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Ollama direct test successful")
            print(f"Response: {result.get('response', 'No response')}")
        else:
            print(f"❌ Ollama API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Ollama API call failed: {e}")

test_ollama_directly()