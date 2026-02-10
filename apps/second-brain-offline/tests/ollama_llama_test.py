# import requests
# import json

# def test_ollama_directly():
#     """Test Ollama API directly"""
    
#     # Test if Ollama is running
#     try:
#         response = requests.get('http://localhost:11434/api/tags')
#         print(f"Ollama status: {response.status_code}")
#         print(f"Available models: {response.json()}")
#     except Exception as e:
#         print(f"Cannot connect to Ollama: {e}")
#         return

#     # Test a simple completion
#     payload = {
#         "model": "qwen3:4b",
#         "prompt": "Summarize this in one sentence: Machine learning quantization reduces model size by using lower precision numbers.",
#         "stream": False
#     }
    
#     try:
#         response = requests.post(
#             'http://localhost:11434/api/generate',
#             json=payload,
#             timeout=30
#         )
        
#         if response.status_code == 200:
#             result = response.json()
#             print("✓ Ollama direct test successful")
#             print(f"Response: {result.get('response', 'No response')}")
#         else:
#             print(f"❌ Ollama API error: {response.status_code} - {response.text}")
            
#     except Exception as e:
#         print(f"❌ Ollama API call failed: {e}")

# test_ollama_directly()


import asyncio
from litellm import acompletion

MODEL_ID = "ollama/qwen3:4b"


async def run_check():
    response = await acompletion(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": "Explain what a retrieval augmented generation system is in one paragraph."
            }
        ],
        temperature=0.2,
        max_tokens=200,
        api_base="http://localhost:11434",
    )

    print("\n--- Ollama Response ---\n")
    print(response["choices"][0]["message"]["content"])



# if __name__ == "__main__":
#     asyncio.run(run_check())


from litellm import completion

response = completion(
    model="ollama/llama3.1:8b",
    messages=[
        {"role": "user", "content": "Explain what a retrieval augmented generation system is in one paragraph."}
    ],
    temperature=0.2,
    max_tokens=200,
    api_base="http://localhost:11434",
)

print(f"repsonse : {response}")
print(response["choices"][0]["message"]["content"])
