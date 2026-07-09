from ollama import chat

MODEL = "qwen2.5:7b"

print("Testing Ollama model:", MODEL)

response = chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Return only the word OK."}
    ],
    options={
        "temperature": 0,
        "num_predict": 10,
    },
)

if isinstance(response, dict):
    print(response["message"]["content"])
else:
    print(response.message.content)
