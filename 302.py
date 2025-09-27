import requests

API_KEY = "sk-sOGi7c1ilx6KZYuY5815XpWioGoPX1CK6g2wYMJEbP75JFjH"
API_URL = "https://api.302.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-5-nano-2025-08-07",   # you can pick another model available on 302.ai
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a fun fact about airplanes."}
    ]
}

response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    # the text is usually inside choices[0].message.content
    print("Assistant:", result["choices"][0]["message"]["content"])
else:
    print("Error:", response.status_code, response.text)