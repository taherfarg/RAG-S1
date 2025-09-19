import requests
import json

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-proj-1234567890"
}
data = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.5
}
response = requests.post(url, headers=headers, json=data)
print(response.json())


