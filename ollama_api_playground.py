import json
import requests

systemPrompt = "Write a concise summary of the text, return your responses with 5 lines that cover the key points of the text given."
prompt = "hi"

url = "http://localhost:11434/api/generate"

payload = {
    "model": "mistral-openorca",
    "prompt": prompt,
    "system": systemPrompt,
    "stream": False
}

payload_json = json.dumps(payload)
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=payload_json, headers=headers)

print(json.loads(response.text)["response"])
