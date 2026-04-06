"""Archived Groq connectivity scratch file from an earlier prototype."""

import urllib.request
import json
import ssl
import os

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "user", "content": "Hello, this is a test call."}
    ],
    "max_tokens": 50
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')

try:
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, context=context) as response:
        print("Status Code:", response.getcode())
        resp_data = response.read().decode('utf-8')
        try:
            print("Response:", json.dumps(json.loads(resp_data), indent=2))
        except json.JSONDecodeError:
            print("Response:", resp_data)
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    try:
        err_msg = e.read().decode('utf-8')
        print("Error Response:", json.dumps(json.loads(err_msg), indent=2))
    except (json.JSONDecodeError, Exception):
        print("Error Response:", err_msg)
except Exception as e:
    print("Error:", e)
