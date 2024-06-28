import requests
import json

json_data = {
    "urls": [
        "https://example.com/presigned-url1",
        "https://example.com/presigned-url2",
        "https://example.com/presigned-url3"
    ],
    "memberid": "user123",
    "eyeclosing": True,
    "blurred": True
}

url = 'http://127.0.0.1:8000/process-urls'
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, json=json_data)

print(response.status_code)
print(response.json())
