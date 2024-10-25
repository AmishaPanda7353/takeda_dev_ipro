import requests

url = "https://openai-mcd.openai.azure.com//openai/deployments/mcd-insights-pro-4o/chat/completions?api-version=2023-03-15-preview"
headers = {
    "api-key": "1fc6ffcf95594676934236dfb69a2a81",
    "Content-Type": "application/json"
}
data = {
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 100
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 404:
    print("Resource not found. Check deployment name and API endpoint.")
else:
    print(response.json())
