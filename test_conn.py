import requests

try:
    response = requests.get("http://127.0.0.1:1234/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Models: {response.json()}")
except Exception as e:
    print(f"Connection Failed: {e}")
