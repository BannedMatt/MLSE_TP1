import requests

r = requests.get("http://127.0.0.1:8000/predict")
print(r.status_code)
print(r.json())