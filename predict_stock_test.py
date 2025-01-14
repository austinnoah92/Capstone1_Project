import requests
url = 'http://localhost:9990/predict'
data = {
    'Price_Change': -0.585, 
    'MA5': 85.23 * 500, 
    'MA20': 16.500, 
    'Volume_Change': -5000
    }
response = requests.post(url, json = data).json()
print(response)