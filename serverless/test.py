import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

request = {
    "url": "https://raw.githubusercontent.com/eerga/CapstoneMLZoomcamp/main/readme_images/test_burger.jpg"
}

result = requests.post(url, json=request).json()
print(result)