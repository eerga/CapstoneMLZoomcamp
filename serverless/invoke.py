import boto3
import json

lambda_client = boto3.client('lambda')

url = {
  "url": "https://raw.githubusercontent.com/eerga/CapstoneMLZoomcamp/main/readme_images/test_burger.jpg"
}

response = lambda_client.invoke(
    FunctionName='food-classification-lambda',
    InvocationType='RequestResponse',
    Payload=json.dumps(url)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))