import boto3
import json

lambda_client = boto3.client('lambda')

url = {
  "url": "https://github.com/eerga/CapstoneMLZoomcamp/blob/main/readme_images/test_burger.jpg"
}

response = lambda_client.invoke(
    FunctionName='food-classification-docker',
    InvocationType='RequestResponse',
    Payload=json.dumps(url)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))