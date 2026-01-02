import boto3
import json

lambda_client = boto3.client('lambda')

url = {
  "url": "https://imgur.com/a/Q1KiHkT"
}

response = lambda_client.invoke(
    FunctionName='food-classification-docker',
    InvocationType='RequestResponse',
    Payload=json.dumps(url)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))