ECR_URL=542892327487.dkr.ecr.us-east-1.amazonaws.com
REPO_URL=${ECR_URL}/food-classification-lambda
LOCAL_IMAGE=food-classifier

aws ecr get-login-password \
  --region "us-east-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

REMOTE_IMAGE_TAG="${REPO_URL}:v1"

docker build -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done"