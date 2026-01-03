ECR_URL=[account_number].dkr.ecr.us-east-1.amazonaws.com
REPO_URL=${ECR_URL}//[name_of_docker_container]-lambda
LOCAL_IMAGE=[name_of_docker_container]



aws ecr get-login-password \
  --region "[your_geographic_aws_zone]" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

REMOTE_IMAGE_TAG="${REPO_URL}:v1"

docker build -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done"