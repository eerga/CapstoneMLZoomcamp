FROM public.ecr.aws/lambda/python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Copy pyproject.toml first
COPY pyproject.toml ./

ARG MODEL_NAME=food_classifier_efficientnet_v6.onnx
ENV MODEL_NAME=${MODEL_NAME}

# Download from Hugging Face (replace with your username)
ARG HF_REPO=erikyshkin/food-classification-model  
RUN curl -L -o ${MODEL_NAME} \
    "https://huggingface.co/${HF_REPO}/resolve/main/${MODEL_NAME}"

RUN uv pip install --system -r <(uv export --format requirements-txt)
COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]