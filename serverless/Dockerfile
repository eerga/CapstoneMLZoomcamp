FROM public.ecr.aws/lambda/python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# Copy pyproject.toml first (for dependency installation)
COPY pyproject.toml ./

ARG MODEL_NAME=food_classifier_efficientnet_v6.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY ${MODEL_NAME} ${MODEL_NAME}
RUN uv pip install --system -r <(uv export --format requirements-txt)
#RUN uv pip install --system --no-deps -e .
COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]