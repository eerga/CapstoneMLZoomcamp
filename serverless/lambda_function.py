import os
import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor


model_name = os.getenv("MODEL_NAME", "food_classifier_efficientnet_v6.onnx")


def preprocess_keras_style(X):
    """
    Not an Exact EfficientNet preprocessing from Keras source
    """
    X = X.astype(np.float32)
    return X


preprocessor = create_preprocessor(
    preprocess_keras_style,
    target_size=(299, 299)
)

session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = ['chicken_curry',
 'chocolate_cake',
 'fish_and_chips',
 'hamburger',
 'ice_cream',
 'pad_thai',
 'pizza',
 'ramen',
 'sushi',
 'tacos']


def predict(url):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    predictions = dict(zip(classes, float_predictions))
    sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    return sorted_predictions


def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result