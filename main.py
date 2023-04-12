import io
import numpy as np
from PIL import Image
import keras
from flask import Flask, jsonify, request
import tensorflow as tf
import os
import pathlib
# Import Libraries
import warnings
warnings.filterwarnings("ignore")
import os
import glob
# Keras API
import keras
# Keras Model Imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pickle


model = tf.keras.models.load_model('model/emotion.h5')
print("Emotion Model Loaded Successfully!!!")



# Loading model and predict.
Classes = ["கோவம்","வெறுப்பு","பயம்","சந்தோஷம்","நடுநிலை","வருத்தம்","ஆச்சரியம்"]
def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((256, 256))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


img_width = 256
img_height = 256
"""def prepare(img):
    img_sec = load_img(img, target_size=(256, 256))
    x = img_to_array(img_sec)
    x = x / 255
    return np.expand_dims(x, axis=0)"""


def predict_result(img):
    result = model.predict(img)
    classes_x = np.argmax(result, axis=1)
    # result = np.argmax(model.predict([prepare('drive/My Drive/Emotions/surprise/35863.jpg')]),axis=1)
    """emotion = load_img(img)
    plt.imshow(emotion)
    print(emotion)"""
    print(Classes[int(classes_x)])
    return (Classes[int(classes_x)])

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        msg = "Please try again. The Image doesn't exist"
        return jsonify(warning_msg = msg)

    file = request.files.get('file')

    if not file:
        return

    """img_bytes = file.read()
    img = prepare(img_path=)"""
#    img = prepare_image()

    img_bytes = file.read()
    img = prepare_image(img_bytes)
    predicted_emotion = predict_result(img)
   # generated_poem =  predict_poem(predicted_emotion)
    return jsonify(emotion=predicted_emotion)

@app.route('/health', methods=['GET'])
def index():
    return 'SUCCESS:Health Check Done!'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='9090')