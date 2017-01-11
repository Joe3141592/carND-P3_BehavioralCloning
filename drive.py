import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
cv2.namedWindow("frame", 1)
import pickle
mode = "auto"

# This function resizes the images
def translate(image, x, y):
    #http://www.programcreek.com/python/example/87702/cv2.warpAffine
    columns, rows = 200,66
    return cv2.warpAffine(
        image,
        np.float32([[1, 0, x], [0, 1, y]]),
        (columns, rows)
    )

@sio.on('telemetry')
def telemetry(sid, data):
    global mode
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString))).resize((210,105)) #Added this
    image_array = np.array(image).astype("float32")

    image_array=translate(image_array, -5 , -24 )
    BGR_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    BGR_img = cv2.resize(BGR_img,(400,122))
    # I show the camera view of the car in a new window
    cv2.imshow("frame", BGR_img.astype("uint8"))

    pressed_button=cv2.waitKey(1)
    #transformed_image_array = RGB_img[None, :, :, :]#
    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # One can interfere in the driving by pressing a - left turn - or b - right turn. This is only used for debugging certain models.
    if (pressed_button==100):
        steering_angle = 1
        print("ch")
    if (pressed_button == 97):
        steering_angle = -1
        print("ch")

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3 # I increased the throttle to 0.3
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
