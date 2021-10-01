from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

model_path = r"C:\Users\hrnay\Documents\project\2\model.h5"
image_size = 80
model = tf.keras.models.load_model(model_path)
print ("Model loaded successfully")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction', methods=["POST"])
def prediction():

    img = request.files['img']

    img.save("img.jpg")

    image = cv2.imread("img.jpg")

    image = cv2.resize(image, (image_size,image_size)) / 255

    image = np.reshape(image, (1,image_size,image_size,3))

    pred = model.predict(image)
    pred = np.argmax(pred)

    if pred == 1:
        return render_template("index.html", data = "It is a DOG")

    if pred == 0:
        return render_template("index.html", data = "It is a CAT")


if __name__ == "__main__":
    app.run(debug=True)
