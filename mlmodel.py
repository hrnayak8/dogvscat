import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd

REBUILD_DATA = True

class DogsVsCats():
  IMAGE_SIZE = 80
  CATS = "/content/PetImages/Cat"
  DOGS = "/content/PetImages/Dog"
  LABELS = {CATS: 0, DOGS: 1}

  training_data = []
  catcount = 0
  dogcount = 0

  def make_traing_data(self):
    for label in self.LABELS:
      print(label)
      for f in tqdm(os.listdir(label)):
        if "jpg" in f:
          try:
            path = os.path.join(label, f)
            img = cv2.imread(path)
            img = cv2.resize(img, (self.IMAGE_SIZE,self.IMAGE_SIZE))
            self.training_data.append([np.array(img), self.LABELS[label]])

            if label == self.CATS:
              self.catcount += 1

            else:
              self.dogcount += 1
          except Exception as e:
            pass
    np.random.shuffle(self.training_data)
    np.save("training_data.npy", self.training_data)
    print("Cats:", self.catcount)
    print("Dogs:", self.dogcount)

if REBUILD_DATA:
  dogsvscats = DogsVsCats()
  dogsvscats.make_traing_data()

X = training_data[:,0]
y = training_data[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

X_train = tf.convert_to_tensor(X_train.tolist()) / 255
X_test = tf.convert_to_tensor(X_test.tolist()) / 255
y_train = tf.convert_to_tensor(y_train.tolist())
y_test = tf.convert_to_tensor(y_test.tolist())

tf.random.set_seed(42)

model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation= "relu", input_shape = (80, 80, 3)),
                             tf.keras.layers.MaxPool2D(pool_size=2, padding="same"),
                             tf.keras.layers.Conv2D(10, 3, activation="relu"),
                             tf.keras.layers.MaxPool2D(pool_size=2, padding="same"),
                             tf.keras.layers.Conv2D(10, 3, activation="relu"),
                             tf.keras.layers.MaxPool2D(pool_size=2, padding="same"),
                             tf.keras.layers.Conv2D(10, 3, activation="relu"),
                             tf.keras.layers.Conv2D(10, 3, activation="relu"),
                             tf.keras.layers.MaxPool2D(pool_size=2, padding="same"),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(2, activation = "sigmoid")
])

model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
              metrics="accuracy"
              )

model.fit(X_train,
          tf.one_hot(y_train, depth=2),
          batch_size = 32,
          epochs = 30,
          validation_data=(X_test, tf.one_hot(y_test, depth=2))
          )
