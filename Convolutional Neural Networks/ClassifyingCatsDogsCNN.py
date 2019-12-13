# Numpy for array operations
import numpy as np

# Using matplotlib to show the image
import matplotlib.pyplot as plt

# Iterate through directories / join paths
import os

# Image operations
import cv2

# Specify a data directory
DATADIR = "C:/Users/USER/Desktop/Machine_Learning/catsDogs/PetImages"

# Specify categories
CATEGORIES = ["Dog","Cat"]

# Show an example to see how our images load
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break

# Build the training Data
IMG_SIZE = 100
training_data = []

# Get training data
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        
        # Create classification as a numerical value. In this case it will be the index of the category.
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            # Try to do it, if it fails, skip
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                
                # Resize images to lessen data load
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# Randomize data to make sure it learns
import random
random.shuffle(training_data)

# Capital X are features, lowercase y is your label
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# Getting correct shape
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# Get your libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Before feeding data through a neural network, normalize the data. Easiest way to normalize the data is to SCALE the data
# In imagery data, the max will be 255, so we can just divide by 255
X = X/255.0

# Build the model
model = Sequential()

# 64 'windows' or 'filters'. The input shape is X.shape (24946, 100, 100, 1) - 24946 is the number of items in the feature set.
# The shape of each feature (i.e. image) is (100, 100, 1).
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Need to flatten the layer since dense expects a 1D shape
model.add(Flatten())
model.add(Dense(64))

# Add the output layer - a 1 node output
model.add(Dense(1, activation = "sigmoid"))

# Since we're categorizing between 2 separate things, we'll use binary-crossentropy. Adam is a good optimizer. We'll use "accuracy" as a metrics base.
model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

# Feed it images in batches of 32, validation split is 30% (out of sample data)
model.fit(X, y, batch_size=32, validation_split=0.1, epochs=20)

# Use the trained model
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

img = prepare('cat.jpg')
img = img / 255.0
prediction = model.predict(img)

print(prediction)
