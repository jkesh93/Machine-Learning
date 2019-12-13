# Understanding ClassifyingCatsDogsCNN #
My Code Reflections ('CodeReflex') is where I go back and review the code in a way that helps myself and others to understand it at a deeper level than when I first went through it.

Although I try my best to comment my code neatly, it is often difficult for someone who may think of a problem / solution differently to understand, so I have created this markdown to explain my thoughts a little more clearly.

# Source #
**All credit** goes to **sentdex** for his incredible tutorials. I encourage anyone who reads my code to watch his videos.

Video Credit
["sentdex"](https://www.youtube.com/user/sentdex "sentdex YouTube Channel")

[2nd Video in the series](https://www.youtube.com/watch?v=j-3vuBynnOE "Convolutional Neural Networks...")

[3rd Video in the series](https://www.youtube.com/watch?v=WvoLTXIjBYU "Convolutional Neural Networks...")

# CodeReflex #

## Early portion of code consisted mostly of directory setup and library imports ##

### Library Imports ###
```python
    # Numpy for array operations
    import numpy as np

    # Using matplotlib to show the image
    import matplotlib.pyplot as plt

    # Iterate through directories / join paths
    import os

    # Image operations
    import cv2
```

### Directory setup ###
Specifying my base directory and then the categories we'll be using. The directory had struggle similar to
        
    PetImages -> Dog / Images ** This is my dog training images
                 Cat / Images ** This is my cat training images

```python
DATADIR = "C:/Users/Jay/Desktop/Machine_Learning/kagglecatsanddogs_3367a/PetImages"

### Specify categories
CATEGORIES = ["Dog","Cat"]
```

## Verify the data loads ##

### But before building the training data, its best to check that images can be loaded ###

```python
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
```
In the above, I have two "break" statements to prevent going through the entire set - I just want to make sure the image loads.

Now that I know the data loads, I can build the training data.

## Build the training data ##
Building the training data will consist of setting an image size - that is what width x height is. Data going into a machine learning model is easier done when its a standard size. Also, this reduces data load. To determine a size, I sampled the images at various sizes. As I can tell a dog is a dog even at 100 x 100 pixels, I've chosen to go with 100 pixels for the size.

```python
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
```

## Randomization and formatting

### Randomization ###
Since Machine Learning looks for a way to minmize 'loss' a non-fancy term for 'incorrect answer count', if the data follows the order of **all** cats and then **all** dogs, the machine may opt to just switch to figuring out 'when' that switch occurs. So we need to randomize the presentation of the training data.

### Formatting ###
We create the variables **X** and **y**. This is standard practice (the uppercase X for **features** and lowercase **y** for 'label').

**Features** These are our inputs or what is *known*

**Label** This is what we want to find out or predict for. Also known as ***classification***.

**X/y** In math, f(x) = y is to say that if we put 'x' into a function 'f' then this equals (=) 'y'. In the same way that if we have an image of a dog (X) then we want to put it through our model (f) and get a classification (y).

```python
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
```

### Reshaping ###
We need to reshape the data as when we **appended** the features and labels it put them into a **list** which is incompatible. The **shape** of the data is the **dimensionality** of it. **-1** means **any number**.

In the above X is of shape (24946, 100, 100, 1). This is because there are:

24946 pictures which are
100 pixels in length by
100 pixels in wide and
1 color value (light intensity) in depth

## Building the model ##

```python
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
```

#### The convolution layer ####
While the code is mostly self-explanatory, I think one of the things worth point out from the above is how the keras convolution layer works. It's found in this portion of the code
```python
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
```
What we see in the above is that we're asking for the following

64 is the output shape
3,3 is the 'window'
X.shape[1:] is the input shape

**Output shape**
When we talk about shape, we're talking about dimensionality and so the output shape is going to be a *feature map* with a dimension of 64.

**Window**
3,3 is the window size. In a convolutional neural network, there is some subsampling going on and as such this 'area' / 'window' whatever we want to call it is going to be defined as our (input dimensionality x some height). As our incoming data shape is a vector of 3 (x,x,x) and so we're saying we are considering 3 of those stacked on top of each other.

#### The MaxPooling2D layer ####
The MaxPooling2D layer takes another window and condenses it. That is, a matrix operation is performed and the large feature map from before is reduced to a smaller one and reducing the complexity of the overall operation.

#### The Flatten Layer ####
The Flatten layer takes the existing higher dimensionality data and squashes it back into a 1D layer so that it can be fed into the Dense layer.

#### Binary-Crossentropy Optimization ####
*Binary-Crossentropy Optimization* is often used for dealing with binary categorical data.
