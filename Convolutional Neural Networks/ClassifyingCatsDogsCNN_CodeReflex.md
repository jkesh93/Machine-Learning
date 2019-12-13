# Understanding ClassifyingCatsDogsCNN
My Code Reflections (CodeReflex) is a breakdown of the code. Although I try my best to comment my code neatly, it is often difficult for someone who may think of a problem / solution differently to understand, so I have created this markdown to explain my thoughts a little more clearly.

# Source
**All credit** goes to **sentdex** for his incredible tutorials. I encourage anyone who reads my code to watch his videos.

Video Credit
["sendex"](https://www.youtube.com/user/sentdex "sentdex YouTube Channel")
[2nd Video in the series](https://www.youtube.com/watch?v=j-3vuBynnOE "Convolutional Neural Networks...")
[3rd Video in the series](https://www.youtube.com/watch?v=WvoLTXIjBYU "Convolutional Neural Networks...")

# CodeReflex

## Early portion of code consisted mostly of directory setup and library imports

### Library Imports
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

### Directory setup
Specifying my base directory and then the categories we'll be using. The directory had struggle similar to
        
    PetImages -> Dog / Images ** This is my dog training images
                 Cat / Images ** This is my cat training images

```python
DATADIR = "C:/Users/Jay/Desktop/Machine_Learning/kagglecatsanddogs_3367a/PetImages"

### Specify categories
CATEGORIES = ["Dog","Cat"]
```

## Verify the data loads

### But before building the training data, its best to check that images can be loaded

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

## Build the training data
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
