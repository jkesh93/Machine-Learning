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
# Specify a data directory -- base directory from which w
DATADIR = "C:/Users/Jay/Desktop/Machine_Learning/kagglecatsanddogs_3367a/PetImages"

### Specify categories
CATEGORIES = ["Dog","Cat"]

### Iterate through all examples of dog and cat
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
