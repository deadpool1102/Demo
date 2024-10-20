#!/usr/bin/env python
# coding: utf-8

# FOR COLAB

# In[ ]:


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Path to the image files in your Google Drive
image_path1 = '/content/lion.jpg'
image_path2 = '/content/road.jpg'

# Function to read, resize, and copy an image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Display the original image
    print("Original Image:")
    cv2_imshow(image)
    print("Original Shape of the image:", image.shape)
    print("Type of the image:", type(image))

    # Resize the image
    new_size = (800, 400)
    resized_image = cv2.resize(image, new_size)

    # Display the resized image
    print("Resized Image:")
    cv2_imshow(resized_image)
    print("Resized Shape of the image:", resized_image.shape)
    print("Type of the image:", type(resized_image))

    # Copy the resized image
    copied_image = resized_image.copy()

    # Display the copied image
    print("Copied Image:")
    cv2_imshow(copied_image)

# Process the first image
process_image(image_path1)

# Process the second image
process_image(image_path2)

# Creating a binary matrix as an image
a = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])

# Convert matrix to image
image = (a * 255).astype(np.uint8)  # Convert to 0s and 255s for display
print("Binary Image:")
cv2_imshow(image)
print("Image size:", image.size)
print("Image shape:", image.shape)

# Resize the created image
resized_image = cv2.resize(image, (800, 400), interpolation=cv2.INTER_NEAREST)
print("Resized Binary Image:")
cv2_imshow(resized_image)
print("Resized Image size:", resized_image.size)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# FOR JUPYTER

# In[1]:


import cv2
import numpy as np
from IPython.display import display, Image  # For image display in Jupyter

# Function to display image in Jupyter Notebook
def jupyter_imshow(image, title="Image"):
    # Convert BGR (OpenCV format) to RGB (display format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, encoded_img = cv2.imencode('.png', image_rgb)  # Encode image to PNG
    display(Image(data=encoded_img.tobytes()))  # Display the image inline

# Path to the image files
image_path1 = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\lion.jpg"
image_path2 = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\road.jpg"

# Function to read, resize, and copy an image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Display the original image
    print("Original Image:")
    jupyter_imshow(image)
    print("Original Shape of the image:", image.shape)
    print("Type of the image:", type(image))

    # Resize the image
    new_size = (800, 400)
    resized_image = cv2.resize(image, new_size)

    # Display the resized image
    print("Resized Image:")
    jupyter_imshow(resized_image)
    print("Resized Shape of the image:", resized_image.shape)
    print("Type of the image:", type(resized_image))

    # Copy the resized image
    copied_image = resized_image.copy()
    
    # Display the copied image
    print("Copied Image:")
    jupyter_imshow(copied_image)

# Process the first image
process_image(image_path1)

# Process the second image
process_image(image_path2)

# Creating a binary matrix as an image
a = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])

# Convert matrix to image
image = (a * 255).astype(np.uint8)  # Convert to 0s and 255s for display
print("Binary Image:")
jupyter_imshow(image)
print("Image size:", image.size)
print("Image shape:", image.shape)

# Resize the created image
resized_image = cv2.resize(image, (800, 400), interpolation=cv2.INTER_NEAREST)
print("Resized Binary Image:")
jupyter_imshow(resized_image)
print("Resized Image size:", resized_image.size)

