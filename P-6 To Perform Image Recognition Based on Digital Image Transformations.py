#!/usr/bin/env python
# coding: utf-8

# FOR COLAB

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Function to display images
def show_image(img, title='Image'):
    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()






# Import color Image
image_path_gray = '/content/dog.jpg'
gray_image = cv2.imread(image_path_gray)
cv2_imshow(gray_image)





# Import color Image
image_path_color = '/content/trail.jpg'
color_image = cv2.imread(image_path_color)
cv2_imshow(color_image)






# Detect the boundaries in the color image
edges_color = cv2.Canny(color_image, 100, 200)
show_image(edges_color, 'Edges - Color Image')

# Detect the boundaries in the grayscale image
edges_gray = cv2.Canny(gray_image, 100, 200)
show_image(edges_gray, 'Edges - Grayscale Image')





# Convert grayscale image to grayscale (redundant)
gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

# Apply DCT
dct = cv2.dct(np.float32(gray_image))
dct_norm = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX)
show_image(dct_norm, 'DCT')





# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(color_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

show_image(color_image, 'Faces Detected')


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
import matplotlib.pyplot as plt

# Function to display images using matplotlib
def show_image(img, title='Image', cmap_type='gray'):
    plt.figure(figsize=(10, 5))
    if len(img.shape) == 3:  # If the image has color channels
        cmap_type = None
    plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Import grayscale image
image_path_gray = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\dog.jpg"  # Adjust path if needed
gray_image = cv2.imread(image_path_gray, cv2.IMREAD_GRAYSCALE)
show_image(gray_image, 'Grayscale Image')

# Import color image
image_path_color = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\trail.jpg"  # Adjust path if needed
color_image = cv2.imread(image_path_color)
color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
show_image(color_image_rgb, 'Color Image', cmap_type=None)

# Detect edges in the color image
edges_color = cv2.Canny(color_image, 100, 200)
show_image(edges_color, 'Edges - Color Image')

# Detect edges in the grayscale image
edges_gray = cv2.Canny(gray_image, 100, 200)
show_image(edges_gray, 'Edges - Grayscale Image')

# Apply DCT on the grayscale image
dct = cv2.dct(np.float32(gray_image))
dct_norm = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX)
show_image(dct_norm, 'DCT')

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(color_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(color_image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

show_image(color_image_rgb, 'Faces Detected', cmap_type=None)

