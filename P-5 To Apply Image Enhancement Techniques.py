#!/usr/bin/env python
# coding: utf-8

# FOR COLAB

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load and display the color image
image_path = '/content/lion.jpg'
color_image = cv2.imread(image_path)

print("Color Image:")
cv2_imshow(color_image)

# Convert the image to grayscale and display it
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

print("Grayscale Image:")
cv2_imshow(gray_image)

# Create a black-and-white image and resize it
image = np.array([[255, 0, 0, 255],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [255, 0, 0, 255]], dtype=np.uint8)

new_size = (800, 400)
black_white_image = cv2.resize(image, new_size)

print("Black and White Image:")
cv2_imshow(black_white_image)

# Adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 50, beta=brightness)
    return adjusted

bright_image = adjust_brightness_contrast(color_image, brightness=30, contrast=50)

print("Brightened Color Image:")
cv2_imshow(bright_image)

bright_gray_image = adjust_brightness_contrast(gray_image, brightness=30, contrast=50)

print("Brightened Grayscale Image:")
cv2_imshow(bright_gray_image)

bright_black_white_image = adjust_brightness_contrast(black_white_image, brightness=30, contrast=50)

print("Brightened Black and White Image:")
cv2_imshow(bright_black_white_image)

# Create and display negative images
negative_image = 255 - color_image

print("Negative Color Image:")
cv2_imshow(negative_image)

negative_gray_image = 255 - gray_image

print("Negative Grayscale Image:")
cv2_imshow(negative_gray_image)

negative_black_white_image = 255 - black_white_image

print("Negative Black and White Image:")
cv2_imshow(negative_black_white_image)

# Extract and print RGB values
blue, green, red = cv2.split(color_image)
print("Blue Channel:\n", blue)
print("Green Channel:\n", green)
print("Red Channel:\n", red)

# Create custom negative RGB image
negative_blue = 255 - blue
negative_green = 255 - green
negative_red = 255 - red

negative_image_custom = cv2.merge((negative_blue, negative_green, negative_red))

# Function to plot images with titles
def plot_images(images, titles, cmap=None):
    plt.figure(figsize=(20, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap=cmap if cmap else 'gray')
        else:  # Color image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# Plot multiple images with titles
images1 = [color_image, bright_image, negative_image]
titles1 = ['Original Color Image', 'Brightened Image', 'Negative Image']
plot_images(images1, titles1)

images2 = [gray_image, bright_gray_image, negative_gray_image]
titles2 = ['Original Gray Image', 'Brightened Image', 'Negative Image']
plot_images(images2, titles2, cmap='gray')

images3 = [black_white_image, bright_black_white_image, negative_black_white_image]
titles3 = ['Original Black and White Image', 'Brightened Image', 'Negative Image']
plot_images(images3, titles3, cmap='gray')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# FOR JUPYTER

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\lion.jpg"  # Make sure the path points to your local file
color_image = cv2.imread(image_path)

# Display using matplotlib for compatibility with Jupyter
def show_image(image, title="Image", cmap=None):
    plt.figure(figsize=(6, 6))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap=cmap if cmap else 'gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display the color image
show_image(color_image, title="Color Image")

# Convert the image to grayscale
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
show_image(gray_image, title="Grayscale Image", cmap='gray')

# Creating a Black and White image
image = np.array([[255, 0, 0, 255],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [255, 0, 0, 255]], dtype=np.uint8)

# Resize the black and white image
new_size = (800, 400)
black_white_image = cv2.resize(image, new_size)

# Display the resized black and white image
show_image(black_white_image, title="Black and White Image", cmap='gray')

# Adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 50, beta=brightness)
    return adjusted

# Brightness and contrast adjustments
bright_image = adjust_brightness_contrast(color_image, brightness=30, contrast=50)
show_image(bright_image, title="Brightened Color Image")

bright_gray_image = adjust_brightness_contrast(gray_image, brightness=30, contrast=50)
show_image(bright_gray_image, title="Brightened Grayscale Image", cmap='gray')

bright_black_white_image = adjust_brightness_contrast(black_white_image, brightness=30, contrast=50)
show_image(bright_black_white_image, title="Brightened Black and White Image", cmap='gray')

# Create negative images
negative_image = 255 - color_image
show_image(negative_image, title="Negative Color Image")

negative_gray_image = 255 - gray_image
show_image(negative_gray_image, title="Negative Grayscale Image", cmap='gray')

negative_black_white_image = 255 - black_white_image
show_image(negative_black_white_image, title="Negative Black and White Image", cmap='gray')

# Extracting RGB values
blue, green, red = cv2.split(color_image)
print("Blue:", blue)
print("Green:", green)
print("Red:", red)

# Reversing RGB values
negative_blue = 255 - blue
negative_green = 255 - green
negative_red = 255 - red

# Create a new pixel value from the modified colors
negative_image_custom = cv2.merge((negative_blue, negative_green, negative_red))

# Plot multiple images using matplotlib
def plot_images(images, titles, cmap=None):
    plt.figure(figsize=(20, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap=cmap if cmap else 'gray')
        else:  # Color image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()

# Plot the sets of images
images1 = [color_image, bright_image, negative_image]
titles1 = ['Original Color Image', 'Brightness and Contrast Image', 'Negative Image']
plot_images(images1, titles1)

images2 = [gray_image, bright_gray_image, negative_gray_image]
titles2 = ['Original Gray Image', 'Brightness and Contrast Image', 'Negative Image']
plot_images(images2, titles2, cmap='gray')

images3 = [black_white_image, bright_black_white_image, negative_black_white_image]
titles3 = ['Original Black and White Image', 'Brightness and Contrast Image', 'Negative Image']
plot_images(images3, titles3, cmap='gray')


# In[ ]:




