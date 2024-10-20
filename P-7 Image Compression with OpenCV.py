#!/usr/bin/env python
# coding: utf-8

# FOR COLAB

# In[ ]:


import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os

# Load the images
image_path = '/content/nature1.jpg'
image_path2 = '/content/nature2.jpg'
image = cv2.imread(image_path)
image2 = cv2.imread(image_path2)

# Display the original images
print("Displaying Original Images:")
print("Image 1:")
cv2_imshow(image)
print("Image 2:")
cv2_imshow(image2)

# Save the images with lossless and lossy compression
cv2.imwrite('image_lossless.png', image)
cv2.imwrite('image_lossless2.png', image2)
cv2.imwrite('image_lossy.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 10])
cv2.imwrite('image_lossy2.jpg', image2, [cv2.IMWRITE_JPEG_QUALITY, 10])

# Reload the compressed images
image_lossless = cv2.imread('image_lossless.png')
image_lossy = cv2.imread('image_lossy.jpg')
image_lossless2 = cv2.imread('image_lossless2.png')
image_lossy2 = cv2.imread('image_lossy2.jpg')

# Plot the images with labels for Image 1
plt.figure(figsize=(12, 6))
plt.suptitle('Comparing Compressions for Image 1', fontsize=16)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image_lossless, cv2.COLOR_BGR2RGB))
plt.title('Lossless Compression (PNG) 1')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image_lossy, cv2.COLOR_BGR2RGB))
plt.title('Lossy Compression (JPEG) 1')
plt.axis('off')

plt.show()

# Plot the images with labels for Image 2
plt.figure(figsize=(12, 6))
plt.suptitle('Comparing Compressions for Image 2', fontsize=16)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title('Original Image 2')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image_lossless2, cv2.COLOR_BGR2RGB))
plt.title('Lossless Compression (PNG) 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image_lossy2, cv2.COLOR_BGR2RGB))
plt.title('Lossy Compression (JPEG) 2')
plt.axis('off')

plt.show()

# Get and print file sizes
original_size = os.path.getsize(image_path)
lossless_size = os.path.getsize('image_lossless.png')
lossy_size = os.path.getsize('image_lossy.jpg')

print("\nImage 1 - Original Shape:", image.shape)
print("Lossless Shape:", image_lossless.shape)
print("Lossy Shape:", image_lossy.shape)
print(f"Original Size: {original_size / 1024:.2f} KB")
print(f"Lossless Size: {lossless_size / 1024:.2f} KB")
print(f"Lossy Size: {lossy_size / 1024:.2f} KB")

original_size2 = os.path.getsize(image_path2)
lossless_size2 = os.path.getsize('image_lossless2.png')
lossy_size2 = os.path.getsize('image_lossy2.jpg')

print("\nImage 2 - Original Shape:", image2.shape)
print("Lossless Shape:", image_lossless2.shape)
print("Lossy Shape:", image_lossy2.shape)
print(f"Original Size: {original_size2 / 1024:.2f} KB")
print(f"Lossless Size: {lossless_size2 / 1024:.2f} KB")
print(f"Lossy Size: {lossy_size2 / 1024:.2f} KB")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# FOR JUPYTER

# In[3]:


import cv2
import matplotlib.pyplot as plt
import os

# Load the images
image_path = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\nature1.jpg"
image_path2 = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\nature2.jpg"
image = cv2.imread(image_path)
image2 = cv2.imread(image_path2)

# Save with lossless and lossy compression
cv2.imwrite('image_lossless.png', image)
cv2.imwrite('image_lossless2.png', image2)
cv2.imwrite('image_lossy.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 10])
cv2.imwrite('image_lossy2.jpg', image2, [cv2.IMWRITE_JPEG_QUALITY, 10])

# Reload the compressed images
image_lossless = cv2.imread('image_lossless.png')
image_lossy = cv2.imread('image_lossy.jpg')
image_lossless2 = cv2.imread('image_lossless2.png')
image_lossy2 = cv2.imread('image_lossy2.jpg')

# Create a figure with two grids for both images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# First Image - Original, Lossless, and Lossy
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image 1')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(image_lossless, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Lossless Compression (PNG) 1')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(image_lossy, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Lossy Compression (JPEG) 1')
axes[0, 2].axis('off')

# Second Image - Original, Lossless, and Lossy
axes[1, 0].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Original Image 2')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(image_lossless2, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Lossless Compression (PNG) 2')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(image_lossy2, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title('Lossy Compression (JPEG) 2')
axes[1, 2].axis('off')

# Display the complete grid
plt.tight_layout()
plt.show()

# Get and print file sizes
original_size = os.path.getsize(image_path)
lossless_size = os.path.getsize('image_lossless.png')
lossy_size = os.path.getsize('image_lossy.jpg')

print("Image 1 - Original Shape:", image.shape)
print("Lossless Shape:", image_lossless.shape)
print("Lossy Shape:", image_lossy.shape)
print(f"Original Size: {original_size / 1024:.2f} KB")
print(f"Lossless Size: {lossless_size / 1024:.2f} KB")
print(f"Lossy Size: {lossy_size / 1024:.2f} KB\n")

original_size2 = os.path.getsize(image_path2)
lossless_size2 = os.path.getsize('image_lossless2.png')
lossy_size2 = os.path.getsize('image_lossy2.jpg')

print("Image 2 - Original Shape:", image2.shape)
print("Lossless Shape:", image_lossless2.shape)
print("Lossy Shape:", image_lossy2.shape)
print(f"Original Size: {original_size2 / 1024:.2f} KB")
print(f"Lossless Size: {lossless_size2 / 1024:.2f} KB")
print(f"Lossy Size: {lossy_size2 / 1024:.2f} KB")

