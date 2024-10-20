#!/usr/bin/env python
# coding: utf-8

# FOR COLAB

# In[ ]:


import cv2
from google.colab.patches import cv2_imshow

# Load and display the first image (color)
image_path = '/content/lion.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Print the entire pixel array
    print("\n--- Image 1 (Color) Pixel Array ---")
    print(image)  # Entire pixel array

    # Display the image
    cv2_imshow(image)

    # Print the truncated pixel array
    print("\n--- Image 1 (Color) Truncated Pixel Array ---")
    print(image[:5, :5])  # Truncated pixel array (first 5x5 pixels)

    # Save the processed image
    new_image_path = '/content/processed_image.jpg'
    cv2.imwrite(new_image_path, image)






# Load and display the second image (grayscale)
image_path = '/content/road.jpg'
image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image2 is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Print the entire pixel array
    print("\n--- Image 2 (Grayscale) Pixel Array ---")
    print(image2)  # Entire pixel array

    # Display the image
    cv2_imshow(image2)

    # Print the truncated pixel array
    print("\n--- Image 2 (Grayscale) Truncated Pixel Array ---")
    print(image2[:5, :5])  # Truncated pixel array (first 5x5 pixels)

    # Save the grayscale image
    new_image_path = '/content/greyscale.jpg'
    cv2.imwrite(new_image_path, image2)


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

# Function to display images inline in Jupyter
def display_image(image, title="Image", cmap=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load and display the first image (color)
image_path = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\lion.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Print the entire pixel array
    print("\n--- Image 1 (Color) Pixel Array ---")
    print(image)  # Entire pixel array
    
    # Display the image inline using matplotlib
    display_image(image, title="Image 1 (Color)")

    # Print the truncated pixel array
    print("\n--- Image 1 (Color) Truncated Pixel Array ---")
    print(image[:5, :5])  # Truncated pixel array (first 5x5 pixels)
    
    # Save the processed image
    new_image_path = 'processed_image.jpg'
    cv2.imwrite(new_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving

    
    
# Load and display the second image (grayscale)
image_path = "C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\IMAGE AND VIDEO ANALYTICS\\road.jpg"
image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image2 is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Print the entire pixel array
    print("\n--- Image 2 (Grayscale) Pixel Array ---")
    print(image2)  # Entire pixel array
    
    # Display the image inline using matplotlib (with grayscale colormap)
    display_image(image2, title="Image 2 (Grayscale)", cmap='gray')

    # Print the truncated pixel array
    print("\n--- Image 2 (Grayscale) Truncated Pixel Array ---")
    print(image2[:5, :5])  # Truncated pixel array (first 5x5 pixels)
    
    # Save the grayscale image
    new_image_path = 'greyscale.jpg'
    cv2.imwrite(new_image_path, image2)

