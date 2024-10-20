#!/usr/bin/env python
# coding: utf-8

# In[ ]:


FOR COLAB


# In[ ]:


import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Create a black image of size 512x512 for the line
image = np.zeros((512, 512, 3), np.uint8)
cv2.line(image, (0, 0), (500, 500), (255, 255, 255), 5)
cv2_imshow(image)


# Create a black image of size 512x512 for the arrowed line
image = np.zeros((512, 512, 3), np.uint8)
cv2.arrowedLine(image, (500, 0), (0, 500), (255, 255, 255), 5)
cv2_imshow(image) 


# Create a black image of size 512x512 for the rectangle
image = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), 5)
cv2_imshow(image)


# Create a black image of size 512x512 for the circle
image = np.zeros((512, 512, 3), np.uint8)
cv2.circle(image, (250, 250), 100, (255, 255, 255), 5)
cv2_imshow(image)


# Create a black image of size 512x512 for the text
image = np.zeros((512, 512, 3), np.uint8)
cv2.putText(image, 'OpenCV', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
cv2_imshow(image)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


FOR JUPYTER


# In[3]:


import cv2
import numpy as np
from IPython.display import display, Image as IPImage

# Helper function to display an image in Jupyter Notebook
def show_image(image):
    _, encoded_img = cv2.imencode('.png', image)  # Encode image as PNG
    display(IPImage(data=encoded_img))            # Display the image

# Create a black image of size 512x512 for the line
image = np.zeros((512, 512, 3), np.uint8)
cv2.line(image, (0, 0), (500, 500), (255, 255, 255), 5)
show_image(image)

# Create a black image of size 512x512 for the arrowed line
image = np.zeros((512, 512, 3), np.uint8)
cv2.arrowedLine(image, (500, 0), (0, 500), (255, 255, 255), 5)
show_image(image)

# Create a black image of size 512x512 for the rectangle
image = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(image, (100, 100), (400, 400), (255, 255, 255), 5)
show_image(image)

# Create a black image of size 512x512 for the circle
image = np.zeros((512, 512, 3), np.uint8)
cv2.circle(image, (250, 250), 100, (255, 255, 255), 5)
show_image(image)

# Create a black image of size 512x512 for the text
image = np.zeros((512, 512, 3), np.uint8)
cv2.putText(image, 'OpenCV', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
show_image(image)

