#!/usr/bin/env python
# coding: utf-8

# USING CNN TO APPLY FILTERS ON THE GIVEN IMAGE.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb
from scipy.signal import convolve2d




#Reading Image
Lion = imread(" ")
plt.figure(num=None, figsize=(8,6), dpi=80)
imshow(Lion)




#Filter MAtrices

sharpen = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]])

blur = np.array([[0.11,0.11,0.11],
                    [0.11,0.11,0.11],
                    [0.11,0.11,0.11]])


vertical = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])


gaussian = (1/16.0) * np.array([[1,2,1],
                                [2,4,2],
                                [1,2,1]])




#plotting the filters
fig,ax = plt.subplots(1,3, figsize = (17,10))
ax[0].imshow(sharpen, cmap='gray')
ax[0].set_title(f'Sharpen', fontsize=18)

ax[1].imshow(blur, cmap='gray')
ax[1].set_title(f'Blur', fontsize=18)

ax[2].imshow(vertical, cmap='gray')
ax[2].set_title(f'Vertical', fontsize=18)




#Grayscaling Image
Lion_gray = rgb2gray(Lion)
plt.figure(num=None, figsize=(8,6), dpi=80)
imshow(Lion_gray)




#Function for applying filters
def multi_convolver(image, kernel, iterations):
  for i in range(iterations):
    image = convolve2d(image, kernel, 'same', boundary = 'fill', fillvalue = 0)
  return image

convolved_image = multi_convolver(Lion_gray, sharpen, 1)

plt.figure(num=None, figsize=(8,6), dpi=80)
imshow(convolved_image);




#For colored Image
def convolver_rgb(image, kernel, iterations = 1):
  convolved_image_r = multi_convolver(image[:,:,0], kernel, iterations)
  convolved_image_g = multi_convolver(image[:,:,1], kernel, iterations)
  convolved_image_b = multi_convolver(image[:,:,2], kernel, iterations)

  reformed_image = np.dstack((np.rint(abs(convolved_image_r)),np.rint(abs(convolved_image_g)),np.rint(abs(convolved_image_b))))/255

  fig,ax = plt.subplots(1,3, figsize = (17,10))

  ax[0].imshow(abs(convolved_image_r), cmap='Reds')
  ax[0].set_title(f'Red', fontsize=15)

  ax[1].imshow(abs(convolved_image_g), cmap='Greens')
  ax[1].set_title(f'Green', fontsize=18)

  ax[2].imshow(abs(convolved_image_b), cmap='Blues')
  ax[2].set_title(f'Blue', fontsize=18)

  return np.array(reformed_image*255).astype(np.uint8)




#Can add different filters (defined above) here
convolved_rgb_gauss = convolver_rgb(Lion, vertical.T ,1)




plt.figure(num=None, figsize=(8,6), dpi=80)
plt.imshow(convolved_rgb_gauss,vmin=0,vmax=255);


# In[ ]:





# CNN PARAMETER CALCULATION

# In[10]:


from keras.models import Sequential
from keras.layers import Conv2D




model = Sequential()
model.add(Conv2D(32, input_shape=(28,28,3),
                 kernel_size = (5,5),
                 padding='same',
                 use_bias=False))
model.add(Conv2D(17, (3,3), padding='same', use_bias=False))
model.add(Conv2D(13, (3,3), padding='same', use_bias=False))
model.add(Conv2D(7, (3,3), padding='same', use_bias=False))

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

model.summary()




model = Sequential()
model.add(Conv2D(32, input_shape=(28,28,3),
                 kernel_size = (5,5),
                 padding='same',
                 use_bias=True))
model.add(Conv2D(17, (3,3), padding='same', use_bias=True))
model.add(Conv2D(13, (3,3), padding='same', use_bias=True))
model.add(Conv2D(7, (3,3), padding='same', use_bias=True))

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

model.summary()




model = Sequential()
model.add(Conv2D(10, input_shape=(28,28,3),
                 kernel_size = (5,5),
                 strides = (1,1),
                 padding='valid',
                 use_bias=False))
model.add(Conv2D(20, (5,5), (2,2), padding='valid', use_bias=False))
model.add(Conv2D(40, (5,5), (2,2), padding='valid', use_bias=False))

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

model.summary()




model = Sequential()
model.add(Conv2D(10, input_shape=(39,39,3),
                 kernel_size = (3,3),
                 padding='valid',))
model.add(Conv2D(20, (5,5), (2,2), padding='valid' ))
model.add(Conv2D(40, (5,5), (2,2),padding='valid'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam')

model.summary()


# In[ ]:





# MAX POOLING

# In[11]:


import tensorflow as tf
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]])
x = tf.reshape(x, [1, 3, 3, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
max_pool_2d(x)




x = tf.constant([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
x = tf.reshape(x, [1, 3, 4, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(2, 2), padding='valid')
max_pool_2d(x)




x = tf.constant([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
x = tf.reshape(x, [1, 3, 4, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(2, 2), padding='same')
max_pool_2d(x)




x = tf.constant([[1., 2., 3., 4.],
                 [5., 6., 7., 8.],
                 [9., 10., 11., 12.]])
x = tf.reshape(x, [1, 3, 4, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
   strides=(1, 1), padding='same')
max_pool_2d(x)





