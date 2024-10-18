#!/usr/bin/env python
# coding: utf-8

# In[ ]:


METHOD 1:


# In[1]:


import math

# Define a simple manual hash function for numbers
def manual_hash(x):
    return (3 * x + 7) % 32

# Convert the hash value to a binary string
def hash_to_binary(x):
    return bin(manual_hash(x))[2:]

# Function to count trailing zeros in a binary string
def count_trailing_zeros(binary_string):
    return len(binary_string) - len(binary_string.rstrip('0'))

# Flajolet-Martin algorithm to estimate the number of distinct items
def flajolet_martin(stream):
    max_trailing_zeros = 0

    for item in stream:
        # Get binary hash of the number using the manual hash function
        binary_hash = hash_to_binary(item)
        # Count trailing zeros in the binary hash
        trailing_zeros = count_trailing_zeros(binary_hash)
        # Keep track of the maximum number of trailing zeros
        max_trailing_zeros = max(max_trailing_zeros, trailing_zeros)

    # The estimate of distinct items is 2 raised to the power of max_trailing_zeros
    estimate = 2 ** max_trailing_zeros
    return estimate, max_trailing_zeros

# Example usage with a stream of numbers
stream = [3, 1, 4, 1, 5, 9, 2, 6, 5]
distinct_estimate , max_zero = flajolet_martin(stream)
print(f"Estimated number of distinct items: {distinct_estimate}")
print(f"Max Traling Zero is {max_zero}")


# In[ ]:





# METHOD 2:

# In[2]:


# Custom hash function: (3x + 7) mod 32
def custom_hash_function(x):
    return (3 * x + 7) % 32

# Function to count the number of trailing zeros in binary representation
def count_trailing_zeros(n):
    if n == 0:
        return 32  # Edge case for 0
    binary_rep = bin(n)[::-1]  # Reverse the binary representation
    return binary_rep.find('1')  # Find the position of the first '1'

# Flajolet-Martin implementation for custom hash function
def flajolet_martin(stream):
    max_trailing_zeros = 0  # Store max number of trailing zeros

    for element in stream:
        # Get the hashed value using the custom hash function
        hashed_value = custom_hash_function(element)
        # Count trailing zeros in the binary representation of the hash
        num_trailing_zeros = count_trailing_zeros(hashed_value)
        # Keep track of the maximum number of trailing zeros seen
        max_trailing_zeros = max(max_trailing_zeros, num_trailing_zeros)

    # The estimated number of distinct elements is 2^max_trailing_zeros
    return max_trailing_zeros


# Given elements
stream = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Compute R (2^r) where r is the max number of trailing zeros
r = flajolet_martin(stream)
R = 2 ** r

print(f"The value of R (2^r) is: {R}")
print(f"The value of r (max trailing zeros) is: {r}")

