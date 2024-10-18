#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# Define the input matrix
matrix = np.array([
    (1, 0, 0, 1, 0),  # Set 1
    (0, 0, 1, 0, 0),  # Set 2
    (0, 1, 0, 1, 1),  # Set 3
    (1, 0, 1, 1, 0)   # Set 4
])

# Define the hash functions
def h1(x):
    return (x + 1) % 5

def h2(x):
    return (3 * x + 1) % 5

def h3(x):
    return (4 * x + 1) % 5

# List of hash functions
hash_functions = [h1, h2, h3]

# Initialize the signature matrix with infinity
num_hashes = len(hash_functions)
num_sets, num_elements = matrix.shape
signature_matrix = np.full((num_hashes, num_sets), np.inf)

# Compute the MinHash signatures
for i in range(num_elements):
    if any(matrix[:, i]):  # If the column has a '1'
        for h in range(num_hashes):
            hash_value = hash_functions[h](i)
            for s in range(num_sets):
                if matrix[s, i] == 1:
                    signature_matrix[h, s] = min(signature_matrix[h, s], hash_value)

# Print the signature matrix
print("MinHash Signature Matrix:")
print(signature_matrix)

