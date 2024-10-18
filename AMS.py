#!/usr/bin/env python
# coding: utf-8

# In[7]:


from collections import defaultdict

def ams_algorithm(stream, x_values):
    # Initialize counters for the frequency of each element in the stream
    frequency = defaultdict(int)

    # Process the stream to count frequencies
    for num in stream:
        frequency[num] += 1

    # Initialize AMS estimates for each x value
    estimates = []

    for x in x_values:
        # Get the element at index x
        element = stream[x]

        # Count occurrences of the element from index x to the end of the stream
        f_i = sum(1 for i in range(x, len(stream)) if stream[i] == element)

        # Calculate the AMS estimate for this x value
        estimate = len(stream) * (2 * f_i - 1)
        estimates.append(estimate)

        # Print the frequency and estimate value
        print(f"x={x}: Element={element}, Frequency={f_i}, Estimate={estimate}")

    # Calculate the average of the AMS estimates to get the surprise number
    surprise_number = sum(estimates) / len(estimates)
    return surprise_number

# Given stream and x values
stream = [1, 2, 7, 1, 4, 9, 4, 6, 1, 6, 4, 4, 5, 5, 5, 9, 8, 7, 2, 2, 4, 4, 1]
x_values = [1, 4, 7]  # Indices of the stream

# Compute the surprise number using the AMS algorithm
surprise_number = ams_algorithm(stream, x_values)
print(f"\nEstimated Surprise Number: {surprise_number}")

