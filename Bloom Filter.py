#!/usr/bin/env python
# coding: utf-8

# In[1]:


class BloomFilter:
    def __init__(self, size):
        self.size = size
        self.bit_array = [0] * size

    def hash1(self, x):
        return (x + 1) % self.size

    def hash2(self, x):
        return (2 * x + 3) % self.size

    def add(self, item):
        # Apply both hash functions
        index1 = self.hash1(item)
        index2 = self.hash2(item)

        # Set the bits at the calculated positions
        self.bit_array[index1] = 1
        self.bit_array[index2] = 1

    def check(self, item):
        # Apply both hash functions
        index1 = self.hash1(item)
        index2 = self.hash2(item)

        # Check if both bits are set
        return self.bit_array[index1] == 1 and self.bit_array[index2] == 1

    def get_bit_array(self):
        return self.bit_array

# Initialize Bloom filter with 13 bits
bloom_filter = BloomFilter(13)

# Add numbers to the Bloom filter
numbers_to_add = [8, 17, 25, 14, 20]
for number in numbers_to_add:
    bloom_filter.add(number)

# Get the final state of the Bloom filter
final_bit_array = bloom_filter.get_bit_array()
print(f"Final Bloom Filter Bit Array: {final_bit_array}")

# Check if numbers 7 and 5 are in the Bloom filter
check_numbers = [7, 5]
results = {num: bloom_filter.check(num) for num in check_numbers}

# Output the results
for num, result in results.items():
    print(f"Number {num} is {'probably in the set' if result else 'not in the set'}")

