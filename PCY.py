#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

# Transaction Data
transactions = [
    {1, 2, 3},
    {2, 3, 4},
    {3, 4, 5},
    {4, 5, 6},
    {1, 3, 5},
    {2, 4, 6},
    {1, 3, 4},
    {2, 4, 5},
    {3, 4, 6},
    {1, 2, 4},
    {2, 3, 5},
    {2, 4, 6}
]

# Minimum Support Threshold
min_support = 3

# Hash Function (i * j) % 10
def hash_function(pair):
    return (pair[0] * pair[1]) % 10

# First Pass - Count individual items and hash pairs
item_count = Counter()
bucket_count = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        item_count[item] += 1
    for pair in combinations(transaction, 2):
        bucket_count[hash_function(pair)] += 1

# Create Bitmap
bitmap = {bucket: 1 if count >= min_support else 0 for bucket, count in bucket_count.items()}

# Second Pass - Count candidate pairs
candidate_pairs = Counter()
for transaction in transactions:
    for pair in combinations(transaction, 2):
        if bitmap[hash_function(pair)] == 1:
            candidate_pairs[pair] += 1

# Filter Candidate Pairs by Minimum Support
frequent_pairs = {pair: count for pair, count in candidate_pairs.items() if count >= min_support}

# Prepare Data for DataFrame
data = []
for bucket, count in bucket_count.items():
    pairs_in_bucket = [pair for pair in frequent_pairs if hash_function(pair) == bucket]
    if pairs_in_bucket:
        highest_support = max(frequent_pairs[pair] for pair in pairs_in_bucket)
        for pair in pairs_in_bucket:
            data.append({
                'Bit Vector': bitmap[bucket],
                'Bucket No.': bucket,
                'Highest Support Count': highest_support,
                'Pairs': pair,
                'Candidate Set': pair
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# Print DataFrame
print(df)

