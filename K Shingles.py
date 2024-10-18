#!/usr/bin/env python
# coding: utf-8

# In[2]:


import string

def generate_k_shingles_word(filename, k):
  shingles = set()
  with open(filename, 'r') as file:
    text = file.read()
    # Remove punctuation and brackets
    text = text.translate(str.maketrans('', '', string.punctuation + '[]'))
    words = text.split()
    # Initialize an empty list to store results
    output = []
    # Loop to generate substrings of length k
    for i in range(len(words) - k + 1):
        substring = ' '.join(words[i:i + k])
        output.append(substring)
  return output

k = 5  #value of k for k-shingles
shingles1 = generate_k_shingles_word("C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\BIG DATA\\file3.txt", k)
shingles2 = generate_k_shingles_word("C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\BIG DATA\\file1.txt", k)
shingles3 = generate_k_shingles_word("C:\\Users\\Lalit\\OneDrive\\Desktop\\CODES\\BIG DATA\\file2.txt", k)
print(f"Shingles for file1.txt:  \n {shingles1}\n\n")
print(f"Shingles for file2.txt:  \n {shingles2}\n\n")
print(f"Shingles for file3.txt:  \n {shingles3}\n\n")




# Combine the shingles
combined_list = shingles1 + shingles2 + shingles3
unique_list = list(set(combined_list))
# Sort the unique list
unique_list.sort()
print("Combined list with duplicates removed:\n", unique_list)
print(f"\n\nfile 1 lenght: {len(shingles1)} \nfile 2 length: {len(shingles2)} \nfile 3 length: {len(shingles3)}")
print(f"Total Length Before: {len(shingles1)+len(shingles2)+len(shingles3)}")
print("Combined List Length After Deduplication:",len(unique_list))





import pandas as pd
#Create the Characteristic Matrix
def create_char_matrix(documents, unique_shingles):
    # Initialize an empty DataFrame with shingles as row indices and documents as columns
    char_mat = pd.DataFrame(index=unique_shingles, columns=[f'file_{i+1}' for i in range(len(documents))])
    # Fill the DataFrame with binary values
    for i, doc in enumerate(documents):
        char_mat[f'file_{i+1}'] = char_mat.index.to_series().apply(lambda x: 1 if x in doc else 0)
    char_mat.insert(0, 'Row Number', range(0,len(char_mat)))
    return char_mat

# List of shingles documents
documents = [shingles1, shingles2, shingles3]
# Generate the Characteristic Matrix
Char_Mat = create_char_matrix(documents, unique_list)
print("\nCharacteristic Matrix:")
print(Char_Mat)




#
Char_Mat['h1'] = (Char_Mat['Row Number'] + 1) % 257
Char_Mat['h2'] = (2 * Char_Mat['Row Number'] + 5) % 257
# Sort the DataFrame by h1 and h2
Char_Mat = Char_Mat.sort_values(by=['h1', 'h2'])
# Print the updated DataFrame
print(Char_Mat)



# To gnerate Signature Matrix
def generate_signature_matrix(char_matrix, hash_functions):
  signature_matrix = pd.DataFrame(index=hash_functions, columns=char_matrix.columns[1:-len(hash_functions)])
  for doc in signature_matrix.columns:
    for hash_func in hash_functions:
      # Find the first row where the document has a '1' for the current hash function
      first_one_row = char_matrix.loc[(char_matrix[doc] == 1) & (char_matrix[hash_func] != 0), hash_func].iloc[0] if any((char_matrix[doc] == 1) & (char_matrix[hash_func] != 0)) else None
      signature_matrix.at[hash_func, doc] = first_one_row
  return signature_matrix

hash_functions = ['h1', 'h2']
signature_matrix = generate_signature_matrix(Char_Mat, hash_functions)
print("\nSignature Matrix:")
print(signature_matrix)

