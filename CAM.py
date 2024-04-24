import numpy as np


# Define memory vectors
mem_vectors = np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0]
])

# Get dimensions
q, n = mem_vectors.shape

# Bipolarize memory vectors(-1 and 1 mein covert)
bip_mem_vectors = 2 * mem_vectors - 1

# Initialize weight matrix
zd_wt_mat = np.zeros((n, n))

# Compute weight matrix
for i in range(q):
    zd_wt_mat += np.outer(bip_mem_vectors[i], bip_mem_vectors[i])

zd_wt_mat -= q * np.eye(n)  # Zero diag

# Input probe vector
probe = input('Enter the probe vector: ')
probe = np.array(list(map(int, probe)))

# Signal vector
signal_vector = 2 * probe - 1
signal_vector = signal_vector.reshape(1, -1)  # Reshape to (1, n)

flag = 0  # Initialize flag
while flag != n:
    permindex = np.random.permutation(n)  # Randomize order
    old_signal_vector = np.copy(signal_vector)

    for j in range(n):
        act_vec = np.dot(signal_vector, zd_wt_mat)
        if act_vec[0, permindex[j]] > 0:
            signal_vector[0, permindex[j]] = 1
        elif act_vec[0, permindex[j]] < 0:
            signal_vector[0, permindex[j]] = -1

    flag = np.dot(signal_vector, old_signal_vector.T)

print('The recalled vector is:')
print(0.5 * (signal_vector + 1))
