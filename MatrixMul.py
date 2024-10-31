import time
import numpy as np

N = 250
A = np.ones((N, N), dtype=int)
B = np.ones((N, N), dtype=int)
C = np.zeros((N, N), dtype=int)

start_time = time.time()

for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]

end_time = time.time()

print(f"Time taken for matrix multiplication in Python: {end_time - start_time:.6f} seconds")
