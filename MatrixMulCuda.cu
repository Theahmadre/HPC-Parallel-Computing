#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;

#define N 8192

__global__ void matrixMulKernel(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

int main() {
    vector<int> h_A(N * N, 1);
    vector<int> h_B(N * N, 1);
    vector<int> h_C(N * N, 0);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    ofstream results("performance.csv");
    results << "Threads,Time\n";

    int configurations[] = {1, 2, 4, 8, 16, 32, 64, 128};
    for (int config : configurations) {
        dim3 threadsPerBlock(config, config);
        dim3 blocksPerGrid((N + config - 1) / config, (N + config - 1) / config);

        auto start = chrono::high_resolution_clock::now();

        matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        int numThreads = config * config;
        results << numThreads << "," << elapsed.count() << endl;

        cout << "Threads: " << numThreads << ", Time: " << elapsed.count() << " seconds" << endl;
    }

    results.close();

    cudaMemcpy(h_C.data(), d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

