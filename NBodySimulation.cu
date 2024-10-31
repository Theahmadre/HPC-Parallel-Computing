#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

using namespace std;

#define N 2097152
#define G 6.67430e-11f
#define TIME_STEP 0.01f
#define NUM_STEPS 500

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

__device__ void calculateForce(Particle &p1, Particle &p2, float3 &force) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float dz = p2.z - p1.z;
    float distSqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    float distSixth = distSqr * distSqr * distSqr;
    float f = G * p1.mass * p2.mass / sqrt(distSixth);
    force.x += f * dx;
    force.y += f * dy;
    force.z += f * dz;
}

__global__ void nBodyKernel(Particle *particles, Particle *newParticles, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Particle p = particles[i];
    float3 totalForce = {0.0f, 0.0f, 0.0f};

    for (int j = 0; j < n; j++) {
        if (i != j) {
            calculateForce(p, particles[j], totalForce);
        }
    }

    p.vx += totalForce.x / p.mass * TIME_STEP;
    p.vy += totalForce.y / p.mass * TIME_STEP;
    p.vz += totalForce.z / p.mass * TIME_STEP;

    p.x += p.vx * TIME_STEP;
    p.y += p.vy * TIME_STEP;
    p.z += p.vz * TIME_STEP;

    newParticles[i] = p;
}

int main() {
    vector<Particle> h_particles(N);
    vector<Particle> h_newParticles(N);

    for (int i = 0; i < N; i++) {
        h_particles[i].x = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        h_particles[i].y = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        h_particles[i].z = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        h_particles[i].vx = h_particles[i].vy = h_particles[i].vz = 0.0f;
        h_particles[i].mass = static_cast<float>(rand()) / RAND_MAX * 10.0f + 1.0f;
    }

    Particle *d_particles, *d_newParticles;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMalloc(&d_newParticles, N * sizeof(Particle));
    cudaMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);

    ofstream results("nbody_performance.csv");
    results << "Threads,Time\n";

    int configurations[] = {64, 128, 256};

    for (int config : configurations) {
        dim3 threadsPerBlock(config);
        dim3 blocksPerGrid((N + config - 1) / config);

        auto start = chrono::high_resolution_clock::now();

        for (int step = 0; step < NUM_STEPS; ++step) {
            nBodyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_newParticles, N);
            cudaDeviceSynchronize();
            swap(d_particles, d_newParticles);
        }

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;

        results << config << "," << elapsed.count() << endl;
        cout << "Threads: " << config << ", Time: " << elapsed.count() << " seconds" << endl;
    }

    results.close();

    cudaMemcpy(h_newParticles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaFree(d_particles);
    cudaFree(d_newParticles);

    return 0;
}
