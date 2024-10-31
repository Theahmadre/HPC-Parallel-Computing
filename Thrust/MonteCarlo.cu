#include <iostream>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <chrono>

using namespace std;

struct is_inside_circle {
    __device__ bool operator()(const float2& p) {
        return (p.x * p.x + p.y * p.y <= 1.0f);
    }
};

int main() {
    const long long numSamples = 1000000;
    thrust::device_vector<float2> points(numSamples);
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto start = chrono::high_resolution_clock::now();

    for (long long i = 0; i < numSamples; ++i) {
        points[i] = make_float2(dist(rng), dist(rng));
    }

    long long insideCircleCount = thrust::count_if(points.begin(), points.end(), is_inside_circle());
    double piEstimate = (4.0 * insideCircleCount) / numSamples;

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Estimated value of π: " << piEstimate << endl;
    cout << "Time taken: " << elapsed.count() << " seconds" << endl;

    return 0;
}
