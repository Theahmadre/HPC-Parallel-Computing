#include <omp.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace std;

typedef complex<double> Complex;

void fft(vector<Complex>& x) {
    int N = x.size();
    if (N <= 1) return;

    vector<Complex> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    #pragma omp parallel
    {
        #pragma omp single
        {
            fft(even);
            fft(odd);
        }
    }

    for (int i = 0; i < N / 2; ++i) {
        double theta = -2 * M_PI * i / N;
        Complex t = polar(1.0, theta) * odd[i];
        x[i] = even[i] + t;
        x[i + N / 2] = even[i] - t;
    }
}

int main() {
    const int N = 1024;
    vector<Complex> x(N);
    for (int i = 0; i < N; ++i) {
        x[i] = Complex(i, 0);
    }

    fft(x);

    for (const auto& val : x) {
        cout << val << endl;
    }

    return 0;
}
