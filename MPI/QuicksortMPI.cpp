#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <chrono>

using namespace std;

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quicksort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100000;
    vector<int> arr;

    if (rank == 0) {
        arr.resize(N);
        for (int& val : arr) {
            val = rand() % 1000;
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = N / size;
    vector<int> local_arr(local_size);

    MPI_Scatter(arr.data(), local_size, MPI_INT, local_arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    auto start_time = chrono::high_resolution_clock::now();

    quicksort(local_arr, 0, local_size - 1);

    MPI_Gather(local_arr.data(), local_size, MPI_INT, arr.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        vector<int> sorted_arr;
        sorted_arr.reserve(N);

        for (int i = 0; i < size; ++i) {
            vector<int> sub_arr(arr.begin() + i * local_size, arr.begin() + (i + 1) * local_size);
            quicksort(sub_arr, 0, local_size - 1);
            sorted_arr.insert(sorted_arr.end(), sub_arr.begin(), sub_arr.end());
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_time = end_time - start_time;

        cout << "Time taken for sorting: " << elapsed_time.count() << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
