#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void kernel(int** c) {
    int index = threadIdx.x;
    printf("Thread | val: %i\n", index * 10);
    c[index][0] = index * 10;  // Access the int pointed to by c[index]
}

int main() {
    int length = 10;

    // Allocate host memory
    int* h_c = new int[length];
    
    // Allocate device memory for array of pointers
    int** dev_c;
    cudaMalloc((void**)&dev_c, sizeof(int*) * length);

    // Allocate device memory for each integer and copy pointers to them
    int* temp;
    for (int i = 0; i < length; i++) {
        cudaMalloc((void**)&temp, sizeof(int));
        cudaMemcpy(&dev_c[i], &temp, sizeof(int*), cudaMemcpyHostToDevice);
    }

    cout << "Launching kernel..." << endl;

    // Launch kernel
    kernel<<<1, length>>>(dev_c);
    cudaDeviceSynchronize();

    cout << "Kernel result:" << endl;

    // Copy results from device to host
    cudaMemcpy(h_c, dev_c, sizeof(int*) * length, cudaMemcpyDeviceToHost);

    // Print results
    cout << "Result: ";
    for (int i = 0; i < length; i++) {
      cout << h_c[i] << " ";
    }

    // Clean up
    for (int i = 0; i < length; i++) {
        cudaFree(dev_c[i]);  // Free each allocated int
    }
    cudaFree(dev_c);  // Free the array of pointers
    delete[] h_c;  // Free host memory

    return 0;
}