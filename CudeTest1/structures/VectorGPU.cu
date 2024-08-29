
#include "VectorGPU.cuh"
// #include <cuda_runtime.h>
using namespace gpu_std;

// TODO define method headers (i.e. VectorGPU<T>::method_name, use generic template)

template <typename T>
VectorGPU<T>::VectorGPU() {
  // TODO implement constructor
}

template <typename T>
VectorGPU<T>::~VectorGPU() {
  // TODO implement destructor
}

// This is where stuff starts to get buggy, we need to make sure this actually works LOL
template <typename T>
__host__
boolean VectorGPU<T>::operator=(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
boolean VectorGPU<T>::operator=(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

template <typename T>
__host__
void VectorGPU<T>::operator+(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
void VectorGPU<T>::operator+(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

template <typename T>
__host__
void VectorGPU<T>::operator-(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
void VectorGPU<T>::operator-(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

template <typename T>
__host__
void VectorGPU<T>::operator*(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
void VectorGPU<T>::operator*(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

template <typename T>
__host__
void VectorGPU<T>::operator/(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
void VectorGPU<T>::operator/(const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

template <typename T>
__host__
T VectorGPU<T>::operator[](const VectorGPU<T>& other) {
  // TODO implement copy constructor
}

// same for device
template <typename T>
__device__
T VectorGPU<T>::operator[](const VectorGPU<T>& other) {
  // TODO implement copy constructor
}