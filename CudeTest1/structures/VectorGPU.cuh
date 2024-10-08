
namespace gpu_std {
  #ifndef VECTOR_GPU_H // include guard
  #define VECTOR_GPU_H
  /**
   * @file gpu_vector.cuh
   * @brief A vector class that uses GPU for certain functionality points
   * NOTE This class can be passed directly as a Kernel argument, because we have properly
   * allocated necessary memory on GPU and ONLY use the class itself as a reference to the GPU memory
   */
  template <typename T>
  class VectorGPU {
    public:
      /**
       * @brief Construct a new gpu_vector object
       */
      VectorGPU();

      /**
       * @brief Deconstruct vector -> free on GPU/CPU
       */
      ~VectorGPU();

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ bool operator=(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ bool operator=(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ VectorGPU* operator+(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ void operator+(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ void operator-(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ void operator-(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ void operator*(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ void operator*(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ void operator/(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ void operator/(const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😔
       */
      __host__ T operator[](const gpu_vector<T>& other);

      /**
       * @brief Copy constructor
       * This will NOT implement the GPU 😎
       */
      __device__ T operator[](const gpu_vector<T>& other);

      // HOST FUNCTIONS //

      /**
       * @brief Get the size of the vector
       * @return size of the vector
       */
      __host__ int size();

      /**
       * @brief Add element to back of vector
       * @param added element to add
       * @return true if added, false if not
       */
      __host__ boolean add(T added);

      /**
       * @brief Add element to specific index
       * @param added element to add
       * @param index index to add element
       * @return true if added, false if not
       */
      __host__ boolean add(T added, int index);

      /**
       * @brief Remove first occurrance of a specific element
       * @param removed element to remove
       * @return removed element
       */
      __host__ T remove_first(T removed);

      /**
       * @brief return element at an index
       * @param index index to remove element
       * @return removed element
       */
      __host__ T remove(int index);

      /**
       * @brief Remove all occurrances of a specific element
       * @param element element to remove
       * @return true if all removed, false if not
       */
      __host__ boolean removeAll(T removed);

      // DEVICE FUNCTIONS //
      /**
       * @brief Compare to another vector
       * This method works very uniquely:
       * given a pointer to memory in GPU, and assuming that our current dev_content pointer is working properly,
       * we want to compare the two vectors. We will use the dev_content pointer to compare the on-GPU.
       */
      __device__ int compareTo(int** dev_content);

      // SUPPLEMENTARY FUNCTIONS BC WE R SO COOL //
      __device__ int dot_product(int** dev_content);
      __device__ int cross_product(int** dev_content);
      __device__ int magnitude(int** dev_content);
      __device__ double angle(int** dev_content);
      __device__ int distance(int** dev_content);
      __device__ int projection(int** dev_content);
    private:
      // HOST VARIABLES //
      int size;
      T* content;
      bool updated;

      // DEVICE VARIABLES //
      int* dev_size;
      T** dev_content;

      /**
       * @brief reallocate the dev_content pointer on the gpu (i.e. vec obj has been updated)
       * works in association with updated preferably
       */
      __host__ boolean reallocate();

      /**
       * @brief set the updated flag
       * @param updated updated flag
       */
      __host__ void set_updated(bool updated) {
        this->updated = updated;
      }
  };
  #endif
}