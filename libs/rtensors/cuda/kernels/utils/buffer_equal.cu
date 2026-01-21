#include "../../include/utils.h"

/*
    KERNEL
*/

template <typename T>
__global__ void buffer_equal_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    bool* __restrict__ result,
    size_t n
) {
    // Each block checks if there's any inequality
    __shared__ bool block_equal;
    
    if (threadIdx.x == 0) {
        block_equal = true;
    }
    __syncthreads();
    
    // Grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        
        if (a[i] != b[i]) {
            block_equal = false;
        }
    }
    
    __syncthreads();
    
    // First thread in each block writes the result
    if (threadIdx.x == 0) {
        if (!block_equal) {
            *result = false;
        }
    }
}

/*
    LAUNCHER
*/

template <typename T>
void launch_buffer_equal(
    const T* a,
    const T* b,
    bool* result,
    size_t n,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);
    
    const unsigned int grid = std::min((unsigned int)((n + block_size - 1) / block_size), 65535u);
    
    // Initialize result to true on device before kernel launch
    cudaMemset(result, 1, sizeof(bool));
    
    buffer_equal_kernel<T><<<grid, block_size>>>(a, b, result, n);
}

/*
    EXTERN C DECLARATIONS FOR ALL TYPES
*/

#define DECLARE_BUFFER_EQUAL_LAUNCHER(TYPE, SUFFIX) \
    extern "C" void launch_buffer_equal_##SUFFIX( \
        const TYPE* a, \
        const TYPE* b, \
        bool* result, \
        size_t n, \
        unsigned int block_size \
    ) { \
        launch_buffer_equal<TYPE>(a, b, result, n, block_size); \
    }

// Declare for all types
DECLARE_BUFFER_EQUAL_LAUNCHER(float, f32)
DECLARE_BUFFER_EQUAL_LAUNCHER(double, f64)
DECLARE_BUFFER_EQUAL_LAUNCHER(uint8_t, u8)
DECLARE_BUFFER_EQUAL_LAUNCHER(uint16_t, u16)
DECLARE_BUFFER_EQUAL_LAUNCHER(uint32_t, u32)
DECLARE_BUFFER_EQUAL_LAUNCHER(uint64_t, u64)
DECLARE_BUFFER_EQUAL_LAUNCHER(__uint128_t, u128)
DECLARE_BUFFER_EQUAL_LAUNCHER(int8_t, i8)
DECLARE_BUFFER_EQUAL_LAUNCHER(int16_t, i16)
DECLARE_BUFFER_EQUAL_LAUNCHER(int32_t, i32)
DECLARE_BUFFER_EQUAL_LAUNCHER(int64_t, i64)
DECLARE_BUFFER_EQUAL_LAUNCHER(__int128_t, i128)
DECLARE_BUFFER_EQUAL_LAUNCHER(bool, boolean)
