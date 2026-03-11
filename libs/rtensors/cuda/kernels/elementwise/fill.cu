#include "../../include/fill.h"

/*
    KERNELS
*/

template <typename T>
__global__ void fill_contiguous_kernel(
    T* __restrict__ data,
    size_t n,
    T value
) {
    // grid-stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {

        data[i] = value;
    }
}

template <typename T>
__global__ void fill_strided_kernel(
    T* __restrict__ data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    T value
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < len;
         i += blockDim.x * gridDim.x) {

        size_t idx = (size_t)((ptrdiff_t)start + (ptrdiff_t)i * stride);
        data[idx] = value;
    }
}

template <typename T>
__global__ void fill_nd_affine_kernel(
    T* __restrict__ data,
    size_t offset,
    const ptrdiff_t* __restrict__ stride,
    const size_t* __restrict__ shape,
    size_t rank,
    size_t size,
    T value
) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x
    ) {
        size_t linear = idx;
        ptrdiff_t phys = (ptrdiff_t)offset;

        for (int dim = (int)rank - 1; dim >= 0; --dim) {
            size_t coord = linear % shape[dim];
            linear /= shape[dim];
            phys += (ptrdiff_t)coord * stride[dim];
        }

        size_t final_idx = (size_t)phys; 

        data[final_idx] = value;
    }
}

/*
    LAUNCHERS
*/

template <typename T>
void launch_fill_contiguous_impl(
    T* data,
    size_t n,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((n + block_size - 1) / block_size), 65535u);
    fill_contiguous_kernel<T><<<grid, block_size>>>(data, n, value);
}

template <typename T>
void launch_fill_strided_impl(
    T* data,
    size_t start,
    ptrdiff_t stride,
    size_t len,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((len + block_size - 1) / block_size), 65535u);
    fill_strided_kernel<T><<<grid, block_size>>>(data, start, stride, len, value);
}

template <typename T>
void launch_fill_nd_affine_impl(
    T* data,
    size_t offset,
    const ptrdiff_t* stride,
    const size_t* shape,
    size_t rank,
    size_t size,
    T value,
    unsigned int block_size
) {
    block_size = ALIGN_BLOCK_SIZE(block_size);

    const unsigned int grid = std::min((unsigned int)((size + block_size - 1) / block_size), 65535u);
    fill_nd_affine_kernel<T><<<grid, block_size>>>(data, offset, stride, shape, rank, size, value);
}

#define DECLARE_FILL_LAUNCHERS(TYPE, SUFFIX) \
    extern "C" void launch_fill_contiguous_##SUFFIX( \
        TYPE* data, size_t n, TYPE value, unsigned int block_size \
    ) { \
        launch_fill_contiguous_impl<TYPE>(data, n, value, block_size); \
    } \
    \
    extern "C" void launch_fill_strided_##SUFFIX( \
        TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size \
    ) { \
        launch_fill_strided_impl<TYPE>(data, start, stride, len, value, block_size); \
    } \
    \
    extern "C" void launch_fill_nd_affine_##SUFFIX( \
        TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, \
        size_t rank, size_t size, TYPE value, unsigned int block_size \
    ) { \
        launch_fill_nd_affine_impl<TYPE>(data, offset, stride, shape, rank, size, value, block_size); \
    }

// Declare launchers for all types
DECLARE_FILL_LAUNCHERS(float,  f32)
DECLARE_FILL_LAUNCHERS(double, f64)
DECLARE_FILL_LAUNCHERS(uint8_t,  u8)
DECLARE_FILL_LAUNCHERS(uint16_t, u16)
DECLARE_FILL_LAUNCHERS(uint32_t, u32)
DECLARE_FILL_LAUNCHERS(uint64_t, u64)
DECLARE_FILL_LAUNCHERS(__uint128_t, u128)
DECLARE_FILL_LAUNCHERS(int8_t,  i8)
DECLARE_FILL_LAUNCHERS(int16_t, i16)
DECLARE_FILL_LAUNCHERS(int32_t, i32)
DECLARE_FILL_LAUNCHERS(int64_t, i64)
DECLARE_FILL_LAUNCHERS(__int128_t, i128)
DECLARE_FILL_LAUNCHERS(bool, boolean)
