#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scalar binary operations - Add
#define DECLARE_SCALAR_OP_HEADERS(op, TYPE, SUFFIX) \
    void launch_##op##_contiguous_##SUFFIX(TYPE* data, size_t n, TYPE value, unsigned int block_size); \
    void launch_##op##_strided_##SUFFIX(TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size); \
    void launch_##op##_nd_affine_##SUFFIX(TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, TYPE value, unsigned int block_size);

// Add operation - all types
DECLARE_SCALAR_OP_HEADERS(add, float, f32)
DECLARE_SCALAR_OP_HEADERS(add, double, f64)
DECLARE_SCALAR_OP_HEADERS(add, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(add, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(add, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(add, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(add, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(add, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(add, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(add, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(add, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(add, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(add, bool, boolean)

// Sub operation - all types
DECLARE_SCALAR_OP_HEADERS(sub, float, f32)
DECLARE_SCALAR_OP_HEADERS(sub, double, f64)
DECLARE_SCALAR_OP_HEADERS(sub, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(sub, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(sub, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(sub, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(sub, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(sub, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(sub, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(sub, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(sub, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(sub, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(sub, bool, boolean)

// Mul operation - all types
DECLARE_SCALAR_OP_HEADERS(mul, float, f32)
DECLARE_SCALAR_OP_HEADERS(mul, double, f64)
DECLARE_SCALAR_OP_HEADERS(mul, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(mul, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(mul, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(mul, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(mul, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(mul, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(mul, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(mul, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(mul, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(mul, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(mul, bool, boolean)

// Div operation - all types
DECLARE_SCALAR_OP_HEADERS(div, float, f32)
DECLARE_SCALAR_OP_HEADERS(div, double, f64)
DECLARE_SCALAR_OP_HEADERS(div, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(div, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(div, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(div, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(div, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(div, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(div, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(div, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(div, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(div, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(div, bool, boolean)

DECLARE_SCALAR_OP_HEADERS(log, float, f32)
DECLARE_SCALAR_OP_HEADERS(log, double, f64)

DECLARE_SCALAR_OP_HEADERS(log1p, float, f32)
DECLARE_SCALAR_OP_HEADERS(log1p, double, f64)

DECLARE_SCALAR_OP_HEADERS(elu, float, f32)
DECLARE_SCALAR_OP_HEADERS(elu, double, f64)

// Leaky ReLU operation - all types
DECLARE_SCALAR_OP_HEADERS(leaky_relu, float, f32)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, double, f64)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, uint8_t, u8)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, uint16_t, u16)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, uint32_t, u32)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, uint64_t, u64)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, __uint128_t, u128)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, int8_t, i8)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, int16_t, i16)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, int32_t, i32)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, int64_t, i64)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, __int128_t, i128)
DECLARE_SCALAR_OP_HEADERS(leaky_relu, bool, boolean)

// Negation operations
void launch_negate_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i8(int8_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i16(int16_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i32(int32_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i64(int64_t* data, size_t start, size_t len, unsigned int block_size);
void launch_negate_contiguous_i128(__int128_t* data, size_t start, size_t len, unsigned int block_size);


void launch_relu_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i8(int8_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i16(int16_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i32(int32_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i64(int64_t* data, size_t start, size_t len, unsigned int block_size);
void launch_relu_contiguous_i128(__int128_t* data, size_t start, size_t len, unsigned int block_size);

void launch_relu_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i8(int8_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i16(int16_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i32(int32_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i64(int64_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_relu_strided_i128(__int128_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);


void launch_relu_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i8(int8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i16(int16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i32(int32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i64(int64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_relu_nd_affine_i128(__int128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);


void launch_negate_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i8(int8_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i16(int16_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i32(int32_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i64(int64_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_negate_strided_i128(__int128_t* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_negate_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i8(int8_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i16(int16_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i32(int32_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i64(int64_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_negate_nd_affine_i128(__int128_t* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);



void launch_sigmoid_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_sigmoid_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);




void launch_sigmoid_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_sigmoid_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_sigmoid_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_sigmoid_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);

void launch_tanh_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size);
void launch_tanh_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size);

void launch_tanh_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);
void launch_tanh_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size);

void launch_tanh_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);
void launch_tanh_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);

#define DECLARE_UNARY_FLOAT_OP_HEADERS(op)                                                     \
    void launch_##op##_contiguous_f32(float* data, size_t start, size_t len, unsigned int block_size); \
    void launch_##op##_contiguous_f64(double* data, size_t start, size_t len, unsigned int block_size); \
                                                                                                \
    void launch_##op##_strided_f32(float* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size); \
    void launch_##op##_strided_f64(double* data, size_t offset, ptrdiff_t stride, size_t len, unsigned int block_size); \
                                                                                                \
    void launch_##op##_nd_affine_f32(float* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size); \
    void launch_##op##_nd_affine_f64(double* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, unsigned int block_size);


DECLARE_UNARY_FLOAT_OP_HEADERS(abs)
DECLARE_UNARY_FLOAT_OP_HEADERS(sqrt)
DECLARE_UNARY_FLOAT_OP_HEADERS(sin)
DECLARE_UNARY_FLOAT_OP_HEADERS(cos)
DECLARE_UNARY_FLOAT_OP_HEADERS(tan)
DECLARE_UNARY_FLOAT_OP_HEADERS(asin)
DECLARE_UNARY_FLOAT_OP_HEADERS(acos)
DECLARE_UNARY_FLOAT_OP_HEADERS(atan)
DECLARE_UNARY_FLOAT_OP_HEADERS(sinh)
DECLARE_UNARY_FLOAT_OP_HEADERS(cosh)
// DECLARE_UNARY_FLOAT_OP_HEADERS(tan)
DECLARE_UNARY_FLOAT_OP_HEADERS(asinh)
DECLARE_UNARY_FLOAT_OP_HEADERS(acosh)
DECLARE_UNARY_FLOAT_OP_HEADERS(atanh)
DECLARE_UNARY_FLOAT_OP_HEADERS(rsqrt)
DECLARE_UNARY_FLOAT_OP_HEADERS(reciprocal)
DECLARE_UNARY_FLOAT_OP_HEADERS(square)
DECLARE_UNARY_FLOAT_OP_HEADERS(cube)
DECLARE_UNARY_FLOAT_OP_HEADERS(exp)
DECLARE_UNARY_FLOAT_OP_HEADERS(sign)
DECLARE_UNARY_FLOAT_OP_HEADERS(ln)
DECLARE_UNARY_FLOAT_OP_HEADERS(ln1p)
DECLARE_UNARY_FLOAT_OP_HEADERS(floor)
DECLARE_UNARY_FLOAT_OP_HEADERS(ceil)
DECLARE_UNARY_FLOAT_OP_HEADERS(round)
DECLARE_UNARY_FLOAT_OP_HEADERS(trunc)
DECLARE_UNARY_FLOAT_OP_HEADERS(expm1)
DECLARE_UNARY_FLOAT_OP_HEADERS(silu)

// Reduction operations
#define DECLARE_REDUCTION_OP_HEADERS(TYPE, SUFFIX)                                                \
    void launch_flat_contiguous_reduce_##SUFFIX(                                                  \
        const TYPE *data, TYPE *out, size_t start, size_t len,                                    \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);        \
    void launch_nd_reduce_contiguous_##SUFFIX(                                                    \
        TYPE *data, TYPE *out, size_t offset, size_t outer, size_t r, size_t inner,              \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);

// Float types
DECLARE_REDUCTION_OP_HEADERS(float,  f32)
DECLARE_REDUCTION_OP_HEADERS(double, f64)

// Returns 1 (and clears the flag) if a reduction launcher on this thread skipped its work
// because a scratch allocation failed.
int rtensors_take_cuda_alloc_error(void);


// Reduction operations
#define DECLARE_ARGMAX_OP_HEADERS(TYPE, SUFFIX)                                                \
    void launch_flat_contiguous_argmax_##SUFFIX(                                                  \
        const TYPE *data, uint64_t *out, size_t start, size_t len,                                    \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);        \
    void launch_nd_argmax_contiguous_##SUFFIX(                                                    \
        TYPE *data, uint64_t *out, size_t offset, size_t outer, size_t r, size_t inner,              \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size);

// Float types
DECLARE_ARGMAX_OP_HEADERS(float,  f32)
DECLARE_ARGMAX_OP_HEADERS(double, f64)



#ifdef __cplusplus
}
#endif
