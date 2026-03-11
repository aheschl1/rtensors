#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fill operations
#define DECLARE_FILL_HEADERS(TYPE, SUFFIX) \
    void launch_fill_contiguous_##SUFFIX(TYPE* data, size_t n, TYPE value, unsigned int block_size); \
    void launch_fill_strided_##SUFFIX(TYPE* data, size_t start, ptrdiff_t stride, size_t len, TYPE value, unsigned int block_size); \
    void launch_fill_nd_affine_##SUFFIX(TYPE* data, size_t offset, const ptrdiff_t* stride, const size_t* shape, size_t rank, size_t size, TYPE value, unsigned int block_size);

// Fill operation - all types
DECLARE_FILL_HEADERS(float, f32)
DECLARE_FILL_HEADERS(double, f64)
DECLARE_FILL_HEADERS(uint8_t, u8)
DECLARE_FILL_HEADERS(uint16_t, u16)
DECLARE_FILL_HEADERS(uint32_t, u32)
DECLARE_FILL_HEADERS(uint64_t, u64)
DECLARE_FILL_HEADERS(__uint128_t, u128)
DECLARE_FILL_HEADERS(int8_t, i8)
DECLARE_FILL_HEADERS(int16_t, i16)
DECLARE_FILL_HEADERS(int32_t, i32)
DECLARE_FILL_HEADERS(int64_t, i64)
DECLARE_FILL_HEADERS(__int128_t, i128)
DECLARE_FILL_HEADERS(bool, boolean)

#ifdef __cplusplus
}
#endif
