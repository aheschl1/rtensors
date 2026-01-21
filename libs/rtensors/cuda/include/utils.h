#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Buffer equality check - returns true if all elements are equal, false otherwise
#define DECLARE_BUFFER_EQUAL_HEADER(TYPE, SUFFIX) \
    void launch_buffer_equal_##SUFFIX(const TYPE* a, const TYPE* b, bool* result, size_t n, unsigned int block_size);

// Declare for all types
DECLARE_BUFFER_EQUAL_HEADER(float, f32)
DECLARE_BUFFER_EQUAL_HEADER(double, f64)
DECLARE_BUFFER_EQUAL_HEADER(uint8_t, u8)
DECLARE_BUFFER_EQUAL_HEADER(uint16_t, u16)
DECLARE_BUFFER_EQUAL_HEADER(uint32_t, u32)
DECLARE_BUFFER_EQUAL_HEADER(uint64_t, u64)
DECLARE_BUFFER_EQUAL_HEADER(__uint128_t, u128)
DECLARE_BUFFER_EQUAL_HEADER(int8_t, i8)
DECLARE_BUFFER_EQUAL_HEADER(int16_t, i16)
DECLARE_BUFFER_EQUAL_HEADER(int32_t, i32)
DECLARE_BUFFER_EQUAL_HEADER(int64_t, i64)
DECLARE_BUFFER_EQUAL_HEADER(__int128_t, i128)
DECLARE_BUFFER_EQUAL_HEADER(bool, boolean)

#ifdef __cplusplus
}
#endif
