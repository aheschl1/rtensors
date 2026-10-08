#include <cuda_runtime.h>

#include <stdio.h>
#include "../../include/common.h"
#include <limits.h>
#include <cub/device/device_reduce.cuh>
// #include <cub/iterator/transform_input_iterator.cuh>

/// @brief The summation functor.
struct SumOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct LessOp
{
    template <typename T>
    __device__ __forceinline__ bool
    operator()(const T &a, const T &b) const
    {
        return a > b;
    }
};

struct GreaterOp
{
    template <typename T>
    __device__ __forceinline__ bool
    operator()(const T &a, const T &b) const
    {
        return a < b;
    }
};

struct L1SumOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a + abs(b);
    }
};

struct SquareUnaryOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a) const
    {
        return a * a;
    }
};

struct L2SumOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a + (b * b);
    }
};

struct ProdOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};

struct MaxOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return (a > b) ? a : b;
    }
};

struct MinOp
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(const T &a, const T &b) const
    {
        return (a < b) ? a : b;
    }
};

struct PostNothing
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(T out, size_t n) const
    {
        return out;
    }
};

struct PostSqrt
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(T out, size_t n) const
    {
        return sqrt(out);
    }
};

struct PostDivTotal
{
    template <typename T>
    __device__ __forceinline__
        T
        operator()(T out, size_t n) const
    {

        return out /= (T)n;
    }
};

template <typename T>
struct WelfordState
{
    T mean;
    T m2;
    int count;
};

template <typename T>
__host__ __device__ __forceinline__ WelfordState<T> welford_init()
{
    return WelfordState<T>{(T)0, (T)0, 0};
}

template <typename T>
__host__ __device__ __forceinline__ void welford_update(WelfordState<T> *state, T x)
{
    state->count += 1; // increment the counter.

    T delta = x - state->mean;
    state->mean += delta / (T)state->count;

    T delta2 = x - state->mean;
    state->m2 += delta * delta2;
}

template <typename T>
__host__ __device__ __forceinline__ WelfordState<T> welford_combine(WelfordState<T> a, WelfordState<T> b)
{
    if (a.count == 0)
    {
        return b;
    }
    if (b.count == 0)
    {
        return a;
    }

    WelfordState<T> result;

    T delta = b.mean - a.mean;
    int n = a.count + b.count;

    result.mean = a.mean + delta * (T(b.count) / T(n));
    result.m2 = a.m2 + b.m2 + (delta * delta) * ((T(a.count) * T(b.count)) / T(n));
    result.count = n;

    return result;
}

template <typename T, typename PostOp>
__global__ void post_transform_kernel(T *out, size_t n, PostOp post)
{

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // printf("HI i = %d, out[%d] = %g\n", i, i, out[i]);
        out[i] = post(out[i], n);
    }
}

/// @brief Designed to run a reduction operation on a contiguous array where we would like to reduce
/// everything to a single element.
/// @tparam T the datatype to use
/// @tparam Op the functor
/// @param data the pointer to the input data
/// @param d_out the poitner to the result buffer
/// @param start the start of where we should read
/// @param num_items how big the buffer is
/// @param block_size the block size
/// @param op the operator, i.e., the actual functor.
/// @param init the base case for reduction.
template <typename T, typename Op, typename PostOp>
void launch_flat_contiguous_reduce(
    const T *data,
    T *d_out,
    size_t start,
    size_t num_items,
    unsigned int block_size,
    Op op,
    T init,
    PostOp post)
{
    // Declare, allocate, and initialize device-accessible pointers for
    // input and output
    const T *d_in = &data[start];

    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, op, init);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out, num_items, op, init);

    // Free the temporary storage allocation.
    cudaFree(d_temp_storage);

    post_transform_kernel<<<1, 1>>>(d_out, num_items, post);
}

template <typename T>
__global__ void square_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        T x = in[i];
        out[i] = x * x;
    }
}

template <typename T>
void launch_flat_contiguous_l2norm(
    const T *data,
    T *d_out,
    size_t start,
    size_t num_items,
    unsigned int block_size)
{
    // 1) allocate temp for squares
    T *d_tmp = nullptr;
    cudaMalloc(&d_tmp, num_items * sizeof(T));

    // 2) map: tmp[i] = data[start+i]^2
    const T *d_in = data + start;
    int threads = 256;
    int blocks = (int)((num_items + threads - 1) / threads);
    square_kernel<<<blocks, threads>>>(d_in, d_tmp, num_items);

    // 3) reduce sum(tmp) + sqrt
    launch_flat_contiguous_reduce<T, SumOp, PostSqrt>(
        d_tmp,
        d_out,
        /*start=*/0,
        num_items,
        block_size,
        SumOp{},
        (T)0,
        PostSqrt{});

    cudaFree(d_tmp);
}

template <typename T>
struct ToWelfordState
{
    __host__ __device__ __forceinline__
        WelfordState<T>
        operator()(T x) const
    {
        return WelfordState<T>{x, T(0), 1};
    }
};

template <typename T>
struct WelfordCombine
{
    __host__ __device__ __forceinline__
        WelfordState<T>
        operator()(WelfordState<T> a, WelfordState<T> b) const
    {
        return welford_combine(a, b);
    }
};

template <typename T>
__global__ void finalize_var_kernel(
    T *out_var,
    WelfordState<T> state,
    bool unbiased)
{
    if (threadIdx.x == 0)
    {
        int n = state.count;
        if (n <= 1)
        {
            out_var[0] = T(0);
            return;
        }
        T denom = unbiased ? (T)(n - 1) : (T)n;
        out_var[0] = state.m2 / denom;
    }
}

template <typename T>
__host__ __device__ __forceinline__ WelfordState<T> welford_from_value(T x)
{
    return WelfordState<T>{x, T(0), 1};
}

template <typename T>
__global__ void to_welford_state_kernel(const T *__restrict__ in,
                                        WelfordState<T> *__restrict__ out,
                                        size_t n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = welford_from_value<T>(in[idx]);
}

template <typename T>
__global__ void finalize_var_kernel_ptr(
    T *out_var,
    const WelfordState<T> *state,
    bool unbiased,
    bool should_sqrt)
{
    if (threadIdx.x == 0)
    {
        int n = state->count;
        if (n <= 1)
        {
            out_var[0] = T(0);
            return;
        }
        T denom = unbiased ? (T)(n - 1) : (T)n;
        out_var[0] = state->m2 / denom;
        if (should_sqrt)
        {
            out_var[0] = sqrt(out_var[0]);
        }
    }
}

template <typename T>
void launch_flat_contiguous_reduce_variance(
    const T *data,
    T *d_out,
    size_t start,
    size_t num_items,
    const ReductionSettings *settings,
    unsigned int block_size)
{

    using State = WelfordState<T>;

    State *d_states = nullptr;
    cudaMalloc(&d_states, num_items * sizeof(State));

    int threads = 256;
    int blocks = (int)((num_items + threads - 1) / threads);
    to_welford_state_kernel<<<blocks, threads>>>(&data[start], d_states, num_items);

    // Allocate the output, we only need one.
    State *d_states_out = nullptr;
    cudaMalloc(&d_states_out, sizeof(State));

    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        nullptr, temp_storage_bytes,
        d_states, d_states_out, num_items,
        WelfordCombine<T>{}, welford_init<T>());

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes,
        d_states, d_states_out, num_items, WelfordCombine<T>{}, welford_init<T>());

    finalize_var_kernel_ptr<<<1, 1>>>(d_out, d_states_out, settings->unbiased, settings->is_std);

    // Free the temporary storage allocation.
    cudaFree(d_temp_storage);
    cudaFree(d_states_out);
    cudaFree(d_states);

    // post_transform_kernel<<<1, block_size>>>(d_out, num_items, post);
}

template <typename T, typename Op, typename PostOp>
__global__ void fold_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    Op op,
    T init,
    PostOp post)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial sum over k = threadIdx.x, threadIdx.x+blockDim.x, ...
    T thread_sum = init;
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        thread_sum = op(thread_sum, base[k * inner]);
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ T smem[256]; // assumes blockDim.x <= 256; adjust if you want bigger
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = op(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[out_linear] = post((T)smem[0], R);
    }
}

template <typename T, typename Op, typename PostOp, typename MapOp>
__global__ void map_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    MapOp map,
    Op op,
    T init,
    PostOp post)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial sum over k = threadIdx.x, threadIdx.x+blockDim.x, ...
    T thread_sum = init;
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        thread_sum = op(thread_sum, map(base[k * inner]));
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ T smem[256]; // assumes blockDim.x <= 256; adjust if you want bigger
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = op(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[out_linear] = post((T)smem[0], R);
    }
}

template <typename T, typename PostOp>
__global__ void var_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    PostOp post,
    bool unbiased,
    bool is_std)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial sum over k = threadIdx.x, threadIdx.x+blockDim.x, ...
    WelfordState<T> state = welford_init<T>();
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        welford_update(&state, base[k * inner]);
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ WelfordState<T> smem[256]; // assumes blockDim.x <= 256; adjust if you want bigger
    smem[threadIdx.x] = state;
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = welford_combine(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        WelfordState<T> reso = smem[0];

        // UNBIASED
        T denom;
        if (unbiased)
        {
            denom = (reso.count > 1) ? (T)(reso.count - 1) : (T)1;
        }
        else
        {
            denom = (reso.count > 0) ? (T)(reso.count) : (T)1;
        }

        if (is_std)
        {
            out[out_linear] = sqrt(reso.m2 / denom);
        }
        else
        {
            out[out_linear] = reso.m2 / denom;
        }
    }
}

template <typename T>
struct LogSumExpState
{
    T m; // max
    T s; // sum exp(x - m)
};

template <typename T>
__device__ __forceinline__ LogSumExpState<T> lse_init()
{
    // m = -inf, s = 0
    return LogSumExpState<T>{-INFINITY, (T)0};
}

template <typename T>
__device__ __forceinline__ LogSumExpState<T> lse_from_value(T x)
{
    // for a single value: m=x, s=1
    return LogSumExpState<T>{x, (T)1};
}

template <typename T>
__device__ __forceinline__ LogSumExpState<T> lse_combine(LogSumExpState<T> a, LogSumExpState<T> b)
{
    if (a.s == (T)0)
        return b;
    if (b.s == (T)0)
        return a;

    T m = (a.m > b.m) ? a.m : b.m;
    // rescale both sums into the new max frame
    T s = a.s * exp(a.m - m) + b.s * exp(b.m - m);
    return LogSumExpState<T>{m, s};
}

template <typename T>
__global__ void logsumexp_axis_contig_kernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner)
{
    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    // Each thread accumulates a partial LSE state over k
    LogSumExpState<T> st = lse_init<T>();
    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {
        T x = base[k * inner];
        st = lse_combine(st, lse_from_value<T>(x));
    }

    // Reduce across the block
    __shared__ LogSumExpState<T> smem[256]; // assumes blockDim.x <= 256
    smem[threadIdx.x] = st;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] = lse_combine(smem[threadIdx.x], smem[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        LogSumExpState<T> r0 = smem[0];
        // If R==0 (shouldn't happen), keep sane output
        out[out_linear] = (R == 0) ? (T)-INFINITY : (r0.m + log(r0.s));
    }
}

template <typename T>
void sum_axis_strided_fast_launch(
    const T *d_in,
    T *d_out,
    size_t offset,
    size_t outer,
    size_t r,
    size_t inner,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int blocksize)
{
    size_t out_elems = outer * inner;
    if (out_elems == 0)
    {
        // This is the fast path.
        return;
    }

    int block = blocksize;

    dim3 grid((unsigned)out_elems);

    switch (op)
    {
    case OP_SUM:
        fold_axis_contig_kernel<T, SumOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, SumOp{}, (T)0, PostNothing{});
        break;
    case OP_MAX:
        fold_axis_contig_kernel<T, MaxOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, MaxOp{}, std::numeric_limits<T>::lowest(), PostNothing{});
        break;
    case OP_MIN:
        fold_axis_contig_kernel<T, MinOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, MinOp{}, std::numeric_limits<T>::max(), PostNothing{});
        break;
    case OP_PROD:
        fold_axis_contig_kernel<T, ProdOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, ProdOp{}, (T)1, PostNothing{});
        break;
    case OP_MEAN:
        fold_axis_contig_kernel<T, SumOp, PostDivTotal><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, SumOp{}, (T)0, PostDivTotal{});
        break;
    case OP_VARIANCE:
        var_axis_contig_kernel<T, PostDivTotal><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, PostDivTotal{}, settings->unbiased, settings->is_std);
        break;
    case OP_LOGSUMEXP:
        logsumexp_axis_contig_kernel<T><<<grid, block>>>(d_in, d_out, offset, outer, r, inner);
        break;
    case OP_NORM:
        if (settings->norm_type == 1)
        {
            fold_axis_contig_kernel<T, L1SumOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, L1SumOp{}, (T)0, PostNothing{});
        }
        else if (settings->norm_type == 2)
        {
            map_axis_contig_kernel<T, SumOp, PostSqrt, SquareUnaryOp><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, SquareUnaryOp{}, SumOp{}, (T)0, PostSqrt{});
        }

        break;
    default:
        return;
    }
}

template <typename T>
void dispatch_flat_contiguous_reduce(
    const T *data,
    T *out,
    size_t start,
    size_t len,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int block_size)
{

    switch (op)
    {
    case OP_SUM:
        launch_flat_contiguous_reduce<T, SumOp, PostNothing>(data, out, start, len, block_size, SumOp{}, (T)0, PostNothing{});
        break;
    case OP_MIN:
        launch_flat_contiguous_reduce<T, MinOp, PostNothing>(data, out, start, len, block_size, MinOp{}, (T)std::numeric_limits<T>::max(), PostNothing{});
        break;
    case OP_MAX:
        launch_flat_contiguous_reduce<T, MaxOp, PostNothing>(data, out, start, len, block_size, MaxOp{}, (T)std::numeric_limits<T>::lowest(), PostNothing{});
        break;
    case OP_PROD:
        launch_flat_contiguous_reduce<T, ProdOp, PostNothing>(data, out, start, len, block_size, ProdOp{}, (T)1, PostNothing{});
        break;
    case OP_MEAN:
        launch_flat_contiguous_reduce<T, SumOp, PostDivTotal>(data, out, start, len, block_size, SumOp{}, (T)0, PostDivTotal{});
        break;
    case OP_VARIANCE:
        launch_flat_contiguous_reduce_variance<T>(data, out, start, len, settings, block_size);
        break;
    case OP_NORM:
        if (settings->norm_type == 1)
        {
            launch_flat_contiguous_reduce<T, L1SumOp, PostNothing>(data, out, start, len, block_size, L1SumOp{}, (T)0, PostNothing{});
        }
        else
        {
            launch_flat_contiguous_l2norm(data, out, start, len, block_size);
        }
        break;
    default:
        return;
    }
}

template <typename T>
__global__ void unpack(
    cub::KeyValuePair<int, T> *p,
    uint64_t *out)
{
    if (threadIdx.x == 0)
        *out = (uint64_t)p->key;
}

template <typename T>
void dispatch_flat_contiguous_argmax(
    const T *data,
    uint64_t *out,
    size_t start,
    size_t num_items,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int block_size)
{

    const T *d_in = data + start;

    using Pair = cub::KeyValuePair<int, T>;

    // output pair (index, value)
    Pair *d_pair_out = nullptr;
    cudaMalloc(&d_pair_out, sizeof(Pair));

    void *d_temp = nullptr;
    size_t temp_bytes = 0;

    // temp size
    if(op == OP_ARGMAX) {
        cub::DeviceReduce::ArgMax(nullptr, temp_bytes, d_in, d_pair_out, (int)num_items);
    } else if(op == OP_ARGMIN) {
        cub::DeviceReduce::ArgMin(nullptr, temp_bytes, d_in, d_pair_out, (int)num_items);
    }
    
    cudaMalloc(&d_temp, temp_bytes);

    // run
    if(op == OP_ARGMAX) {
        cub::DeviceReduce::ArgMax(d_temp, temp_bytes, d_in, d_pair_out, (int)num_items);
    } else if(op == OP_ARGMIN) {
        cub::DeviceReduce::ArgMin(d_temp, temp_bytes, d_in, d_pair_out, (int)num_items);
    }
    

    unpack<T><<<1, 1>>>(d_pair_out, out);

    cudaFree(d_temp);
    cudaFree(d_pair_out);
}

template <typename T, typename Op, typename PostOp>
__global__ void argmax_axis_contig_kernel(
    const T *__restrict__ in,
    uint64_t *__restrict__ out,
    size_t offset,
    size_t outer,
    size_t R,
    size_t inner,
    Op op,
    T init,
    PostOp post)
{

    using Pair = cub::KeyValuePair<uint64_t, T>;

    size_t out_linear = (size_t)blockIdx.x;
    size_t out_elems = outer * inner;
    if (out_linear >= out_elems)
        return;

    // Map linear output index -> (o, i)
    size_t o = out_linear / inner;
    size_t i = out_linear - o * inner;

    // Base pointer for this output element: in[o, 0, i]
    const T *base = in + offset + o * (R * inner) + i;

    T thread_biggest = init;
    uint64_t biggest_idx = 0;

    for (size_t k = threadIdx.x; k < R; k += (size_t)blockDim.x)
    {

        T cur = base[k * inner];
        if (op(thread_biggest, cur))
        {
            thread_biggest = cur;
            biggest_idx = (uint64_t)k;
        }
    }

    // Reduce thread_sum across the block (simple shared-memory reduction)
    __shared__ Pair smem[256];
    smem[threadIdx.x] = Pair{biggest_idx, thread_biggest};
    __syncthreads();

    // power-of-two reduction
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (op(smem[threadIdx.x].value, smem[threadIdx.x + s].value))
            {
                smem[threadIdx.x] = smem[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[out_linear] = smem[0].key;
    }
}

template <typename T>
void reduce_axis_strided_fast_launch(
    const T *d_in,
    uint64_t *d_out,
    size_t offset,
    size_t outer,
    size_t r,
    size_t inner,
    ReductionOpCode op,
    const ReductionSettings *settings,
    unsigned int blocksize)
{
    size_t out_elems = outer * inner;
    if (out_elems == 0)
    {
        // This is the fast path.
        return;
    }

    int block = blocksize;

    dim3 grid((unsigned)out_elems);

    if (op == OP_ARGMAX)
    {
        argmax_axis_contig_kernel<T, GreaterOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, GreaterOp{}, (T)std::numeric_limits<T>::lowest(), PostNothing{});
    }
    else if (op == OP_ARGMIN)
    {
        argmax_axis_contig_kernel<T, LessOp, PostNothing><<<grid, block>>>(d_in, d_out, offset, outer, r, inner, LessOp{}, (T)std::numeric_limits<T>::max(), PostNothing{});
    }
}

#define DECLARE_ARGMAX_LAUNCHERS(TYPE, SUFFIX)                                                    \
    extern "C" void launch_flat_contiguous_argmax_##SUFFIX(                                       \
        const TYPE *data, uint64_t *out, size_t start, size_t len,                                \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)         \
    {                                                                                             \
        dispatch_flat_contiguous_argmax<TYPE>(data, out, start, len, code, settings, block_size); \
    }                                                                                             \
                                                                                                  \
    extern "C" void launch_nd_argmax_contiguous_##SUFFIX(                                         \
        TYPE *data, uint64_t *out, size_t offset, size_t outer, size_t r, size_t inner,           \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)         \
    {                                                                                             \
        reduce_axis_strided_fast_launch<TYPE>(                                                    \
            data, out, offset, outer, r, inner, code, settings, block_size);                      \
    }

// Float types
DECLARE_ARGMAX_LAUNCHERS(float, f32)
DECLARE_ARGMAX_LAUNCHERS(double, f64)

#define DECLARE_REDUCTION_LAUNCHERS(TYPE, SUFFIX)                                                 \
    extern "C" void launch_flat_contiguous_reduce_##SUFFIX(                                       \
        const TYPE *data, TYPE *out, size_t start, size_t len,                                    \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)         \
    {                                                                                             \
        dispatch_flat_contiguous_reduce<TYPE>(data, out, start, len, code, settings, block_size); \
    }                                                                                             \
                                                                                                  \
    extern "C" void launch_nd_reduce_contiguous_##SUFFIX(                                         \
        TYPE *data, TYPE *out, size_t offset, size_t outer, size_t r, size_t inner,               \
        ReductionOpCode code, const ReductionSettings *settings, unsigned int block_size)         \
    {                                                                                             \
        sum_axis_strided_fast_launch<TYPE>(                                                       \
            data, out, offset, outer, r, inner, code, settings, block_size);                      \
    }

// Float types
DECLARE_REDUCTION_LAUNCHERS(float, f32)
DECLARE_REDUCTION_LAUNCHERS(double, f64)

// // Unsigned integer types
// DECLARE_REDUCTION_LAUNCHERS(uint8_t,  u8)
// DECLARE_REDUCTION_LAUNCHERS(uint16_t, u16)
// DECLARE_REDUCTION_LAUNCHERS(uint32_t, u32)
// DECLARE_REDUCTION_LAUNCHERS(uint64_t, u64)
// DECLARE_REDUCTION_LAUNCHERS(__uint128_t, u128)

// // Signed integer types
// DECLARE_REDUCTION_LAUNCHERS(int8_t,  i8)
// DECLARE_REDUCTION_LAUNCHERS(int16_t, i16)
// DECLARE_REDUCTION_LAUNCHERS(int32_t, i32)
// DECLARE_REDUCTION_LAUNCHERS(int64_t, i64)
// DECLARE_REDUCTION_LAUNCHERS(__int128_t, i128)
