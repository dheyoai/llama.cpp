#include "quantize.cuh"
#include <cstdint>
static __global__ void quantize_q8_1_internal(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2,
        long long * __restrict__ d_per_block_timing_buffer 
) {
    long long start_time = 0;
    if (d_per_block_timing_buffer != nullptr && threadIdx.x == 0 && threadIdx.y == 0) {
        start_time = clock64();
    }

    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (i0 >= ne0) {
        if (d_per_block_timing_buffer != nullptr) {
            __syncthreads(); 
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                long long end_time_early = clock64();
                int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
                d_per_block_timing_buffer[block_id] = end_time_early - start_time;
            }
        }
        return;
    }
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i_cont = ((i3*ne2 + i2) * ne1 + i1) * ne0 + i0;
    block_q8_1 * y = (block_q8_1 *) vy;
    const int64_t ib  = i_cont / QK8_1;
    const int64_t iqs = i_cont % QK8_1;
    const float xi = i0 < ne00 ? x[(i3*s03 + i2*s02 + i1*s01 + i0)] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;
    amax = warp_reduce_max(amax);
    sum  = warp_reduce_sum(sum);
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const int8_t q = d == 0.0f ? 0 : roundf(xi / d);
    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        // This is a partial update, not the end of the block's work.
        // Don't record final timing here. Only the thread with iqs==0 does.
    } else {
        reinterpret_cast<half&>(y[ib].ds.x) = d;
        reinterpret_cast<half&>(y[ib].ds.y) = sum;
    }
    if (d_per_block_timing_buffer != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            long long end_time = clock64();
            int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
            d_per_block_timing_buffer[block_id] = end_time - start_time;
        }
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1_internal(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2,
        long long * __restrict__ d_per_block_timing_buffer
) {
    long long start_time = 0;
    if (d_per_block_timing_buffer != nullptr && threadIdx.x == 0 && threadIdx.y == 0) {
        start_time = clock64();
    }

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        if (d_per_block_timing_buffer != nullptr) {
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                long long end_time_early = clock64();
                int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
                d_per_block_timing_buffer[block_id] = end_time_early - start_time;
            }
        }
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;
    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1);
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4*QK8_1);
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y)); amax = fmaxf(amax, fabsf(xi.z)); amax = fmaxf(amax, fabsf(xi.w));
    #pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE)); }
    float sum = 0.0f; // Initialize to avoid using uninitialized value
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
        #pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE); }
    }
    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    char4 q;
    q.x = roundf(xi.x*d_inv); q.y = roundf(xi.y*d_inv); q.z = roundf(xi.z*d_inv); q.w = roundf(xi.w*d_inv);
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (threadIdx.x % (vals_per_scale/4) == 0) { // Only leader threads for a scale/sum group write
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
            }
            if (iqs % 64 == 0) {
                const float d = 1.0f / d_inv;
                y[ib].d2s6[iqs/64] = d;
            }
        } else { // D4 and DS4 layouts
            if (iqs % 32 == 0) {
                const float d = 1.0f / d_inv;
                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) { y[ib].ds4[iqs/32] = make_half2(d, sum); }
                else { y[ib].d4[iqs/32]  = d; }
            }
        }
    }
    
    if (d_per_block_timing_buffer != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            long long end_time = clock64();
            int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
            d_per_block_timing_buffer[block_id] = end_time - start_time;
        }
    }
}
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream
) {
    GGML_ASSERT(!ids);
    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_internal<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2, nullptr);
    GGML_UNUSED(type_src0);
}
void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream,
        long long * d_per_block_timing_buffer
) {
    GGML_ASSERT(!ids);
    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_internal<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2, d_per_block_timing_buffer);
    GGML_UNUSED(type_src0);
}
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream
) {
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_D4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, nullptr); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_DS4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, nullptr); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_D2S6><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, nullptr); break;
        default: GGML_ABORT("fatal error"); break;
    }
}
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, hipStream_t stream,
        long long * d_per_block_timing_buffer
) 
{
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_D4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, d_per_block_timing_buffer); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_DS4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, d_per_block_timing_buffer); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_internal<MMQ_Q8_1_DS_LAYOUT_D2S6><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, d_per_block_timing_buffer); break;
        default: GGML_ABORT("fatal error"); break;
    }
}