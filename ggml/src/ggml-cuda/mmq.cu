#include "mmq.cuh"       // Contains mul_mat_q_case, launch_mul_mat_q, etc.
#include "quantize.cuh"  // For quantize_mmq_q8_1_cuda (which becomes hip version)
#include "ggml.h"        // For ggml_type_name, and ggml_tensor, etc.

#include <vector>
#include <cstdio>        // For printf
#include <string>        // For error messages
#include <stdexcept>     // For std::runtime_error

#include <hip/hip_runtime.h> // For HIP events and runtime API

#ifndef HIP_CHECK
#define HIP_CHECK(cmd)                                          \
do {                                                            \
    hipError_t hip_error = cmd;                                 \
    if (hip_error != hipSuccess) {                              \
        fprintf(stderr, "HIP error: %s (%d) in function %s at %s:%d\n", \
                hipGetErrorString(hip_error), hip_error, __func__, __FILE__, __LINE__); \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(hip_error)); \
    }                                                           \
} while (0)
#endif
// <<< END OF ADDED MACROS >>>


// Original ggml_cuda_mul_mat_q_switch_type function definition
// Note: The cudaStream_t in its signature will become hipStream_t when compiled with hipcc
//       and ctx.stream() will return hipStream_t.
static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, hipStream_t stream) {
    // This function dispatches to templated host functions in mmq.cuh
    // which will ultimately launch HIP kernels.
    // No GPU timers here as this is host-side dispatch logic.
    switch (args.type_x) {
        case GGML_TYPE_Q4_0: mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream); break;
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_XS:  mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_S:   mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream); break;
        case GGML_TYPE_IQ3_S:   mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream); break;
        case GGML_TYPE_IQ1_S:   mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream); break;
        case GGML_TYPE_IQ4_XS:  mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream); break;
        case GGML_TYPE_IQ4_NL:  mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream); break;
        default: GGML_ABORT("fatal error: unsupported type in mul_mat_q_switch_type"); break;
    }
}

// MODIFIED ggml_cuda_mul_mat_q function (now effectively ggml_hip_mul_mat_q)
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS; // Defines ne00, nb00, ne10, nb10 etc.

    hipStream_t stream = ctx.stream(); // ctx.stream() should return hipStream_t for HIP backend
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc; // Assuming these helpers adapt for HIP

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            HIP_CHECK(hipMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // use_stream_k is CUDA/NVIDIA specific. For HIP, it might be always false or use a HIP-specific condition.
    // For simplicity, let's assume it might not apply directly or a HIP equivalent logic is elsewhere.
    const bool use_stream_k_nvidia = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;
    // For HIP, you might have a similar flag or assume it's not used for this specific custom kernel path.
    // For this example, we'll pass it to mmq_args, but its effect in HIP kernels would need verification/implementation.

    hipEvent_t timer_start, timer_stop;
    float gpu_time_ms = 0.0f;

    printf("[HIP_MMQ_TRACE] ggml_hip_mul_mat_q ENTER. src0: %s (%s), MoE: %s\n",
           src0->name ? src0->name : "N/A", ggml_type_name(src0->type),
           ids ? "YES" : "NO");
    printf("    src0_ne: (%lld %lld %lld %lld), src1_ne: (%lld %lld %lld %lld), dst_ne: (%lld %lld %lld %lld)\n",
        (long long)ne00, (long long)ne01, (long long)ne02, (long long)ne03,
        (long long)ne10, (long long)ne11, (long long)ne12, (long long)ne13,
        (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3);

    if (!ids) { // ----- STANDARD MATRIX MULTIPLICATION PATH -----
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq); // Assuming these helpers are portable or adapted
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1); // Assumes ctx.pool() provides HIP memory

        // --- TIMER AROUND quantize_mmq_q8_1_cuda (Standard Path, now HIP) ---
        HIP_CHECK(hipEventCreate(&timer_start));
        HIP_CHECK(hipEventCreate(&timer_stop));
        HIP_CHECK(hipEventRecord(timer_start, stream));
        {
            const int64_t s11_orig = src1->nb[1] / ts_src1;
            const int64_t s12_orig = src1->nb[2] / ts_src1;
            const int64_t s13_orig = src1->nb[3] / ts_src1;
            // This function name is kept for hipify compatibility, it launches a HIP kernel.
            quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
                ne10, s11_orig, s12_orig, s13_orig, ne10_padded, ne11, ne12, ne13, stream);
            HIP_CHECK(hipGetLastError());
        }
        HIP_CHECK(hipEventRecord(timer_stop, stream));
        HIP_CHECK(hipEventSynchronize(timer_stop));
        HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, timer_start, timer_stop));
        printf("[HIP_MMQ_TIMER] quantize (standard, src0_type %s) took: %f ms\n",
               ggml_type_name(src0->type), gpu_time_ms);
        HIP_CHECK(hipEventDestroy(timer_start));
        HIP_CHECK(hipEventDestroy(timer_stop));
        // --- END TIMER ---

        const int64_t s12_arg = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
        const int64_t s13_arg = ne12*s12_arg;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12_arg, s2,
            ne03, ne13, s03, s13_arg, s3,
            use_stream_k_nvidia // Pass the NVIDIA-specific flag; HIP kernels might ignore it or have their own logic
        };

        // --- TIMER AROUND MMQ Kernel Launch (Standard Path, now HIP) ---
        HIP_CHECK(hipEventCreate(&timer_start));
        HIP_CHECK(hipEventCreate(&timer_stop));
        HIP_CHECK(hipEventRecord(timer_start, stream));

        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream); // This calls host dispatcher
        HIP_CHECK(hipGetLastError());

        HIP_CHECK(hipEventRecord(timer_stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream)); // Sync entire stream for all kernels
        HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, timer_start, timer_stop));
        printf("[HIP_MMQ_TIMER] MMQ Kernel Exec (standard, src0_type %s) took: %f ms\n",
               ggml_type_name(args.type_x), gpu_time_ms);
        HIP_CHECK(hipEventDestroy(timer_start));
        HIP_CHECK(hipEventDestroy(timer_stop));
        // --- END TIMER ---
        return;
    }

    // ----- MIXTURE OF EXPERTS (MoE) PATH -----
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<char> ids_host_vec(ggml_nbytes(ids)); // Renamed to avoid conflict
    std::vector<int32_t> ids_src1_host; ids_src1_host.reserve(ne_get_rows);
    std::vector<int32_t> ids_dst_host;  ids_dst_host.reserve(ne_get_rows);
    std::vector<int32_t> tokens_per_expert_host(ne02);
    std::vector<int32_t> expert_bounds_host(ne02 + 1);
    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool()); // Assumes ctx.pool() handles HIP

    // --- TIMER FOR MoE INDEX COPIES (D2H and H2D) ---
    hipEvent_t moe_idx_start, moe_idx_stop;
    float moe_idx_ms = 0.0f;
    HIP_CHECK(hipEventCreate(&moe_idx_start));
    HIP_CHECK(hipEventCreate(&moe_idx_stop));
    HIP_CHECK(hipEventRecord(moe_idx_start, stream));

    HIP_CHECK(hipMemcpyAsync(ids_host_vec.data(), ids->data, ggml_nbytes(ids), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream)); // Sync for host processing of ids_host_vec

    // ... MoE host loops to populate ids_src1_host, ids_dst_host, expert_bounds_host ...
    for (int64_t i02 = 0; i02 < ne02; ++i02) { for (int64_t i12 = 0; i12 < ne12; ++i12) { for (int64_t iex = 0; iex < n_expert_used; ++iex) {
        const int32_t expert_to_use = *(const int32_t *)(ids_host_vec.data() + i12*ids->nb[1] + iex*ids->nb[0]);
        assert(expert_to_use >= 0 && expert_to_use < ne02); if (expert_to_use == i02) {
        ids_src1_host.push_back(i12*(nb12/nb11) + iex % ne11); ids_dst_host.push_back(i12*ne1 + iex);
        tokens_per_expert_host[i02]++; break;}}}}
    int32_t cumsum = 0; for (int64_t i = 0; i < ne02; ++i) { expert_bounds_host[i] = cumsum; cumsum += tokens_per_expert_host[i];}
    expert_bounds_host[ne02] = cumsum;
    std::vector<int32_t> ids_buf_host_concat;
    ids_buf_host_concat.reserve(ids_src1_host.size() + ids_dst_host.size() + expert_bounds_host.size());
    ids_buf_host_concat.insert(ids_buf_host_concat.end(), ids_src1_host.begin(), ids_src1_host.end());
    ids_buf_host_concat.insert(ids_buf_host_concat.end(), ids_dst_host.begin(), ids_dst_host.end());
    ids_buf_host_concat.insert(ids_buf_host_concat.end(), expert_bounds_host.begin(), expert_bounds_host.end());

    ids_buf_dev.alloc(ids_buf_host_concat.size() + get_mmq_x_max_host(cc));
    HIP_CHECK(hipMemcpyAsync(ids_buf_dev.ptr, ids_buf_host_concat.data(), ids_buf_host_concat.size()*sizeof(int32_t), hipMemcpyHostToDevice, stream));
    // hipStreamSynchronize here is important if quantize_mmq_q8_1_cuda needs ids_buf_dev immediately
    // For timing, let's record event after enqueueing H2D, then sync before getting time.

    HIP_CHECK(hipEventRecord(moe_idx_stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream)); // Ensure all D2H, CPU processing, H2D for indices is done
    HIP_CHECK(hipEventElapsedTime(&moe_idx_ms, moe_idx_start, moe_idx_stop));
    printf("[HIP_MMQ_TIMER] MoE index copies (D2H + H2D) took: %f ms\n", moe_idx_ms);
    HIP_CHECK(hipEventDestroy(moe_idx_start));
    HIP_CHECK(hipEventDestroy(moe_idx_stop));
    // --- END MoE INDEX COPY TIMER ---

    const int32_t * ids_src1_dev      = ids_buf_dev.ptr;
    const int32_t * ids_dst_dev       = ids_src1_dev + ids_src1_host.size();
    const int32_t * expert_bounds_dev = ids_dst_dev + ids_dst_host.size();

    const size_t nbytes_src1_q8_1_moe_alloc = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1_moe_alloc);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    // --- TIMER AROUND quantize_mmq_q8_1_cuda (MoE Path, now HIP) ---
    HIP_CHECK(hipEventCreate(&timer_start));
    HIP_CHECK(hipEventCreate(&timer_stop));
    HIP_CHECK(hipEventRecord(timer_start, stream));
    {
        const int64_t s11_orig = src1->nb[1] / ts_src1;
        const int64_t s12_orig = src1->nb[2] / ts_src1;
        const int64_t s13_orig_moe = src1->nb[2] / ts_src1; // Using original code's logic for s13
        quantize_mmq_q8_1_cuda(src1_d, ids_src1_dev, src1_q8_1.get(), src0->type,
            ne10, s11_orig, s12_orig, s13_orig_moe, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipEventRecord(timer_stop, stream));
    HIP_CHECK(hipEventSynchronize(timer_stop));
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, timer_start, timer_stop));
    printf("[HIP_MMQ_TIMER] quantize (MoE, src0_type %s) took: %f ms\n",
           ggml_type_name(src0->type), gpu_time_ms);
    HIP_CHECK(hipEventDestroy(timer_start));
    HIP_CHECK(hipEventDestroy(timer_stop));
    // --- END TIMER ---

    const int64_t s12_arg_moe = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_arg_moe = ne12*s12_arg_moe;
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.ptr, ids_dst_dev, expert_bounds_dev, dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12_arg_moe, s2,
        ne03, ne13, s03, s13_arg_moe, s3,
        use_stream_k_nvidia // Pass NVIDIA specific flag
    };

    // --- TIMER AROUND MMQ Kernel Launch (MoE Path, now HIP) ---
    HIP_CHECK(hipEventCreate(&timer_start));
    HIP_CHECK(hipEventCreate(&timer_stop));
    HIP_CHECK(hipEventRecord(timer_start, stream));

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipEventRecord(timer_stop, stream));
    HIP_CHECK(hipStreamSynchronize(stream)); // Sync entire stream
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, timer_start, timer_stop));
    printf("[HIP_MMQ_TIMER] MMQ Kernel Exec (MoE, src0_type %s) took: %f ms\n",
           ggml_type_name(args.type_x), gpu_time_ms);
    HIP_CHECK(hipEventDestroy(timer_start));
    HIP_CHECK(hipEventDestroy(timer_stop));
    // --- END TIMER ---
}

// The ggml_cuda_op_mul_mat_q and ggml_cuda_should_use_mmq functions remain largely unchanged
// as their core logic is either dispatching (for op_mul_mat_q) or CPU-based heuristics.
// If op_mul_mat_q also launches kernels directly, timers would go there too.

// (Keep original ggml_cuda_op_mul_mat_q and ggml_cuda_should_use_mmq functions)
// ...
void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, hipStream_t stream) { // Changed cudaStream_t to hipStream_t

    // ... (original content, but CUDA_CHECK should become HIP_CHECK if any CUDA API calls were here) ...
    // ... (The call to ggml_cuda_mul_mat_q_switch_type will correctly pass the hipStream_t) ...

    // Example of what might be timed if this function directly launched kernels:
    // hipEvent_t op_timer_start, op_timer_stop;
    // HIP_CHECK(hipEventCreate(&op_timer_start));
    // HIP_CHECK(hipEventCreate(&op_timer_stop));
    // HIP_CHECK(hipEventRecord(op_timer_start, stream));

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0_dst_dim = dst->ne[0]; // Renamed to avoid conflict with ne0 in args

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device(); // This should become hipGetDevice
    const int cc = ggml_cuda_info().devices[id].cc;

    const int64_t nrows_dst_kernel = id == ctx.device ? ne0_dst_dim : row_diff; // Renamed ne0

    const bool use_stream_k_nvidia_op = GGML_CUDA_CC_IS_NVIDIA(cc) &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst_kernel, // use renamed var
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k_nvidia_op};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    HIP_CHECK(hipGetLastError()); // If compiled for HIP

    // HIP_CHECK(hipEventRecord(op_timer_stop, stream));
    // HIP_CHECK(hipEventSynchronize(op_timer_stop));
    // float op_ms = 0; hipEventElapsedTime(&op_ms, op_timer_start, op_timer_stop);
    // printf("[HIP_MMQ_TIMER] ggml_cuda_op_mul_mat_q for src0_type %s took: %f ms\n", ggml_type_name(src0->type), op_ms);
    // HIP_CHECK(hipEventDestroy(op_timer_start));
    // HIP_CHECK(hipEventDestroy(op_timer_stop));


    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
    // This function is CPU-only logic, no GPU timers needed.
    // ... (original content) ...
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (new_mma_available(cc)) { // This helper would need to be adapted for AMD Matrix Cores
        return true;
    }

    // Check for DP4A equivalent for AMD if this path is critical
    // For NVIDIA:
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A && GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    // For AMD, a similar check for dot product acceleration might be needed, or assume general shader core performance.

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

        return (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE; // MMQ_DP4A_MAX_BATCH_SIZE might need tuning for AMD
}
