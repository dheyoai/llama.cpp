#include "mmq.cuh"       
#include "quantize.cuh"  
#include "ggml.h"        // For ggml_type_name, and ggml_tensor, etc.
#include <fstream>
#include <vector>
#include <cstdio>        // For printf
#include <string>        // For error messages
#include <stdexcept>     // For std::runtime_error
#include <atomic>        // +++ NEW: For the header print guard

#include <hip/hip_runtime.h> // For HIP events and runtime API
#include<map>
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
static void print_csv_line(
    const std::string& layer_name,
    ggml_type type,
    long long rows,
    long long cols,
    long long quant_cycles,
    long long load_cycles,
    long long matmul_cycles,
    long long writeback_cycles
) {
    static FILE* log_file = nullptr;
    static std::map<std::string, int> layer_counts;
    int counter = ++layer_counts[layer_name];
    if (log_file == nullptr) {
        log_file = fopen("performance_log.csv", "a");
        if (log_file == nullptr) {
            fprintf(stderr, "Error: Could not open performance_log.csv for writing.\n");
            return;
        }
        fseek(log_file, 0, SEEK_END);
        long size = ftell(log_file);
        if (size == 0) {
            fprintf(log_file, "LayerName,Type,Rows,Cols,CyclesQuantize,CyclesLoad,CyclesMatmul,CyclesWriteback,Counter\n");
        }
    }

    
    fprintf(log_file, "%s,%s,%lld,%lld,%lld,%lld,%lld,%lld,%d\n",
           layer_name.c_str(),
           ggml_type_name(type),
           rows,
           cols,
           quant_cycles,
           load_cycles,
           matmul_cycles,
           writeback_cycles,
           counter

    fflush(log_file);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, mmq_args & args, hipStream_t stream){
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

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    hipStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

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

    const bool use_stream_k_nvidia = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;

    if (!ids) { 
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        
        const int64_t quant_block_num_y = (ne10_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        const int num_quant_blocks = ne11 * quant_block_num_y * (ne12*ne13);
        
        ggml_cuda_pool_alloc<long long> quant_timing_buffer_dev(ctx.pool(), num_quant_blocks * sizeof(long long));
        HIP_CHECK(hipMemsetAsync(quant_timing_buffer_dev.get(), 0, num_quant_blocks * sizeof(long long), stream));
        
        quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type,
            ne10, src1->nb[1]/ts_src1, src1->nb[2]/ts_src1, src1->nb[3]/ts_src1,
            ne10_padded, ne11, ne12, ne13,
            stream, quant_timing_buffer_dev.get());
        HIP_CHECK(hipGetLastError());

        const int64_t s12_arg = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
        const int64_t s13_arg = ne12*s12_arg;
        mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12_arg, s2,
            ne03, ne13, s03, s13_arg, s3,
            use_stream_k_nvidia,
            nullptr, 0
        };

        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        HIP_CHECK(hipGetLastError());

        std::vector<long long> quant_timing_host(num_quant_blocks > 0 ? num_quant_blocks : 1);
        HIP_CHECK(hipMemcpyAsync(quant_timing_host.data(), quant_timing_buffer_dev.get(), num_quant_blocks * sizeof(long long), hipMemcpyDeviceToHost, stream));

        const int num_mmq_blocks = args.num_mmq_blocks;
        std::vector<long long> mmq_timing_host(num_mmq_blocks > 0 ? num_mmq_blocks * 3 : 1);
        if (args.mmq_timing_buffer != nullptr) {
            HIP_CHECK(hipMemcpyAsync(mmq_timing_host.data(), args.mmq_timing_buffer, num_mmq_blocks * 3 * sizeof(long long), hipMemcpyDeviceToHost, stream));
        }

        HIP_CHECK(hipStreamSynchronize(stream));

        long long total_quant_cycles = 0;
        for(long long cycles : quant_timing_host) {
            total_quant_cycles += cycles;
        }

        long long total_load = 0, total_matmul = 0, total_writeback = 0;
        for (int i = 0; i < num_mmq_blocks; ++i) {
            total_load      += mmq_timing_host[i * 3 + 0];
            total_matmul    += mmq_timing_host[i * 3 + 1];
            total_writeback += mmq_timing_host[i * 3 + 2];
        }

        print_csv_line(src0->name[0] ? src0->name : "N/A", src0->type,
                   ne01, // WeightRows = src0->ne[1]
                   ne00, // WeightCols = src0->ne[0]
                   total_quant_cycles, total_load, total_matmul, total_writeback);
        
        return;
    }
    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<char> ids_host_vec(ggml_nbytes(ids));
    std::vector<int32_t> ids_src1_host; ids_src1_host.reserve(ne_get_rows);
    std::vector<int32_t> ids_dst_host;  ids_dst_host.reserve(ne_get_rows);
    std::vector<int32_t> tokens_per_expert_host(ne02);
    std::vector<int32_t> expert_bounds_host(ne02 + 1);
    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool());

    HIP_CHECK(hipMemcpyAsync(ids_host_vec.data(), ids->data, ggml_nbytes(ids), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

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

    const int32_t * ids_src1_dev      = ids_buf_dev.ptr;
    const int32_t * ids_dst_dev       = ids_src1_dev + ids_src1_host.size();
    const int32_t * expert_bounds_dev = ids_dst_dev + ids_dst_host.size();

    const size_t nbytes_src1_q8_1_moe_alloc = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1_moe_alloc);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    quantize_mmq_q8_1_cuda(src1_d, ids_src1_dev, src1_q8_1.get(), src0->type,
        ne10, src1->nb[1]/ts_src1, src1->nb[2]/ts_src1, src1->nb[2]/ts_src1, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
    HIP_CHECK(hipGetLastError());

    const int64_t s12_arg_moe = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13_arg_moe = ne12*s12_arg_moe;
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.ptr, ids_dst_dev, expert_bounds_dev, dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12_arg_moe, s2,
        ne03, ne13, s03, s13_arg_moe, s3,
        use_stream_k_nvidia,
        nullptr, 0 
    };

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    HIP_CHECK(hipGetLastError());
}
void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, hipStream_t stream) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    const int64_t ne0_dst_dim = dst->ne[0];
    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int64_t nrows_dst_kernel = id == ctx.device ? ne0_dst_dim : row_diff;
    const bool use_stream_k_nvidia_op = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && src1_ncols == ne11;

    mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst_kernel,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k_nvidia_op,
        nullptr, 0 
    };

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    HIP_CHECK(hipGetLastError());

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif

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

    if (new_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A && GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    return (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}