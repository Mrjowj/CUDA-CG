/**
 * @file cg_cuda.cu
 * @brief Full GPU Conjugate Gradient Solver Implementation
 * @author Optimized by Gemini
 */

#include "../include/kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cmath>

// =================================================================================
// 1. Error Handling & Constants
// =================================================================================

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__              \
                  << " in " << #call << " : " << cudaGetErrorString(err__)        \
                  << std::endl;                                                   \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

#define CUDA_KERNEL_CHECK() do {                                                  \
    cudaError_t err__ = cudaGetLastError();                                       \
    if (err__ != cudaSuccess) {                                                   \
        std::cerr << "Kernel launch error at " << __FILE__ << ":" << __LINE__     \
                  << " : " << cudaGetErrorString(err__) << std::endl;             \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;

// 解决宏定义冲突
#ifndef ELL_NULL
constexpr int ELL_NULL = -1;
#endif

// =================================================================================
// 2. Context Structure
// =================================================================================

struct CgContext {
    int N;
    int ell_width;

    // --- GPU Memory (Full Residency) ---
    FLOAT* d_val = nullptr; // Matrix Value (Col-Major)
    int*   d_col = nullptr; // Matrix Index (Col-Major)
    FLOAT* d_b   = nullptr; // RHS
    FLOAT* d_x   = nullptr; // Solution
    FLOAT* d_r   = nullptr; // Residual
    FLOAT* d_p   = nullptr; // Search Direction
    FLOAT* d_Ap  = nullptr; // A * p
    FLOAT* d_Ax  = nullptr; // Temp for Init

    // --- Reduction Buffer ---
    // 用于存放每个 Block 归约后的部分和
    // 假设 GridSize 最大不超过 4096 (1M 粒子足够了)
    FLOAT* d_partial_sums = nullptr; 
    FLOAT* h_partial_sums = nullptr; // Pinned memory on Host for fast readback
    int num_blocks = 0;
};

// =================================================================================
// 3. Device Helper: Warp Shuffle Reduction
// =================================================================================

template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    // __shfl_down_sync 需要 mask，0xffffffff 代表 warp 内所有线程都参与
    // 每次向下移动 16, 8, 4, 2, 1 位，累加
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// =================================================================================
// 4. Kernels
// =================================================================================

/**
 * @brief SpMV Kernel (Column-Major)
 * y = A * x
 */
template <typename T>
__global__ void spmv_kernel(
    const int N, const int width,
    const int* __restrict__ col_idx,
    const T* __restrict__ val,
    const T* __restrict__ x,
    T* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        T sum = 0.0;
        for (int j = 0; j < width; ++j) {
            int idx = j * N + row; // Col-Major Access
            int c = col_idx[idx];
            if (c != ELL_NULL) {
                // FMA here
                sum += val[idx] * x[c];
            }
        }
        y[row] = sum;
    }
}

/**
 * @brief Dot Product Kernel (Partial Reduction)
 * Computes dot(x, y) and stores partial sums per block in result array.
 */
template <typename T>
__global__ void dot_kernel(
    const int N,
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ partial_sums
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    T sum = 0.0;
    // Grid-Stride Loop (in case N > Total Threads)
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += x[i] * y[i];
    }

    // 1. Warp Reduction
    sum = warp_reduce_sum(sum);

    // 2. Block Reduction (using Shared Memory)
    // 一个 Block 最多 32 个 Warp，只需 32 个槽位(Block Size 1024 / 32 = 32)
    static __shared__ T shared[32]; 
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;

    if (lane == 0) {
        shared[wid] = sum; // 每个 Warp 的第一个线程把结果写入 Shared Mem
    }
    __syncthreads();

    // 3. Final reduction by first Warp
    // 只有第一个 Warp 负责把 Shared Mem 里的 32 个数加起来
    sum = (tid < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0;
    
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
    }

    // 4. Store Block Result
    if (tid == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

/**
 * @brief Init Residual Kernel
 * r = b - Ax
 * p = r
 * Also computes partial dot product of (r, r) for initial residue
 */
template <typename T>
__global__ void init_r_p_dot_kernel(
    const int N,
    const T* __restrict__ b,
    const T* __restrict__ Ax,
    T* __restrict__ r,
    T* __restrict__ p,
    T* __restrict__ partial_sums
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    T sum_sq = 0.0;
    
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        T val_r = b[i] - Ax[i];
        r[i] = val_r;
        p[i] = val_r;
        sum_sq += val_r * val_r;
    }

    // Reduction Logic (Same as dot_kernel)
    sum_sq = warp_reduce_sum(sum_sq);
    static __shared__ T shared[32];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    T val = (tid < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);
    if (tid == 0) partial_sums[blockIdx.x] = val;
}

/**
 * @brief Fused Update X, R and Compute R_Norm
 * x = x + alpha * p
 * r = r - alpha * Ap
 * return dot(r, r)
 */
template <typename T>
__global__ void update_xr_dot_kernel(
    const int N,
    const T alpha,
    const T* __restrict__ p,
    const T* __restrict__ Ap,
    T* __restrict__ x,
    T* __restrict__ r,
    T* __restrict__ partial_sums
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    T sum_sq = 0.0;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        // Load
        T val_x = x[i];
        T val_r = r[i];
        T val_p = p[i];
        T val_Ap = Ap[i];

        // Update
        val_x += alpha * val_p;
        val_r -= alpha * val_Ap;

        // Store
        x[i] = val_x;
        r[i] = val_r;

        // Accumulate
        sum_sq += val_r * val_r;
    }

    // Reduction
    sum_sq = warp_reduce_sum(sum_sq);
    static __shared__ T shared[32];
    int lane = tid % WARP_SIZE;
    int wid  = tid / WARP_SIZE;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    T val = (tid < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);
    if (tid == 0) partial_sums[blockIdx.x] = val;
}

/**
 * @brief Update P
 * p = r + beta * p
 */
template <typename T>
__global__ void update_p_kernel(
    const int N,
    const T beta,
    const T* __restrict__ r,
    T* __restrict__ p
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        p[i] = r[i] + beta * p[i];
    }
}

// =================================================================================
// 5. Host Functions (Controller)
// =================================================================================

// 辅助：CPU端计算部分和
FLOAT sum_reduce_cpu(FLOAT* partial_sums, int count) {
    FLOAT total = 0.0;
    for (int i = 0; i < count; ++i) {
        total += partial_sums[i];
    }
    return total;
}

CgCudaHandle cg_cuda_init(int N, int ell_width, 
                          const FLOAT* h_val_cm, const int* h_col_cm, const FLOAT* h_b) 
{
    CgContext* ctx = new CgContext();
    ctx->N = N;
    ctx->ell_width = ell_width;

    // Grid Config
    // 这里的策略是：Blocks 数量足够覆盖 N，但不超过 GPU 的 SM 承载极限太多
    // 对于 200k 粒子，256 线程/block -> 约 782 blocks。完全没问题。
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 限制最大 Block 数以防超大 N 导致 reduce 数组过大 (可选，这里设 4096 足够)
    if (num_blocks > 4096) num_blocks = 4096;
    
    ctx->num_blocks = num_blocks;

    size_t sz_mat_f = (size_t)N * ell_width * sizeof(FLOAT);
    size_t sz_mat_i = (size_t)N * ell_width * sizeof(int);
    size_t sz_vec   = (size_t)N * sizeof(FLOAT);
    size_t sz_part  = (size_t)num_blocks * sizeof(FLOAT);

    printf("[CG-CUDA] Init: Allocating GPU Memory for N=%d...\n", N);

    // Alloc
    CUDA_CHECK(cudaMalloc(&ctx->d_val, sz_mat_f));
    CUDA_CHECK(cudaMalloc(&ctx->d_col, sz_mat_i));
    CUDA_CHECK(cudaMalloc(&ctx->d_b, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_x, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_r, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_p, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_Ap, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_Ax, sz_vec));
    CUDA_CHECK(cudaMalloc(&ctx->d_partial_sums, sz_part));
    CUDA_CHECK(cudaMallocHost(&ctx->h_partial_sums, sz_part)); // Pinned Memory

    // Transfer Static Data
    CUDA_CHECK(cudaMemcpy(ctx->d_val, h_val_cm, sz_mat_f, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col, h_col_cm, sz_mat_i, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_b, h_b, sz_vec, cudaMemcpyHostToDevice));

    // Reset X
    CUDA_CHECK(cudaMemset(ctx->d_x, 0, sz_vec));

    return (CgCudaHandle)ctx;
}

int cg_cuda_solve(CgCudaHandle handle, FLOAT* h_x_out, int max_iter, FLOAT tol) {
    CgContext* ctx = (CgContext*)handle;
    CUDA_CHECK(cudaMemset(ctx->d_x, 0, ctx->N * sizeof(FLOAT)));
    int N = ctx->N;
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(ctx->num_blocks);
    
    FLOAT r0_sq, rk_sq, rk_new_sq;
    FLOAT alpha, beta, pAp;

    // --- 1. Initialization (r = b - Ax, p = r) ---
    // 由于我们总是重置 x=0, 所以 Ax=0, r=b。
    // 但为了通用性（比如支持 Warm Start），我们还是完整算一遍。
    
    // Ax = A * x
    spmv_kernel<FLOAT><<<grid, block>>>(N, ctx->ell_width, ctx->d_col, ctx->d_val, ctx->d_x, ctx->d_Ax);
    
    // r = b - Ax; p = r; r0_sq = dot(r,r)
    init_r_p_dot_kernel<FLOAT><<<grid, block>>>(N, ctx->d_b, ctx->d_Ax, ctx->d_r, ctx->d_p, ctx->d_partial_sums);
    
    // Copy reduction result
    CUDA_CHECK(cudaMemcpy(ctx->h_partial_sums, ctx->d_partial_sums, ctx->num_blocks * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    r0_sq = sum_reduce_cpu(ctx->h_partial_sums, ctx->num_blocks);
    
    rk_sq = r0_sq;
    FLOAT r0_norm = std::sqrt(r0_sq);
    
    if (r0_norm < tol) return 0; // Already converged

    // --- 2. Main Loop ---
    int k;
    for (k = 0; k < max_iter; k++) {
        
        // Ap = A * p
        spmv_kernel<FLOAT><<<grid, block>>>(N, ctx->ell_width, ctx->d_col, ctx->d_val, ctx->d_p, ctx->d_Ap);

        // pAp = dot(p, Ap)
        dot_kernel<FLOAT><<<grid, block>>>(N, ctx->d_p, ctx->d_Ap, ctx->d_partial_sums);
        CUDA_CHECK(cudaMemcpyAsync(ctx->h_partial_sums, ctx->d_partial_sums, ctx->num_blocks * sizeof(FLOAT), cudaMemcpyDeviceToHost));
        
        // Wait for dot product
        CUDA_CHECK(cudaStreamSynchronize(0));
        pAp = sum_reduce_cpu(ctx->h_partial_sums, ctx->num_blocks);

        // Alpha
        if (std::abs(pAp) < 1e-12) break; // Breakdown protection
        alpha = rk_sq / pAp;

        // x += alpha*p; r -= alpha*Ap; rk_new = dot(r,r)
        // [FUSED KERNEL]
        update_xr_dot_kernel<FLOAT><<<grid, block>>>(N, alpha, ctx->d_p, ctx->d_Ap, ctx->d_x, ctx->d_r, ctx->d_partial_sums);
        CUDA_CHECK(cudaMemcpyAsync(ctx->h_partial_sums, ctx->d_partial_sums, ctx->num_blocks * sizeof(FLOAT), cudaMemcpyDeviceToHost));
        
        // Wait for dot product
        CUDA_CHECK(cudaStreamSynchronize(0));
        rk_new_sq = sum_reduce_cpu(ctx->h_partial_sums, ctx->num_blocks);

        // Convergence Check
        if (std::sqrt(rk_new_sq) < tol * r0_norm) { // Relative Tolerance
            k++;
            break;
        }

        // Beta
        beta = rk_new_sq / rk_sq;
        rk_sq = rk_new_sq;

        // p = r + beta*p
        update_p_kernel<FLOAT><<<grid, block>>>(N, beta, ctx->d_r, ctx->d_p);
    }

    // --- 3. Output ---
    CUDA_CHECK(cudaMemcpy(h_x_out, ctx->d_x, N * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    return k;
}

void cg_cuda_free(CgCudaHandle handle) {
    if (!handle) return;
    CgContext* ctx = (CgContext*)handle;
    
    if (ctx->d_val) cudaFree(ctx->d_val);
    if (ctx->d_col) cudaFree(ctx->d_col);
    if (ctx->d_b)   cudaFree(ctx->d_b);
    if (ctx->d_x)   cudaFree(ctx->d_x);
    if (ctx->d_r)   cudaFree(ctx->d_r);
    if (ctx->d_p)   cudaFree(ctx->d_p);
    if (ctx->d_Ap)  cudaFree(ctx->d_Ap);
    if (ctx->d_Ax)  cudaFree(ctx->d_Ax);
    if (ctx->d_partial_sums) cudaFree(ctx->d_partial_sums);
    if (ctx->h_partial_sums) cudaFreeHost(ctx->h_partial_sums);
    
    delete ctx;
    printf("[CG-CUDA] Freed GPU resources.\n");
}