#ifndef SPMV_CUDA_H
#define SPMV_CUDA_H

#include "extern_gloval_variables.h" // 引入 FLOAT 定义

// 定义一个不透明的句柄类型，用于隐藏 GPU 内部状态
typedef void* SpmvHandle;

/**
 * @brief 初始化 SpMV 上下文
 * * 1. 分配 GPU 显存 (x, y, val, col)
 * 2. 将静态矩阵数据 (val, col) 从 CPU 传输到 GPU
 * 3. 预热 Kernel
 * * @param N           粒子数 (行数)
 * @param ell_width   ELL 宽度 (最大邻居数)
 * @param h_val_cm    列主序 (Column-Major) 的矩阵数值数组指针 (Host)
 * @param h_col_cm    列主序 (Column-Major) 的列索引数组指针 (Host)
 * @return SpmvHandle 返回上下文句柄，失败返回 NULL
 */
SpmvHandle spmv_init(int N, int ell_width, const FLOAT* h_val_cm, const int* h_col_cm);

/**
 * @brief 执行 SpMV (y = A * x)
 * * 1. 将向量 x 从 Host 传到 Device
 * 2. 执行 Kernel
 * 3. 将结果 y 从 Device 传回 Host
 * * @param handle SpmvHandle 句柄
 * @param h_x    输入向量 x (Host 指针)
 * @param h_y    输出向量 y (Host 指针)
 */
void spmv_exec(SpmvHandle handle, const FLOAT* h_x, FLOAT* h_y);

/**
 * @brief 释放 SpMV 上下文
 * * 释放所有相关的 GPU 显存并销毁句柄
 */
void spmv_free(SpmvHandle handle);

#endif // SPMV_CUDA_H