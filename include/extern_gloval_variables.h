#ifndef EXTERN_GLOBAL_VARIABLES
#define EXTERN_GLOBAL_VARIABLES

// ==========================================
// 1. 精度与宏定义 (Configuration)
// ==========================================

#if defined(PRECISION_FLOAT)
  #define FLOAT float
  #define FLOAT_SIZE 32
#else
  #define FLOAT double
  #define FLOAT_SIZE 64
#endif

// 求解器核心参数
#define Ell_Length 160     // ELL 矩阵宽
#define TMAX 300           // CG 最大迭代次数
#define EPS 1.0e-8         // 收敛容差
#define ELL_NULL -1        // ELL 空位标记

// ==========================================
// 2. 核心求解器变量 (Solver Arrays)
// ==========================================

// ELL 稀疏矩阵数据
extern FLOAT *MatA_ELLVal; // 矩阵数值 [N * Ell_Length]
extern int   *MatA_ELLCol; // 列索引   [N * Ell_Length]

// CG 迭代向量
extern FLOAT *x;           // 未知量/解向量 (Pressure)
extern FLOAT *b;           // 右端项 (Source Term)
extern FLOAT *r;           // 残差向量
extern FLOAT *p;           // 搜索方向向量
extern FLOAT *Ax;          // A*x 临时向量
extern FLOAT *Ap;          // A*p 临时向量

// ==========================================
// 3. 全局标量 (Scalars)
// ==========================================

// 粒子数 / 矩阵维度 (N)
extern int NumberOfParticles;

// ==========================================
// 4. 调试与性能统计 (Debug & Profiling)
// ==========================================

// 累计 CG 迭代总次数 (用于计算平均收敛速度)
extern int cg_sum_loop_count_profile;

// 全局时间步 (用于控制 Debug 输出频率，例如只在第 666 步输出)
extern int iTimeStep_gloval;

// [新增] CG 内部耗时统计 (单位: 毫秒)
extern double time_spmv;   // 稀疏矩阵乘法耗时 (Ap = A*p)
extern double time_dot;    // 内积/范数耗时 (dot product)
extern double time_axpy;   // 向量更新耗时 (x+=alpha*p, r-=alpha*Ap, p=r+beta*p)

#endif /* EXTERN_GLOBAL_VARIABLES */