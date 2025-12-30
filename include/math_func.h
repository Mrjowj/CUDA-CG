#ifndef MATH_FUNC
#define MATH_FUNC

#include "extern_gloval_variables.h"

// static inline double weight(double distance, double re);
// 汎用的な計算に用いる関数群を定義する
static inline FLOAT weight( FLOAT distance, FLOAT re )
{
    if( distance >= re )
    {
        return 0.0;
    }
    else
    {
        return (re/distance) - 1.0;
    }
}

void vector_x_matrix(FLOAT *y, FLOAT* a, FLOAT* x);
FLOAT dot_product(FLOAT* x, FLOAT* y);
FLOAT vector_norm(FLOAT* x);
FLOAT ip_omp(FLOAT* x, FLOAT* y);

void mvm_ell_normal(FLOAT *mat_ellVal, int *mat_ellVCol, FLOAT  *x, FLOAT  *Ax);

#endif /* MATH_FUNC */
