# GPU-Accelerated CG/SCG Solvers for SIMPS

## Run all programs from the project root directory.

## GenData

```bash
g++ -O3 -fopenmp -I./include src/generator_main.cpp src/math_func.cpp -o generate_data
./generate_data
```

## Run

```bash
# 4070ti: sm_89 | 5070ti: sm_120 | A5000: sm_86
nvcc -O3 -std=c++17 -arch=sm_89 -I./include \
    -lineinfo -Xcompiler -fopenmp \
    src/main.cpp src/kernel.cu src/cg_cg.cpp src/math_func.cpp \
    -o solver
./solver
```

## NCU

```bash
# $(which ncu)
# change [spmv_kernel] into any kernelname u want to profile
sudo ncu -k spmv_kernel --launch-skip 25 --launch-count 1 --set full -o cg_spmv_report ./solver

# no -o & -full
sudo ncu -k spmv_kernel --launch-skip 25 --launch-count 1 --section SpeedOfLight --print-details ./solver
```

## NS

```bash
# On WSL2, there might not be enough permissions to view GPU data?(idk) 
sudo nsys profile -t cuda,osrt,nvtx -o cg_profile --stats=true ./solver
```
