# GPU-Accelerated CG/SCG Solvers for SIMPS
## GenData
```bash
g++ -O3 -fopenmp -I./include src/generator_main.cpp src/math_func.cpp -o generate_data
./generate_data
```
## Run
```bash
g++ -O3 -fopenmp -I./include src/main.cpp src/cg_cg.cpp src/math_func.cpp -o run_solver
./run_solver
```