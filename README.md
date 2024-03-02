## Block Vecchia Algorithm 

Our computational harness is built using gcc version 10.2.0 (12.2.0) and CUDA version 11.4 (11.8). It was linked with Intel MKL 2022.2.1, MAGMA 2.7.0, GSL 2.6, and NLopt v2.7.1 optimization libraries. 

#### 1. Installation Guidence

Please be asured that all related libraries are well installed and their path included in the system PATH (LD_LIBRARY_PATH)

#### 2. Usage 

Here are two examples illustrating how to use the block Vecchia (afterwards your results are stored in the `./log` fie)

0. Helper 
`./bin/test_dvecchia_batch --help (-h)`

1. Performance test, such as monitor the time or calculate the intermediate results in KL divergence,
`./bin/test_dvecchia_batch --ikernel 1.5:0.1:0.5 --kernel univariate_matern_stationary_no_nugget --num_loc 20000 --perf --vecchia_cs 300 --vecchia_bc 1500 --knn --seed 0`

2. Simulated data/ Real dataset.
`./bin/test_dvecchia_batch --ikernel ?:?:? --vecchia_bc 300 --kernel univariate_matern_stationary_no_nugget --num_loc 20000   --vecchia_cs 150 --knn --xy_path replace_your_location_path --obs_path replace your_observation_path`
(optional)
`--kernel_init 0.1:0.1:0.1  --tol 4 --omp_threads 40`

