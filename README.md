# Block Vecchia Algorithm

## Introduction

The Block Vecchia Algorithm is a computational method for efficient spatial statistics calculations. This implementation is designed for high-performance computing environments, leveraging GPU acceleration and optimized libraries.

Our computational harness is built using:
- GCC version 10.2.0 (12.2.0)
- CUDA version 11.4 (11.8)
- Intel MKL 2022.2.1
- MAGMA 2.7.2
- GSL 2.6
- NLopt v2.7.1 optimization libraries

## Installation

### Prerequisites

Ensure you have the following libraries installed and their paths included in your system's PATH and LD_LIBRARY_PATH:

- GCC 10.2.0 or later
- CUDA 11.4 or later
- Intel MKL 2022.2.1 or later
- MAGMA 2.7.0 or later
- GSL 2.6 or later
- NLopt v2.7.1 or later

## Usage

The main executable for the Block Vecchia Algorithm is `test_dvecchia_batch`. Here are some example use cases:

### 1. View Help Information

To see all available options:

```
./bin/test_dvecchia_batch --help
```

### 2. Performance Test

To run a performance test and monitor execution time or calculate intermediate results in KL divergence:

```
./bin/test_dvecchia_batch --ikernel 1.5:0.1:0.5 --kernel univariate_matern_stationary_no_nugget --num_loc 20000 --perf --vecchia_cs 300 --vecchia_bc 1500 --knn --seed 0
```

### 3. Simulated Data or Real Dataset

To run the algorithm on simulated data or a real dataset:

```
./bin/test_dvecchia_batch --ikernel ?:?:? --vecchia_bc 300 --kernel univariate_matern_stationary_no_nugget --num_loc 20000 --vecchia_cs 150 --knn --xy_path /path/to/locations --obs_path /path/to/observations
```

Optional parameters:
```
--kernel_init 0.1:0.1:0.1 --tol 4 --omp_threads 40
```

### Output

Results are stored in the `./log` file after execution.

## License

This project is licensed under the terms of the Apache license. See the LICENSE file for details.