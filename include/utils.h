/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file include/utils.h

 *
 *
 *
 * @version 1.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/

#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <string>
#include <iomanip>

location *loadXYcsv(const std::string file_path, int n, bool time_flag)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(file_path);
    if (!testFile)
    {
        std::cerr << "Error: File " << file_path << " does not exist\n";
        exit(0);
    }
    location *loc = (location *)malloc(sizeof(location));
    loc->x = (double *)malloc(n * sizeof(double));
    loc->y = (double *)malloc(n * sizeof(double));
    loc->z = time_flag ? (double *)malloc(n * sizeof(double)) : NULL;
    std::ifstream file(file_path);
    std::string line;
    int i = 0;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        loc->x[i] = std::stod(token);
        std::getline(ss, token, ',');
        loc->y[i] = std::stod(token);
        if (time_flag){
            std::getline(ss, token, ',');
            loc->z[i] = std::stod(token);
        }
        // printf("(%f ,%f ,%f , %d)", loc->x[i], loc->y[i], loc->z[i], i);
        ++i;
    }
    file.close();
    return loc;
}

template <class T>
void loadObscsv(const std::string &file_path, int n, T *obs)
{
    // Check if the file exists by trying to open it
    std::ifstream testFile(file_path);
    if (!testFile)
    {
        std::cerr << "Error: File " << file_path << " does not exist\n";
        exit(0);
    }
    // T* obs = (double* ) malloc(n * sizeof(double));
    std::ifstream file(file_path);
    std::string line;
    int i = 0;
    while (std::getline(file, line))
    {
        std::istringstream ss(line);
        T value;
        if (ss >> value)
        {
            obs[i] = value;
            ++i;
        }
    }
    file.close();
}

int createDirectoryIfNotExists(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0)
    {
        if (S_ISDIR(st.st_mode))
        {
            return 0; // Directory already exists
        }
        else
        {
            return -1; // Path exists but is not a directory
        }
    }
    else
    {
        if (mkdir(path, 0777) == 0)
        {
            return 0; // Directory created successfully
        }
        else
        {
            return -1; // Failed to create the directory
        }
    }
}

void createLogFile(Vecchia_opts &opts)
{
    const char *log_dir = "./log";

    int result = createDirectoryIfNotExists(log_dir);

    if (result == 0)
    {
        printf("Directory exists or was created successfully.\n");
    }
    else
    {
        printf("Failed to create the directory.\n");
    }
}

template <class T>
void saveLogFileSum(int iterations, double* theta, double max_llh, double whole_time, Vecchia_opts &opts)
{

    std::string file_path;
    std::string ordering_name;
    if (opts.perf == 1)
    {
        file_path = "./log/locs_" + std::to_string(opts.num_loc) + "_" + "cs_" + std::to_string(opts.vecchia_cs) + "_" + "bc_" + std::to_string(opts.vecchia_bc) + "_" + "seed_" + std::to_string(opts.seed) + "_kernel_" + std::to_string(opts.sigma) + ":" + std::to_string(opts.beta) + ":" + std::to_string(opts.nu);
    }
    else
    {
        file_path = "./log/locs_" + std::to_string(opts.num_loc) + "_" + "cs_" + std::to_string(opts.vecchia_cs) + "_" + "bc_" + std::to_string(opts.vecchia_bc) + "_" + "seed_" + std::to_string(opts.seed) + "_kernel_" + std::to_string(opts.sigma) + ":" + std::to_string(opts.beta) + ":" + std::to_string(opts.nu);
    }
    if (opts.mortonordering)
        file_path = file_path + "_morton";
    else if (opts.randomordering)
        file_path = file_path + "_random";
    else if (opts.kdtreeordering)
        file_path = file_path + "_kdtree";
    else if (opts.hilbertordering)
        file_path = file_path + "_hilbert";
    else if (opts.mmdordering)
        file_path = file_path + "_mmd";

    // Print the log message to the log file using printf
    printf("Total Number of Iterations = %d \n", iterations);
    printf("Total Optimization Time = %lf secs \n", whole_time);
    // matern + power exponential kernel
    if (opts.kernel == 1 || opts.kernel == 2)
    {
        printf("Model Parameters (Variance, range, smoothness): (%.8f, %.8f, %.8f) -> Loglik: %.18f \n",
               theta[0], theta[1], theta[2], max_llh);
        std::ofstream outfile(file_path);

        // Write the headers for the CSV file
        outfile << "Iterations, variance, range, smoothness, log-likelihood, time " << std::endl;
        // Write the log data to the CSV file
        outfile << iterations << ", "
                << theta[0] << ", " << theta[1] << ", " << theta[2] << ", "
                << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh
                << ", " << whole_time << std::endl;
        outfile.close();
    }
    else if (opts.kernel == 3)
    {
        printf("Model Parameters (Variance, range, smoothness, nugget): (%.8f, %.8f, %.8f, %.8f) -> Loglik: %.18f \n",
               theta[0], theta[1], theta[2], theta[3], max_llh);
        std::ofstream outfile(file_path);

        // Write the headers for the CSV file
        outfile << "Iterations, variance, range, smoothness, nugget, log-likelihood, time " << std::endl;
        // Write the log data to the CSV file
        outfile << iterations << ", "
                << theta[0] << ", " << theta[1] << ", " << theta[2] << ", " << theta[3] << ", "
                << std::setprecision(std::numeric_limits<double>::max_digits10) << max_llh << ", "
                << whole_time << std::endl;
        outfile.close();
    }
}

/*
Some Auxiliary print function
*/
void printLocations(int N, location *locations)
{
    fprintf(stderr, "\n---------------------------------\n");
    for (int i = 0; i < N; i++)
    {
        fprintf(stderr, "%d th location: (%lf, %lf)\n", i, locations->x[i], locations->y[i]);
    }
    fprintf(stderr, "-----------------------------------\n");
}

template <class T>
void printMatrixCPU(int m, int n, T *h_A, int lda, int i)
{
    printf("-------------------------------\n");
    printf("%d batch of all. (CPU)\n", i);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // colunm-wised matrix
            // printf("(%d)", i + j * lda);
            printf("%lg ,", (double)h_A[i + j * lda]);
            // printf(", ");
        }
        printf("\n");
    }
    printf("-------------------------------\n");
}

template <class T>
void printVectorCPU(int m, T *h_C, int ldc, int i)
{
    printf("-------------------------------\n");
    printf("%d batch of all. (CPU)\n", i);
    for (int i = 0; i < m; i++)
    {
        // printf("(%d)", i + j * lda);
        printf("%lg ,", (double)h_C[i]);
        // printf(", ");
    }
    printf("\n");
    printf("-------------------------------\n");
}

template <class T>
void printMatrixGPU(int Am, int An, T *d_A, int lda, int k)
{
    // double *devicePointer;
    // cudaMemcpy(&devicePointer, &d_A, sizeof(double *), cudaMemcpyDeviceToHost);
    fprintf(stderr, "-------------------------------\n");
    fprintf(stderr, "%dth Convariance matrix in batch. (GPU)\n", k);
    T *h_A = (T *)malloc(sizeof(T) * An * lda);
    cudaMemcpy(h_A, d_A, sizeof(T) * An * lda, cudaMemcpyDeviceToHost);
    // double sum = 0;
    for (int i = 0; i < Am; i++)
    {
        for (int j = 0; j < An; j++)
        {
            // colunm-wised matrix
            // printf("(%d)", i + j * lda);
            printf("%.10lf ", (double)h_A[i + j * lda]);
            // sum += (double)h_A[i + j * lda];
        }
        printf("\n");
    }
    // printf("The sum is %lf \n", sum);
    printf("-------------------------------\n");
    free(h_A);
}

template <class T>
void printVecGPU(int Cm, int Cn, T *d_C, int lda, int i)
{
    printf("-------------------------------\n");
    printf("%dst batch of all. (GPU) vector\n", i);
    T *h_C = (T *)malloc(sizeof(T) * Cn * lda);
    cudaMemcpy(h_C, d_C, sizeof(T) * Cn * lda, cudaMemcpyDeviceToHost);
    for (int i = 0; i < Cm; i++)
    {
        printf("(%d)", i);
        printf("%lf ", (double)h_C[i]);
    }
    printf("\n-------------------------------\n");
    free(h_C);
}

template <class T>
void printVecGPUv1(int Cm, T *d_C)
{
    printf("-------------------------------\n");
    T *h_C = (T *)malloc(sizeof(T) * Cm);
    cudaMemcpy(h_C, d_C, sizeof(T) * Cm, cudaMemcpyDeviceToHost);
    for (int i = 0; i < Cm; i++)
    {
        // printf("(%d)", i);
        printf("%lf, ", (double)h_C[i]);
    }
    printf("\n-------------------------------\n");
    free(h_C);
}

#endif