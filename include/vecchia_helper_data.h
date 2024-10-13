/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file include/vecchia_helper_data

 *
 *
 *
 * @version 1.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/

#ifndef VECCHIA_HELPER_H
#define VECCHIA_HELPER_H

#define PI (3.141592653589793)
#define BATCHCOUNT_MAX 999999999

#include "magma_v2.h"

// Function to display help text
void displayHelp()
{
    std::cout << "Usage: test_dvecchia_batch [options]\n"
              << "Options:\n"
              << "  --help                Display this help message and exit\n"
              << "  --ikernel             The parameters in kernel, sigma^2:range:smooth, e.g., 1.5:0.1:0.5\n"
              << "  --kernel              The name of kernels, such as matern kernel, e.g., univariate_matern_stationary_no_nugget\n"
              << "  --kernel_init         The initial values of parameters in kernel, sigma^2:range:smooth, e.g., 1.5:0.1:0.5\n"
              << "  --vecchia_bc          [int] The block count in Vecchia method (number of clusters), e.g., 300,\n"
              << "  --vecchia_cs          [int] The conditioning size in Vecchia method, e.g., 1500,\n"
              << "  --num_loc             [int] The number of locations, e.g., 20000,\n"
              << "  --knn                 nearest neighbors searching, default to use.\n"
              << "  --perf                Only calculate the one iteraion of block/classic Vecchia and obs=0.\n"
              << "  --seed                [int] random generation for locations and observations.\n"
              << "  --3D                  [bool] 3D profile spatial modeling.\n"
              << "  --earth               [int] representing GC distance.\n"
              << "  --xy_path             [string] locations path.\n"
              << "  --obs_path            [string] observations path.\n"
              << "  --tol                 [int] tolerance of BOBYQA, 5 -> 1e-5.\n"
              << "  --t_slots             [int] The slot of time, e.g., 10, 20, satisfying num_loc % t_slots = 0.\n"
              << "  --omp_threads         [int] number openmp threads, default 40.\n"
              << "  --permutation         [string] reordering method, default as random, (optional) kdtree, morton, hilber, mmd.\n"
              // Add more options as necessary
              << std::endl;
}

/***************************************************************************/ /**
                                                                               * Macros to handle error checking.
                                                                               */

#define TESTING_CHECK(err)                                                  \
    do                                                                      \
    {                                                                       \
        magma_int_t err_ = (err);                                           \
        if (err_ != 0)                                                      \
        {                                                                   \
            fprintf(stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                    #err, __FILE__, __LINE__,                               \
                    (long long)err_, magma_strerror(err_));                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

typedef struct llh_data
{

    int cs;
    int num_loc;
    size_t batchCount;

    // MAGMA memory allocation in advanace (host)
    double *h_Cov, *h_obs_new;
    double *h_Cov_conditioning, *h_Cov_cross, *h_obs_conditioning;
    double **h_Cov_conditioning_array;
    double **h_Cov_cross_array;
    double **h_obs_conditioning_array_copy;

    // MAGMA memory allocation in advanace (device)
    double *d_Cov, *d_obs;
    double **d_Cov_array, **d_obs_array;
    double *d_Cov_conditioning, *d_Cov_cross, *d_obs_conditioning;
    double **d_Cov_conditioning_array, **d_Cov_cross_array, **d_obs_conditioning_array;
    double *d_obs_copy, *d_obs_conditioning_copy;
    double **d_obs_array_copy, **d_obs_conditioning_array_copy;
    double *d_Cov_offset, *d_mu_offset;
    double **d_Cov_offset_array, **d_mu_offset_array;
    double **h_Cov_array;
    double **h_obs_array_copy;
    double **h_mu_offset_array;
    double **h_Cov_offset_array;

    // MAGMA memory for batch size  related
    magma_int_t *h_lda, *d_lda, *h_ldda, *d_ldda;
    magma_int_t *h_ldacon, *d_ldacon, *h_lddacon, *d_lddacon;
    magma_int_t *d_const1;
    magma_int_t *d_batchNum;
    magma_int_t *batchNumAccum;
    magma_int_t *batchNum;
    magma_int_t *batchNumSquareAccum;
    int total_size_dev_obs;

    // MAGMA config
    magma_queue_t queue;
    magma_int_t *dinfo_magma;
    magma_int_t *hinfo_magma;

    // MAGMA operations flags
    char uplo;
    char transA;
    char transB;
    char side;
    char diag;
    int align; // gpu config, Round up LDDA on GPU to multiple of align, default 32

    // local theta for kernel in GPs
    double sigma;
    double beta;
    double nu;
    bool time_flag;
    double beta_time;
    double nu_time;
    double sep;
    double aux;
    int kernel;
    int num_params;
    int seed;

    // others
    location *locations_new;
    location *locations_con;
    double *norm2_result_h;
    double *logdet_result_h;
    int vecchia;
    int iterations;  // optimization
    int omp_threads; // openmp
    // double vecchia_time_total; // vecchia time monitoring
    int distance_metric; // 0 for euclidean; 1 for earth location. (real dataset)
    int perf;            // performance
    double *localtheta;
    double dist_scale;
} llh_data;

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct Vecchia_opts
    {

        int omp_numthreads;

        // real data path
        std::string xy_path;
        std::string obs_path;

        // local theta for kernel in GPs
        double sigma;
        double beta;
        double nu;
        double nugget;
        double dist_scale;
        // init
        double sigma_init;
        double beta_init;
        double nu_init;
        double nugget_init;
        // bivariate
        double sigma1;
        double sigma2;
        double alpha;
        double nu1;
        double nu2;
        // space-time
        double beta_time;
        double nu_time;
        double sep;
        double aux;
        bool time_flag;
        double beta_time_init;
        double nu_time_init;
        double sep_init;
        double aux_init;
        int t_slots;

        // performance test
        int perf;

        // vecchia
        int vecchia;
        int vecchia_cs;
        int vecchia_bc;

        // optimization
        double tol;
        int maxiter;
        double lower_bound;
        double upper_bound;

        // extra config
        int kernel;
        int num_params;
        int num_loc;
        int earth;

        // bivariate
        int p;

        // k nearest neighbors
        int knn;

        // ordering
        int randomordering;
        int mortonordering;
        int kdtreeordering;
        int hilbertordering;
        int mmdordering;

        // irregular grid
        int seed;

        // gpu config, Round up LDDA on GPU to multiple of align, default 32
        int align;
        int device;

        // BLAS operations
        magma_uplo_t uplo;
        magma_trans_t transA;
        magma_trans_t transB;
        magma_side_t side;
        magma_diag_t diag;

        // queue for default device
        magma_queue_t queue;
        magma_queue_t queues2[3]; // 2 queues + 1 extra NULL entry to catch errors

#ifdef MAGMA_HAVE_CUDA
        // handle for directly calling cublas
        cublasHandle_t handle;
#elif defined(MAGMA_HAVE_HIP)
    hipblasHandle_t handle;
#endif
    } Vecchia_opts;

    int parse_opts(int argc, char **argv, Vecchia_opts *opts);

#ifdef __cplusplus
}
#endif

extern "C" int parse_opts(int argc, char **argv, Vecchia_opts *opts)
{

    //-------------------------------------//
    // --------- problem setting  --------- //
    //-------------------------------------//
    // //real dataset path
    // opts->xy_path;
    // opts->obs_path;

    // extra config
    // 1: univaraite or 2: bivariate 3: time-space
    opts->kernel = 1;
    opts->num_params = 3;
    opts->num_loc = 2000;
    opts->p = 1;
    opts->time_flag = 0;

    // local theta for kernel in GPs
    opts->sigma = 1.0;
    opts->beta = 0.1;
    opts->nu = 0.5;
    opts->nugget = 0.0;
    opts->dist_scale = 1.0;

    // local theta for kernel in GPs
    opts->sigma_init = 0.01;
    opts->beta_init = 0.01;
    opts->nu_init = 0.01;
    opts->nugget_init = 0.01;

    // bivariate
    opts->sigma1 = 0.1;
    opts->sigma2 = 0.1;
    opts->alpha = 0.1;
    opts->nu1 = 0.1;
    opts->nu2 = 0.1;

    // space-time
    opts->nu_time = 0.5;   // smoothness (0,1]
    opts->beta_time = 2.2; // scaling
    opts->sep = 0.5;       // non-seperatbility [0, 1]
    opts->aux = 0.;        // auxi para [0, +inf]
    opts->nu_time_init = 0.01;
    opts->beta_time_init = 0.01;
    opts->sep_init = 0.01;
    opts->aux_init = 0.;
    opts->t_slots = 0; // e.g. 10

    // k nearest neighbors
    opts->knn = 1;

    // random ordering
    opts->randomordering = 1;
    opts->mortonordering = 0;
    opts->kdtreeordering = 0;
    opts->hilbertordering = 0;
    opts->mmdordering = 0;

    // irregular locations generation
    opts->seed = 0;

    // performance test
    opts->perf = 0;

    // vecchia conditioning
    opts->vecchia = 1;
    opts->vecchia_cs = 100;
    opts->vecchia_bc = 200;

    //-------------------------------------//
    // ----- BOBYQA optimization config  ---- //
    //-------------------------------------//
    // optimization setting
    opts->tol = 1e-5;
    opts->maxiter = 1000;
    opts->lower_bound = 0.01;
    opts->upper_bound = 3.;

    //-------------------------------------//
    // ----- host and device setting  ---- //
    //-------------------------------------//
    // openmp
    opts->omp_numthreads = 40;

    // gpu config, Round up LDDA on GPU to multiple of align, default 32
    opts->align = 32;
    opts->device = 0; // default is 0,
    opts->earth = 0;

    // BLAS
    opts->uplo = MagmaLower;
    opts->side = MagmaLeft;
    opts->transA = MagmaNoTrans;
    opts->diag = MagmaNonUnit;

    int info;

    for (int i = 1; i < argc; ++i)
    {
        //-------------------------------------//
        // --------- problem setting  --------- //
        //-------------------------------------//
        // real dataset input
        if (strcmp(argv[i], "--xy_path") == 0 && i + 1 < argc)
        {
            i++;
            opts->xy_path = argv[i]; // The next argument is the path
            std::cout << "xy_path: " << opts->xy_path << std::endl;
        }
        // real dataset input
        else if (strcmp(argv[i], "--obs_path") == 0 && i + 1 < argc)
        {
            i++;
            opts->obs_path = argv[i]; // The next argument is the path
            std::cout << "obs_path: " << opts->obs_path << std::endl;
        }
        // num_loc: number of locations
        else if (strcmp("--num_loc", argv[i]) == 0 && i + 1 < argc)
        {
            i++;
            int num_loc;
            info = sscanf(argv[i], "%d", &num_loc);
            if (info == 1 && num_loc > 0)
                opts->num_loc = num_loc;
            else
            {
                fprintf(stderr, "error: --num_loc %s is invalid; ensure num_loc > 0, info=%d, num_loc=%d.\n",
                        argv[i], info, num_loc);
                exit(1);
            }
        }
        // dist_scale
        else if ((strcmp("--dist_scale", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            double dist_scale;
            info = sscanf(argv[i], "%lf", &dist_scale);
            if (info == 1 && dist_scale > 0)
                opts->dist_scale = dist_scale;
            else
            {
                fprintf(stderr, "error: --dist_scale %s is invalid; ensure dist_scale > 0.\n", argv[i]);
                exit(1);
            }
        }
        // conditioning size
        else if ((strcmp("--vecchia_cs", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int num;
            info = sscanf(argv[i], "%d", &num);
            if (info == 1 && num >= 0)
            {
                opts->vecchia_cs = num;
                opts->vecchia = 1;
            }
            else
            {
                fprintf(stderr, "error: --vecchia_cs %s is invalid; ensure only one number and 0 <= vecchia_cs <= N.\n", argv[i]);
                exit(1);
            }
        }
        // block count / batchcount
        else if ((strcmp("--vecchia_bc", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int num;
            info = sscanf(argv[i], "%d", &num);
            if (info == 1 && num > 0)
                opts->vecchia_bc = num;
            else
            {
                fprintf(stderr, "error: --vecchia_bc %s is invalid; ensure only one number and 0 < vecchia_bc <= N.\n", argv[i]);
                exit(1);
            }
        }
        // used for performance test
        else if (strcmp("--perf", argv[i]) == 0 && i + 1 < argc)
        {
            opts->perf = 1;
            opts->maxiter = 1;
        }
        // k nearest neighbors
        else if (strcmp("--knn", argv[i]) == 0 && i + 1 < argc)
        {
            opts->knn = 1;
        }
        // 3D spatial
        else if (strcmp("--3D", argv[i]) == 0 && i + 1 < argc)
        {
            opts->time_flag = 1;
            opts->t_slots = 0;
        }
        // earth distance
        else if (strcmp("--earth", argv[i]) == 0 && i + 1 < argc)
        {
            opts->earth = 1;
        }
        // ordering
        else if (strcmp("--permutation", argv[i]) == 0 && i + 1 < argc)
        {
            i++;
            std::string reordering = argv[i];
            if (reordering == "random")
                opts->randomordering = 1;
            else
            {
                opts->randomordering = 0;
                if (reordering == "morton")
                    opts->mortonordering = 1;
                if (reordering == "kdtree")
                    opts->kdtreeordering = 1;
                if (reordering == "hilbert")
                    opts->hilbertordering = 1;
                if (reordering == "mmd")
                    opts->mmdordering = 1;
            }
            int _sum_ordering = opts->randomordering + opts->mortonordering + opts->kdtreeordering + opts->hilbertordering + opts->mmdordering;
            if (_sum_ordering > 1)
            {
                std::cout << "Please only contain one permutation methods, you are containing " << _sum_ordering << std::endl;
                exit(0);
            }
        }
        // kernels
        else if ((strcmp("--kernel", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            char *kernel_str = argv[i];

            if (strcmp(kernel_str, "univariate_matern_stationary_no_nugget") == 0)
            {
                fprintf(stderr, "You are using the Matern Kernel (sigma^2, range, smooth)!\n");
                opts->kernel = 1;     // You can change this value as needed
                opts->num_params = 3; // Set appropriate values for the 'matern' kernel
                opts->p = 1;          // You can modify this as per the requirement for 'matern'
            }
            else if (strcmp(kernel_str, "univariate_powexp_stationary_no_nugget") == 0)
            {
                fprintf(stderr, "You are using the Power exponential Kernel (sigma^2, range, smooth)!\n");
                opts->kernel = 2;     // Change as per your requirement for 'powexp'
                opts->num_params = 3; // Set appropriate values for the 'powexp' kernel
                opts->p = 1;          // Modify as needed for 'powexp'
            }
            else if (strcmp(kernel_str, "univariate_powexp_nugget_stationary") == 0)
            {
                fprintf(stderr, "You are using the Power exponential Kernel with nugget (sigma^2, range, smooth, nugget)!\n");
                opts->kernel = 3;     //
                opts->num_params = 4; //
                opts->p = 1;          // Modify as needed for 'powexp'
            }
            else if (strcmp(kernel_str, "univariate_matern_spacetime_stationary") == 0)
            {
                fprintf(stderr, "You are using the space-time Kernel (sigma^2, range_spatial, smooth_spatial, range_time, smooth_time, seperability, aux)!\n");
                opts->kernel = 4;     //
                opts->num_params = 7; // auxi = 0;
                opts->p = 1;          // Modify as needed for 'powexp'
                opts->time_flag = 1;
            }
            else
            {
                fprintf(stderr, "Unsupported kernel type: %s\n", kernel_str);
                exit(1);
            }
        }
        // conditioning size
        else if ((strcmp("--t_slots", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int num;
            info = sscanf(argv[i], "%d", &num);
            if (info == 1 && num >= 0)
                opts->t_slots = num;
        }
        // ture parameters
        else if (strcmp("--ikernel", argv[i]) == 0 && i + 1 < argc)
        {
            i++;
            double a1 = -1, a2 = -1, a3 = -1, a4 = -1, a5 = -1, a6 = -1; // Initialize with default values indicating 'unknown'
            char s1[10], s2[10], s3[10], s4[10], s5[10], s6[10];         // Arrays to hold the string representations
            // Parse the input into string buffers
            int info = sscanf(argv[i], "%9[^:]:%9[^:]:%9[^:]:%9[^:]:%9[^:]:%9[^:]", s1, s2, s3, s4, s5, s6);
            if (info < 3 || info > 6)
            {
                printf("Other kernels have been developing on the way!");
                exit(0);
            }
            // Check and convert each value
            if (strcmp(s1, "?") != 0)
                a1 = atof(s1);
            if (strcmp(s2, "?") != 0)
                a2 = atof(s2);
            if (strcmp(s3, "?") != 0)
                a3 = atof(s3);
            if (strcmp(s4, "?") != 0)
                a4 = atof(s4);
            if (info == 6)
            {
                if (strcmp(s5, "?") != 0)
                    a5 = atof(s5);
                if (strcmp(s6, "?") != 0)
                    a6 = atof(s6);
            }
            // Assign values to opts if they are not unknown
            if (a1 != -1)
                opts->sigma = a1;
            if (a2 != -1)
                opts->beta = a2;
            if (a3 != -1)
                opts->nu = a3;
            if (a4 != -1)
                opts->nugget = a4;
            if (info == 6)
            {
                if (a4 != -1)
                    opts->beta_time = a4;
                if (a5 != -1)
                    opts->nu_time = a5;
                if (a6 != -1)
                    opts->sep = a6;
            }
        }
        // initi parameters
        else if (strcmp("--kernel_init", argv[i]) == 0 && i + 1 < argc)
        {
            i++;
            double a1 = -1, a2 = -1, a3 = -1, a4 = -1, a5 = -1, a6 = -1; 
            char s1[10], s2[10], s3[10], s4[10], s5[10], s6[10];         
            int info = sscanf(argv[i], "%9[^:]:%9[^:]:%9[^:]:%9[^:]:%9[^:]:%9[^:]", s1, s2, s3, s4, s5, s6);
            // Check and convert each value
            if (strcmp(s1, "?") != 0)
                a1 = atof(s1);
            if (strcmp(s2, "?") != 0)
                a2 = atof(s2);
            if (strcmp(s3, "?") != 0)
                a3 = atof(s3);
            if (strcmp(s4, "?") != 0)
                a4 = atof(s4);
            if (info == 6)
            {
                if (strcmp(s5, "?") != 0)
                    a5 = atof(s5);
                if (strcmp(s6, "?") != 0)
                    a6 = atof(s6);
            }
            // Assign values to opts if they are not unknown
            if (a1 != -1)
                opts->sigma_init = a1;
            if (a2 != -1)
                opts->beta_init = a2;
            if (a3 != -1)
                opts->nu_init = a3;
            if (a4 != -1)
                opts->nugget_init = a4;
            if (info == 6)
            {
                if (a4 != -1)
                    opts->beta_time_init = a4;
                if (a5 != -1)
                    opts->nu_time_init = a5;
                if (a6 != -1)
                    opts->sep_init = a6;
            }
        }
        // irregular locations generation seeds
        else if ((strcmp("--seed", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int seed;
            info = sscanf(argv[i], "%d", &seed);
            opts->seed = seed;
        }

        //-------------------------------------//
        // ----- host and device setting  ---- //
        //-------------------------------------//
        else if (strcmp("--omp_threads", argv[i]) == 0 && i + 1 < argc)
        {
            opts->omp_numthreads = atoi(argv[++i]);
        }

        //-------------------------------------//
        // ----- BOBYQA optimization config  ---- //
        //-------------------------------------//
        else if ((strcmp("--maxiter", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int maxiter;
            info = sscanf(argv[i], "%d", &maxiter);
            if (info == 1 && maxiter > 0)
            {
                opts->maxiter = maxiter;
            }
            else
            {
                fprintf(stderr, "error: --maxiter %s is invalid; ensure maxiter > 0 and be integer.\n", argv[i]);
                exit(1);
            }
        }
        else if ((strcmp("--tol", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            int tol;
            info = sscanf(argv[i], "%d", &tol);
            if (info == 1 && tol > 0)
            {
                opts->tol = pow(10, -tol);
            }
            else
            {
                fprintf(stderr, "error: --tol %s is invalid; ensure tol > 0.\n", argv[i]);
                exit(1);
            }
        }
        else if ((strcmp("--lower_bound", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            double lower_bound;
            info = sscanf(argv[i], "%lf", &lower_bound);
            if (info == 1 && lower_bound > 0)
            {
                opts->lower_bound = lower_bound;
            }
            else
            {
                fprintf(stderr, "error: --lower_bound %s is invalid; ensure lower_bound > 0.\n", argv[i]);
                exit(1);
            }
        }
        else if ((strcmp("--upper_bound", argv[i]) == 0) && i + 1 < argc)
        {
            i++;
            double upper_bound;
            info = sscanf(argv[i], "%lf", &upper_bound);
            if (info == 1 && upper_bound < 100)
            {
                opts->upper_bound = upper_bound;
            }
            else
            {
                fprintf(stderr, "error: --upper_bound %s is invalid; ensure upper_bound < 100. (Or you fix 100 in opts file)\n", argv[i]);
                exit(1);
            }
        }

        // ----- usage
        else if (strcmp("-h", argv[i]) == 0 || strcmp("--help", argv[i]) == 0)
        {
            displayHelp();
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unrecognized option %s\n", argv[i]);
            exit(1);
        }
    }

// magma queuex
#if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_HIP)
    magma_setdevice(opts->device);
#endif

    // create queues on this device
    // 2 queues + 1 extra NULL entry to catch errors
    magma_queue_create(opts->device, &opts->queues2[0]);
    magma_queue_create(opts->device, &opts->queues2[1]);
    opts->queues2[2] = NULL;

    opts->queue = opts->queues2[0];

#if defined(MAGMA_HAVE_HIP)
    // handle for directly calling hipblas
    opts->handle = magma_queue_get_hipblas_handle(opts->queue);
#elif defined(MAGMA_HAVE_CUDA)
    // handle for directly calling cublas
    opts->handle = magma_queue_get_cublas_handle(opts->queue);
#else
#error "unknown platform"
#endif

    return 1;
} // end parse_opts

#endif /* VECCHIA_HELPER_H */
