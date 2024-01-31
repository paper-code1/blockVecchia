/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file 

 * 
 *
 *
 * @version 0.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <ctime>
#include <nlopt.h>
#include <vector>
#include <gsl/gsl_errno.h>
#include <typeinfo>

// magma helper file
#include <magma_v2.h>

// // KBLAS helper
// #include "testing_helper.h"
// #include "Xblas_core.ch"

// Used for llh
#include "../include/cluster.h"
#include "../include/ckernel.h"
#include "../include/vecchia_helper_method.h"
#include "../include/flops.h"
// used for vecchia
extern "C"
{
#include "../include/vecchia_helper_data.h"
}
// used for nearest neighbor
#include "../include/nearest_neighbor.h"
// this is not formal statement and clarification, only for convenience
#include "../include/utils.h"
#include "../include/llh_Xvecchia_batch.h"

template <class T>
int test_Xvecchia_batch(Vecchia_opts &opts, T alpha)
{
    TESTING_CHECK(magma_init());
    magma_print_environment();

    // first to create a folder to save the log information
    createLogFile(opts);

    llh_data data;
    location *locations;
    location *locations_con;
    real_Double_t gflops, gpu_perf, gpu_time;
    long long total_size_cpu_Cov, total_size_dev_Cov;
    long long total_size_cpu_obs, total_size_dev_obs;
    magma_int_t info;
    magma_int_t *dinfo_magma;
    magma_int_t *hinfo_magma;
    magma_int_t *h_const1;
    magma_int_t *d_const1;
    magma_int_t *d_batchNum;
    magma_int_t *h_lda, *d_lda, *h_ldda, *d_ldda;
    T *h_Cov, *d_Cov, *h_obs, *d_obs, *d_obs_copy;
    T **h_Cov_array = NULL, **d_Cov_array = NULL;
    T **h_obs_array = NULL, **d_obs_array = NULL;
    T **h_obs_array_copy = NULL, **d_obs_array_copy = NULL;
    // intermidiate results
    T *dot_result_h;
    T *logdet_result_h;
    double *h_obs_tmp, *d_obs_tmp;

    // vecchia
    magma_int_t *h_ldacon, *d_ldacon, *h_lddacon, *d_lddacon;
    double *h_obs_conditioning_tmp, *d_obs_conditioning_tmp;
    // conditioning matrix and vector
    // copy: modification element in place
    T *h_Cov_conditioning, *h_Cov_cross, *h_obs_conditioning;
    T *d_Cov_conditioning, *d_Cov_cross, *d_obs_conditioning, *d_obs_conditioning_copy;
    T *d_Cov_offset, *d_mu_offset;
    T **h_mu_offset_array = NULL, **d_mu_offset_array = NULL;
    T **h_Cov_offset_array = NULL, **d_Cov_offset_array = NULL;
    T **h_Cov_conditioning_array = NULL, **d_Cov_conditioning_array = NULL;
    T **h_Cov_cross_array = NULL, **d_Cov_cross_array = NULL;
    T **h_obs_conditioning_array = NULL, **d_obs_conditioning_array = NULL;
    T **h_obs_conditioning_array_copy = NULL, **d_obs_conditioning_array_copy = NULL;

    //
    long long nclusters;
    if (opts.num_loc < 2 * opts.vecchia_bc){
        // classic vecchia 
        nclusters = opts.num_loc;
    }else{
        // cluster vecchia 
        nclusters = opts.vecchia_bc;
    }
    long long batchCount;
    int *batchIndex, *batchNum, *batchNumAccum, *batchNumSquareAccum;
    int *clusterid, *clusterNum, *max_num_cluster;
    double **centroid;
    int cs = opts.vecchia_cs;
    // Set initial guess
    double localtheta_initial[opts.num_params]; // Initial guess
    double lb[opts.num_params];
    double ub[opts.num_params];

    int *firstClusterSize = (int *)calloc(1, sizeof(int));
    int *firstClusterCount = (int *)calloc(1, sizeof(int));
    int *permIndex = (int *)malloc(nclusters * sizeof(int));
    location *locsCentroid = (location *)malloc(sizeof(location));
    // Allocate memory
    locsCentroid->x = (double *)malloc(nclusters * sizeof(double));
    locsCentroid->y = (double *)malloc(nclusters * sizeof(double));
    locsCentroid->z = NULL;

    // new locations/obs after clustering
    // easier for later MAGMA BLAS operations
    T *h_obs_new = (T *)malloc(opts.num_loc * sizeof(T));
    location *locations_new = (location *)malloc(sizeof(location));
    locations_new->x = (double *)malloc(opts.num_loc * sizeof(double));
    locations_new->y = (double *)malloc(opts.num_loc * sizeof(double));
    locations_new->z = NULL;

    TESTING_CHECK(magma_dmalloc_cpu(&h_obs, opts.num_loc));

    //-------------------------------------------------------------//
    //------------------k-means for partition-----------------------//
    //-------------------------------------------------------------//

    // performance test
    if (opts.perf == 1)
    {
        locations = GenerateXYLoc(opts.num_loc, opts.seed); // 0 is the random seed
        // for(int i = 0; i < opts.num_loc; i++) h_obs[i] = (T) rand()/(T)RAND_MAX;
        for (int i = 0; i < opts.num_loc; ++i)
            h_obs[i] = 0.0;
        // printLocations(opts.num_loc, locations);
        // for(int i = 0; i < Cm * nclusters; i++) printf("%ith %lf \n",i, h_obs[i]);
        // 1/2: matern/powexp kernel
        // 3: matern/powexp kernel with nugget
        if (opts.kernel == 1 || opts.kernel == 2)
        {
            localtheta_initial[0] = opts.sigma;
            localtheta_initial[1] = opts.beta;
            localtheta_initial[2] = opts.nu;
        }
        else if (opts.kernel == 3)
        {
            localtheta_initial[0] = opts.sigma;
            localtheta_initial[1] = opts.beta;
            localtheta_initial[2] = opts.nu;
            localtheta_initial[3] = opts.nugget;
        }
        data.distance_metric = 0;
    }
    else
    {
        /*used for the simulated data and real applications*/
        std::string xy_path = "./trash/synthetic_ds/LOC_640_1";
        std::string z_path = "./trash/synthetic_ds/Z_640_1";
        /*simulations*/
        // std::string xy_path = "./simu_ds/20ks_" + std::to_string(opts.beta) + "_" + std::to_string(opts.nu) + "/LOC_20000_univariate_matern_stationary_" + std::to_string(opts.seed);
        // std::string z_path = "./simu_ds/20ks_" + std::to_string(opts.beta) + "_" + std::to_string(opts.nu) + "/Z1_20000_univariate_matern_stationary_" + std::to_string(opts.seed);
        data.distance_metric = 0;
        locations = loadXYcsv(xy_path, opts.num_loc);
        loadObscsv<T>(z_path, opts.num_loc, h_obs);
        if (opts.kernel == 1 || opts.kernel == 2)
        {
            localtheta_initial[0] = opts.sigma_init;
            localtheta_initial[1] = opts.beta_init;
            localtheta_initial[2] = opts.nu_init;
        }
        else if (opts.kernel == 3)
        {
            localtheta_initial[0] = opts.sigma_init;
            localtheta_initial[1] = opts.beta_init;
            localtheta_initial[2] = opts.nu_init;
            localtheta_initial[3] = opts.nugget_init;
        }
    }

    // locations: 2D spatial location
    // h_obs: obeservations for each location
    // input: locations, nclusters
    // output: clusters
    clusterid = (int *)malloc(opts.num_loc * sizeof(int));
    clusterNum = (int *)calloc(nclusters, sizeof(int));
    max_num_cluster = (int *)calloc(1, sizeof(int));
    centroid = (double **)malloc(nclusters * sizeof(double *));
    for (int i = 0; i < nclusters; ++i)
        centroid[i] = (double *)malloc(2 * sizeof(double));
    
    if (opts.num_loc < 2 * opts.vecchia_bc)
    {
        // classic vecchia
        fprintf(stderr, "You are using the classic Vecchia!\n");
        for (int i = 0; i < opts.num_loc; i++)
        {
            clusterid[i] = i;
            clusterNum[i] = 1;
            max_num_cluster[0] = 1;
            centroid[i][0] = locations->x[i];
            centroid[i][1] = locations->y[i];
        }
    }
    else
    {
        // cluster vecchia
        fprintf(stderr, "You are using the cluster Vecchia!\n");
        clustering_2D(nclusters, opts.num_loc,
                      locations, clusterid, centroid,
                      clusterNum, max_num_cluster, opts.seed);
    }
    fprintf(stderr, "--------------Clustering Done-----------------\n");
    //-------------------------------------------------------------//
    //------------------ Clusters Reordering -----------------------//
    //-------------------------------------------------------------//
    for (int i = 0; i < nclusters; ++i)
    {
        locsCentroid->x[i] = centroid[i][0];
        locsCentroid->y[i] = centroid[i][1];
        permIndex[i] = i;
    }

    // Ordering for locations and observations
    if (opts.randomordering == 1)
    {
        random_reordering(nclusters, locsCentroid);
        fprintf(stderr, "You were using the Random ordering. \n");
    }
    else if (opts.mortonordering == 1)
    {
        zsort_reordering(nclusters, locsCentroid);
        fprintf(stderr, "You were using the Morton ordering. \n");
    }
    else if (opts.kdtreeordering == 1)
    {
        zsort_locations_kdtree(nclusters, locsCentroid);
        fprintf(stderr, "You were using the KDtree ordering. \n");
    }
    else if (opts.hilbertordering == 1)
    {
        zsort_locations_hilbert(nclusters, locsCentroid);
        fprintf(stderr, "You were using the Hilbert ordering. \n");
    }
    // else if (opts.mmdordering == 1)
    // {
    //     zsort_locations_mmd(nclusters, locations);
    //     fprintf(stderr, "You were using the MMD ordering. \n");
    // }
    else
    {
        fprintf(stderr, "Other ordering is being developed. \n");
        exit(0);
    }

    // find the reordered index
    // array of permIndex stores the reordered index, such as permIndex = 12, 3, 4,...
    reorderIndex(locsCentroid, centroid, permIndex, nclusters);

    // the first batch will calcuate full llh as a whole
    // the locations and clusters are reconstructed
    firstbatch(nclusters, clusterNum, permIndex, firstClusterSize, firstClusterCount, cs);
    assert(firstClusterSize[0] > 0 && firstClusterCount[0] > 0);
    batchCount = nclusters - firstClusterCount[0] + 1;
    cluster_combine(firstClusterCount, firstClusterSize, opts.num_loc, clusterid, permIndex, clusterNum);

    // (accmulated) number of batch
    batchIndex = permIndex + firstClusterCount[0] - 1;
    batchNum = (int *)calloc(batchCount + 1, sizeof(int));
    batchNumAccum = (int *)calloc(batchCount + 1, sizeof(int));       // starting from 0
    batchNumSquareAccum = (int *)calloc(batchCount + 1, sizeof(int)); // starting from 0
    // conditioning locations
    locations_con = (location *)malloc(sizeof(location));
    locations_con->x = (double *)malloc(batchCount * cs * sizeof(double));
    locations_con->y = (double *)malloc(batchCount * cs * sizeof(double));
    locations_con->z = NULL;
    h_obs_conditioning = (T *)malloc(batchCount * cs * sizeof(T));
    for (int i = 0; i < batchCount; ++i)
        batchNum[i] = clusterNum[permIndex[i + firstClusterCount[0] - 1]];
    for (int i = 1; i < (batchCount + 1); ++i)
    {
        batchNumAccum[i] += batchNumAccum[i - 1] + batchNum[i - 1];
        batchNumSquareAccum[i] += batchNumSquareAccum[i - 1] + batchNum[i - 1] * batchNum[i - 1];
    }
    assert(batchNumAccum[batchCount] == opts.num_loc);
    cluster_to_batch(opts.num_loc, batchCount, batchNum, batchNumAccum, batchIndex, locations, h_obs, locations_new, h_obs_new, clusterid, locsCentroid);
    // clusterid will be not used anymore, it has been change in place

    // update
    if (max_num_cluster[0] < firstClusterSize[0])
        max_num_cluster[0] = firstClusterSize[0];
    fprintf(stderr, "--------------Reordering Done-----------------\n");

    //-------------------------------------------------------------//
    //------------------Nearest neighbor searching-----------------------//
    //-------------------------------------------------------------//
    // copy for the first independent block (no meaning), random assign
    memcpy(h_obs_conditioning, h_obs_new, sizeof(T) * cs);
    memcpy(locations_con->x, locations_new->x, sizeof(T) * cs);
    memcpy(locations_con->y, locations_new->y, sizeof(T) * cs);

    if (opts.knn)
    {
#pragma omp parallel for
        for (int i = 1; i < batchCount; ++i)
        {
            // how many previous points you would like to include in your nearest neighbor searching
            // int starting_loc = std::max(i * bs - 10000 * bs, 0);
            int starting_loc = 0;
            findNearestPoints(
                h_obs_conditioning, locations_con,
                locsCentroid, firstClusterCount,
                h_obs_new, locations_new,
                starting_loc, batchNumAccum[i],
                cs, i, data.distance_metric);
            // printLocations(cs * (i+1), locations_con);
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 1; i < batchCount; ++i)
        {
            memcpy(locations_con->x + i * cs, locations_new->x + batchNumAccum[i] - cs, sizeof(T) * cs);
            memcpy(locations_con->y + i * cs, locations_new->y + batchNumAccum[i] - cs, sizeof(T) * cs);
            memcpy(h_obs_conditioning + i * cs, h_obs_new + batchNumAccum[i] - cs, sizeof(T) * cs);
        }
    }
    // printLocations(opts.num_loc, locations);
    // printLocations(cs * batchCount, locations_con);

    fprintf(stderr, "--------------Nearest Neighbor Done-----------------\n");
    //-------------------------------------------------------------//
    //------------------ memory set/get -----------------------//
    //-------------------------------------------------------------//
    TESTING_CHECK(magma_imalloc_cpu(&h_ldda, batchCount));
    TESTING_CHECK(magma_imalloc_cpu(&hinfo_magma, batchCount));
    TESTING_CHECK(magma_imalloc_cpu(&h_const1, batchCount));
    TESTING_CHECK(magma_imalloc(&d_batchNum, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&dinfo_magma, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&d_const1, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&d_ldda, batchCount + 1));
    TESTING_CHECK(magma_imalloc(&d_lda, batchCount + 1));
    for (int i = 0; i < batchCount; ++i)
    {
        h_const1[i] = 1;
        hinfo_magma[i] = 0; // check the success of each batch 0: success 1: failure
    }

    // batch sizes configuration and the memory allocations
    h_lda = batchNum;
    total_size_cpu_Cov = total_size_dev_Cov = 0;
    total_size_cpu_obs = total_size_dev_obs = 0;
    gflops = 0;
    for (int k = 0; k < batchCount; k++)
    {
        h_ldda[k] = magma_roundup(batchNum[k], opts.align); // multiple of 32 by default
        total_size_cpu_Cov += batchNum[k] * h_lda[k];
        total_size_dev_Cov += batchNum[k] * h_ldda[k];
        total_size_cpu_obs += h_lda[k];
        total_size_dev_obs += h_ldda[k];
        gflops += FLOPS_DPOTRF(batchNum[k]) / 1e9;
        gflops += FLOPS_DTRSM(opts.side, batchNum[k], 1) / 1e9;
    }
    TESTING_CHECK(magma_dmalloc_cpu(&h_Cov, total_size_cpu_Cov));
    TESTING_CHECK(magma_dmalloc(&d_Cov, total_size_dev_Cov));
    TESTING_CHECK(magma_dmalloc(&d_obs, total_size_dev_obs));
    TESTING_CHECK(magma_dmalloc(&d_obs_copy, total_size_dev_obs));

    // the *_array (pointer to pointer)
    // is the same for each iteration in the log-likleihod
    TESTING_CHECK(magma_malloc_cpu((void **)&h_Cov_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc_cpu((void **)&h_obs_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc_cpu((void **)&h_obs_array_copy, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&d_Cov_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&d_obs_array, batchCount * sizeof(double *)));
    TESTING_CHECK(magma_malloc((void **)&d_obs_array_copy, batchCount * sizeof(double *)));
    h_Cov_array[0] = d_Cov;
    h_obs_array[0] = d_obs;
    h_obs_array_copy[0] = d_obs_copy;
    for (int i = 1; i < batchCount; i++)
    {
        h_Cov_array[i] = h_Cov_array[i - 1] + h_ldda[i - 1] * batchNum[i - 1];
        h_obs_array[i] = h_obs_array[i - 1] + h_ldda[i - 1];
        h_obs_array_copy[i] = h_obs_array_copy[i - 1] + h_ldda[i - 1];
    }
    h_obs_tmp = h_obs;
    d_obs_tmp = d_obs;
    for (int i = 0; i < batchCount; i++)
    {
        magma_dsetvector(batchNum[i],
                         h_obs_tmp, 1,
                         d_obs_tmp, 1,
                         opts.queue);
        h_obs_tmp += h_lda[i];
        d_obs_tmp += h_ldda[i];
    }
    magma_setvector(batchCount, sizeof(double *), h_Cov_array, 1, d_Cov_array, 1, opts.queue);
    magma_setvector(batchCount, sizeof(double *), h_obs_array, 1, d_obs_array, 1, opts.queue);
    magma_setvector(batchCount, sizeof(double *), h_obs_array_copy, 1, d_obs_array_copy, 1, opts.queue);
    magma_setvector(batchCount, sizeof(magma_int_t), batchNum, 1, d_batchNum, 1, opts.queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_lda, 1, d_lda, 1, opts.queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_ldda, 1, d_ldda, 1, opts.queue);
    magma_setvector(batchCount, sizeof(magma_int_t), h_const1, 1, d_const1, 1, opts.queue);

    if (opts.vecchia && opts.vecchia_cs > 0)
    {
        magma_imalloc_cpu(&h_ldacon, batchCount);
        magma_imalloc_cpu(&h_lddacon, batchCount);
        magma_imalloc(&d_ldacon, batchCount + 1);
        magma_imalloc(&d_lddacon, batchCount + 1);
        for (int i = 0; i < batchCount; ++i)
            h_ldacon[i] = cs;
        magma_setvector(batchCount, sizeof(magma_int_t), h_ldacon, 1, d_ldacon, 1, opts.queue);
        for (int i = 0; i < batchCount; ++i)
            h_lddacon[i] = magma_roundup(cs, opts.align);
        magma_setvector(batchCount, sizeof(magma_int_t), h_lddacon, 1, d_lddacon, 1, opts.queue);

        TESTING_CHECK(magma_dmalloc_cpu(&h_Cov_conditioning, cs * cs * batchCount));
        TESTING_CHECK(magma_dmalloc_cpu(&h_Cov_cross, cs * opts.num_loc));
        TESTING_CHECK(magma_dmalloc(&d_Cov_conditioning, h_lddacon[0] * cs * batchCount)); // h_lddacon[0] same along batch
        TESTING_CHECK(magma_dmalloc(&d_Cov_cross, h_lddacon[0] * opts.num_loc));
        TESTING_CHECK(magma_dmalloc(&d_Cov_offset, total_size_dev_Cov));
        TESTING_CHECK(magma_dmalloc(&d_mu_offset, total_size_dev_obs));
        TESTING_CHECK(magma_dmalloc(&d_obs_conditioning, h_lddacon[0] * batchCount));
        TESTING_CHECK(magma_dmalloc(&d_obs_conditioning_copy, h_lddacon[0] * batchCount));

        TESTING_CHECK(magma_malloc_cpu((void **)&h_Cov_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&h_Cov_cross_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&h_Cov_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&h_mu_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&h_obs_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc_cpu((void **)&h_obs_conditioning_array_copy, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_Cov_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_Cov_cross_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_Cov_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_mu_offset_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_obs_conditioning_array, batchCount * sizeof(double *)));
        TESTING_CHECK(magma_malloc((void **)&d_obs_conditioning_array_copy, batchCount * sizeof(double *)));

        h_Cov_conditioning_array[0] = d_Cov_conditioning;
        h_Cov_cross_array[0] = d_Cov_cross;
        h_Cov_offset_array[0] = d_Cov_offset;
        h_mu_offset_array[0] = d_mu_offset;
        h_obs_conditioning_array[0] = d_obs_conditioning;
        h_obs_conditioning_array_copy[0] = d_obs_conditioning_copy;
        for (int i = 1; i < batchCount; i++)
        {
            h_Cov_conditioning_array[i] = h_Cov_conditioning_array[i - 1] + h_lddacon[i - 1] * cs;
            h_Cov_cross_array[i] = h_Cov_cross_array[i - 1] + h_lddacon[i - 1] * batchNum[i - 1];
            h_Cov_offset_array[i] = h_Cov_offset_array[i - 1] + h_ldda[i - 1] * batchNum[i - 1];
            h_mu_offset_array[i] = h_mu_offset_array[i - 1] + h_ldda[i - 1];
            h_obs_conditioning_array[i] = h_obs_conditioning_array[i - 1] + h_lddacon[i - 1];
            h_obs_conditioning_array_copy[i] = h_obs_conditioning_array_copy[i - 1] + h_lddacon[i - 1];
        }
        h_obs_conditioning_tmp = h_obs_conditioning;
        d_obs_conditioning_tmp = d_obs_conditioning;
        for (int i = 0; i < batchCount; i++)
        {
            magma_dsetvector(cs,
                             h_obs_conditioning_tmp, 1,
                             d_obs_conditioning_tmp, 1,
                             opts.queue);
            h_obs_conditioning_tmp += h_ldacon[i];
            d_obs_conditioning_tmp += h_lddacon[i];
        }
        magma_setvector(batchCount, sizeof(double *), h_Cov_conditioning_array, 1, d_Cov_conditioning_array, 1, opts.queue);
        magma_setvector(batchCount, sizeof(double *), h_Cov_cross_array, 1, d_Cov_cross_array, 1, opts.queue);
        magma_setvector(batchCount, sizeof(double *), h_Cov_offset_array, 1, d_Cov_offset_array, 1, opts.queue);
        magma_setvector(batchCount, sizeof(double *), h_mu_offset_array, 1, d_mu_offset_array, 1, opts.queue);
        magma_setvector(batchCount, sizeof(double *), h_obs_conditioning_array, 1, d_obs_conditioning_array, 1, opts.queue);
        magma_setvector(batchCount, sizeof(double *), h_obs_conditioning_array_copy, 1, d_obs_conditioning_array_copy, 1, opts.queue);
    }

    // intermidiate results
    TESTING_CHECK(magma_dmalloc_cpu(&dot_result_h, batchCount));
    TESTING_CHECK(magma_dmalloc_cpu(&logdet_result_h, batchCount));
    fprintf(stderr, "--------------Memory Allocate Done-----------------\n");
    //-------------------------------------------------------------//
    //------------------ llk data structure -----------------------//
    //-------------------------------------------------------------//

    // MAGMA memory allocation in advanace (host)
    data.h_Cov = h_Cov;
    data.h_obs = h_obs;
    data.h_ldda = h_ldda;
    data.h_lda = h_lda;
    data.h_obs_array_copy = h_obs_array_copy;
    data.h_mu_offset_array = h_mu_offset_array;
    data.h_Cov_offset_array = h_Cov_offset_array;
    data.h_Cov_cross_array = h_Cov_cross_array;
    data.h_obs_conditioning_array_copy = h_obs_conditioning_array_copy;
    // MAGMA memory allocation in advanace (device)
    data.d_Cov = d_Cov;
    data.d_obs = d_obs;
    data.d_obs_copy = d_obs_copy;
    data.d_Cov_array = d_Cov_array;
    data.d_obs_array = d_obs_array;
    data.d_obs_array_copy = d_obs_array_copy;
    data.h_Cov_array = h_Cov_array;
    data.d_const1 = d_const1;
    data.d_ldda = d_ldda;
    data.d_lda = d_lda;
    data.d_batchNum = d_batchNum;
    // memory for batch size related
    data.batchNumAccum = batchNumAccum;
    data.batchNum = batchNum;
    data.batchNumSquareAccum = batchNumSquareAccum;
    data.num_loc = opts.num_loc;
    data.batchCount = batchCount;
    data.cs = cs;
    data.total_size_dev_obs = total_size_dev_obs;
    // covariance or vector size
    data.locations_new = locations_new;
    data.logdet_result_h = logdet_result_h;
    data.dot_result_h = dot_result_h;
    // MAGMA config
    data.queue = opts.queue;
    data.hinfo_magma = hinfo_magma;
    data.dinfo_magma = dinfo_magma;
    data.max_num_cluster = max_num_cluster;
    // kernel related
    data.sigma = opts.sigma;
    data.beta = opts.beta;
    data.nu = opts.nu;
    data.iterations = 0; // starting iterations
    data.omp_threads = opts.omp_numthreads;
    data.kernel = opts.kernel;
    data.num_params = opts.num_params;
    data.seed = opts.seed;
    // data.vecchia_time_total = 0; // used for accumulatet the time on vecchia
    data.perf = opts.perf;

    if (opts.vecchia)
    {
        data.vecchia = opts.vecchia;
        data.locations_con = locations_con;
        // conditioning matrix and vector
        data.h_Cov_conditioning = h_Cov_conditioning;
        data.h_Cov_cross = h_Cov_cross;
        data.h_obs_conditioning = h_obs_conditioning;
        data.h_Cov_conditioning_array = h_Cov_conditioning_array;
        data.d_Cov_conditioning = d_Cov_conditioning;
        data.d_Cov_cross = d_Cov_cross;
        data.d_Cov_offset = d_Cov_offset;
        data.d_mu_offset = d_mu_offset;
        data.d_obs_conditioning = d_obs_conditioning;
        data.d_obs_conditioning_copy = d_obs_conditioning_copy;
        data.d_Cov_conditioning_array = d_Cov_conditioning_array;
        data.d_Cov_cross_array = d_Cov_cross_array;
        data.d_Cov_offset_array = d_Cov_offset_array;
        data.d_mu_offset_array = d_mu_offset_array;
        data.d_obs_conditioning_array = d_obs_conditioning_array;
        data.d_obs_conditioning_array_copy = d_obs_conditioning_array_copy;

        // conditioning size
        data.h_ldacon = h_ldacon;
        data.d_ldacon = d_ldacon;
        data.h_lddacon = h_lddacon;
        data.d_lddacon = d_lddacon;
    }

    //-------------------------------------------------------------//
    //------------------ Optimization -----------------------//
    //-------------------------------------------------------------//

    struct timespec start_whole, end_whole;
    double whole_time = 0;
    clock_gettime(CLOCK_MONOTONIC, &start_whole);

    // Create the optimizer
    nlopt_opt opt;
    opt = nlopt_create(NLOPT_LN_BOBYQA, opts.num_params);

    // Set the objective function
    nlopt_set_max_objective(opt, llh_Xvecchia_batch, &data);

    // Set bounds, if any
    for (int i = 0; i < opts.num_params; i++)
    {
        lb[i] = opts.lower_bound;
        ub[i] = opts.upper_bound;
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    // Set stopping criteria
    nlopt_set_xtol_rel(opt, opts.tol);
    nlopt_set_maxeval(opt, opts.maxiter);

    // Variable to store the minimum objective value
    double maxf;

    // Run the optimizer
    if (nlopt_optimize(opt, localtheta_initial, &maxf) < 0)
    {
        printf("nlopt failed!\n");
    }
    else
    {
        printf("Found minimum at f(%g, %g, %g) = %0.10g\n", localtheta_initial[0], localtheta_initial[1], localtheta_initial[2], maxf);
    }
    assert(nlopt_get_numevals(opt) == data.iterations);
    clock_gettime(CLOCK_MONOTONIC, &end_whole);
    whole_time = end_whole.tv_sec - start_whole.tv_sec + (end_whole.tv_nsec - start_whole.tv_nsec) / 1e9;
    saveLogFileSum<T>(data.iterations, localtheta_initial, maxf, whole_time, opts);
    // Free the optimizer
    nlopt_destroy(opt);
    fprintf(stderr, "Done! \n");

    if (opts.vecchia)
    {
        free(locations_con->x);
        free(locations_con->y);
        magma_free(d_Cov_conditioning);
        magma_free(d_Cov_cross);
        magma_free(d_obs_conditioning);
        magma_free(d_obs_conditioning_copy);
        magma_free(d_Cov_offset);
        magma_free(d_mu_offset);
        magma_free_cpu(h_Cov_conditioning);
        magma_free_cpu(h_Cov_cross);
        magma_free_cpu(h_obs_conditioning);
    }
    magma_free(d_Cov_conditioning_array);
    magma_free(d_Cov_cross_array);
    magma_free(d_Cov_offset_array);
    magma_free(d_mu_offset_array);
    magma_free(d_obs_conditioning_array);
    magma_free(d_obs_conditioning_array_copy);
    magma_free(d_Cov_array);
    magma_free(d_obs_array);
    magma_free(d_obs_array_copy);

    // independent free
    free(clusterid);
    free(locations->x);
    free(locations->y);
    free(locations_new->x);
    free(locations_new->y);
    free(clusterNum);
    // free(centroid);
    free(permIndex);
    free(batchNumAccum);
    free(batchNumSquareAccum);

    magma_free_cpu(h_Cov);
    magma_free_cpu(h_obs);
    magma_free_cpu(dot_result_h);
    magma_free_cpu(logdet_result_h);

    magma_free(d_Cov);
    magma_free(d_obs);
    magma_free(d_obs_copy);
    magma_free(d_batchNum);
    magma_free(dinfo_magma);
    magma_free(d_const1);
    magma_free(d_ldda);
    magma_free(d_lda);
    magma_free(d_ldacon);
    magma_free(d_lddacon);

    TESTING_CHECK(magma_finalize());
    return 0;
}

//==============================================================================================
int main(int argc, char **argv)
{

    Vecchia_opts opts;
    parse_opts(argc, argv, &opts);

    double alpha = 1.;
    gsl_set_error_handler_off();
    return test_Xvecchia_batch<double>(opts, alpha);
}
