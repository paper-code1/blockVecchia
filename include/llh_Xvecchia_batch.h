/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file incase/llh_Xvecchia_batch

 *
 *
 *
 * @version 1.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/

#ifndef LLH_XVECCHIA_BATCH_H
#define LLH_XVECCHIA_BATCH_H

#include <omp.h>

template <class T>

T llh_Xvecchia_batch(unsigned n, const T *localtheta, T *grad, void *f_data)
{
    llh_data *data = static_cast<llh_data *>(f_data);

    // operations config
    int *dinfo_magma = data->dinfo_magma;
    int *hinfo_magma = data->hinfo_magma;
    magma_queue_t queue = data->queue;
    // size and etc
    int omp_threads = data->omp_threads;
    size_t batchCount = data->batchCount;
    int cs = data->cs;
    int *batchNum = data->batchNum;
    int *batchNumAccum = data->batchNumAccum;
    int *batchNumSquareAccum = data->batchNumSquareAccum;
    int *h_lda = data->h_lda;
    int *h_ldda = data->h_ldda;
    int *h_ldacon = data->h_ldacon;
    int *h_lddacon = data->h_lddacon;
    int *d_lda = data->d_lda;
    int *d_ldda = data->d_ldda;
    int *d_ldacon = data->d_ldacon;
    int *d_lddacon = data->d_lddacon;
    int *d_const1 = data->d_const1;
    int *d_batchNum = data->d_batchNum;
    int total_size_dev_obs = data->total_size_dev_obs;
    int z_flag = data->time_flag ? 1 : 0; // 1: time/ 3D; 0 : 2D

    // matrix and vectors
    double *h_Cov = data->h_Cov;
    double *d_Cov = data->d_Cov;
    double *h_obs_new = data->h_obs_new;
    double *d_obs = data->d_obs;
    double *h_Cov_conditioning = data->h_Cov_conditioning;
    double *d_Cov_conditioning = data->d_Cov_conditioning;
    double *h_Cov_cross = data->h_Cov_cross;
    double *d_Cov_cross = data->d_Cov_cross;
    double *h_obs_conditioning = data->h_obs_conditioning;
    double *d_obs_conditioning = data->d_obs_conditioning;
    double **d_Cov_array = data->d_Cov_array;
    double **d_obs_array = data->d_obs_array;
    double **d_Cov_conditioning_array = data->d_Cov_conditioning_array;
    double **h_Cov_conditioning_array = data->h_Cov_conditioning_array;
    double **d_Cov_cross_array = data->d_Cov_cross_array;
    double **d_obs_conditioning_array = data->d_obs_conditioning_array;
    // incase for the modification in place
    double *d_obs_copy = data->d_obs_copy;
    double *d_obs_conditioning_copy = data->d_obs_conditioning_copy;
    double **d_obs_array_copy = data->d_obs_array_copy;
    double **h_obs_array_copy = data->h_obs_array_copy;
    double **d_obs_conditioning_array_copy = data->d_obs_conditioning_array_copy;
    double *d_Cov_offset = data->d_Cov_offset;
    double **d_Cov_offset_array = data->d_Cov_offset_array;
    double *d_mu_offset = data->d_mu_offset;
    double **d_mu_offset_array = data->d_mu_offset_array;
    double **h_mu_offset_array = data->h_mu_offset_array;
    double **h_Cov_array = data->h_Cov_array;
    double **h_Cov_offset_array = data->h_Cov_offset_array;
    double **h_Cov_cross_array = data->h_Cov_cross_array;
    double **h_obs_conditioning_array_copy = data->h_obs_conditioning_array_copy;

    omp_set_num_threads(omp_threads);

    // h_obs_update: used for update the "mu_new = mu_old - correctionItem "
    /*
        Example,
        p(y1 | y2) with mean zero gaussian distribution
        \mu'_1 = \sigma_{12}  inv(\sigma_{22}) y_2
        \sigma'_{11} = \sigma_{11} - \sigma_{12}  inv(\sigma_{22}) \sigma_{12}^T

        d_Cov: \sigma'_{11}/ \sigma_{11}
        d_Cov_offset: \sigma_{12}  inv(\sigma_{22}) \sigma_{12}^T because gemm cannot overwrite the value in place
        d_obs: y1
        d_mu_offset: \mu'_1, used for pdf calculation

        d_C_conditioning: y_2
        d_Cov_conditioning: \sigma_{22}
        d_Cov_cross: \sigma_{12}

        PS: there are 5 important generation parts
        d_Cov: \sigma_{11}
        *d_Cov_conditioning: \sigma_{22}
        *d_Cov_cross: \sigma_{12}

        *d_C_conditioning: y_2
        d_obs: y1
    */

    //
    //-----------------------------------------------------------//
    //------------------Covariance matrix generation...-----------------------//
    //-----------------------------------------------------------//
    // loc_batch: for example, p(y1|y2), the locations_new of y1 is the loc_batch
#pragma omp parallel for
    for (size_t i = 0; i < batchCount; i++)
    {
        // h_Cov: \sigma_{lk}
        location *loc_batch = (location *)malloc(sizeof(location));
        loc_batch->x = data->locations_new->x + batchNumAccum[i];
        loc_batch->y = data->locations_new->y + batchNumAccum[i];
        loc_batch->z = data->time_flag ? (data->locations_new->z + batchNumAccum[i]) : NULL;
        // printLocations(data->batchNum[i], loc_batch);
        if (data->time_flag && data->kernel == 4)
        {
            core_dcmg_spacetime_matern(h_Cov + batchNumSquareAccum[i],
                                       batchNum[i], // each single  batch size
                                       batchNum[i],
                                       loc_batch, // starting of the locations_new
                                       loc_batch,
                                       localtheta, data->distance_metric);
        }
        else
        {
            core_dcmg(h_Cov + batchNumSquareAccum[i],
                      batchNum[i], // each single  batch size
                      batchNum[i],
                      loc_batch, // starting of the locations_new
                      loc_batch,
                      localtheta, data->distance_metric, 
                      z_flag, data->dist_scale);
        }
        // printVectorCPU(data->Cm, data->h_obs_new, data->ldc, i);
        // if (i == 0) {
        //     printMatrixCPU(batchNum[i], batchNum[i], h_Cov + batchNumSquareAccum[i], h_lda[i], i);
        // }
        free(loc_batch);
    }
    // h_Cov_cross: \sigma_{12} and h_Cov_conditioning: \sigma_{22}
    if (data->vecchia && cs > 0)
    {
#pragma omp parallel for
        for (size_t i = 0; i < batchCount; i++)
        {
            // printVectorCPU(data->bs, data->h_obs_new, data->ldc, 0);
            // for example, p(y1|y2), the locations_new of y2 is the loc_batch_con
            location *loc_batch_con = (location *)malloc(sizeof(location));
            location *loc_batch = (location *)malloc(sizeof(location));
            loc_batch_con->x = data->locations_con->x + i * cs;
            loc_batch_con->y = data->locations_con->y + i * cs;
            loc_batch_con->z = data->time_flag ? (data->locations_con->z + i * cs) : NULL;
            loc_batch->x = data->locations_new->x + batchNumAccum[i];
            loc_batch->y = data->locations_new->y + batchNumAccum[i];
            loc_batch->z = data->time_flag ? (data->locations_new->z + batchNumAccum[i]) : NULL;
            // please print loc_batch_con and loc_batch in format of "(x,y,z)"
            // if (i == 1) {
            //     for (int j = 0; j < batchNum[i]; j++) {
            //         fprintf(stderr, "\"(%lf, %lf, %lf)\", \n", loc_batch->x[j], loc_batch->y[j], data->h_obs_new[batchNum[0] + j]);
            //     }
            //     printf("-----------------\n");
            //     for (int j = 0; j < cs; j++) {
            //         fprintf(stderr, "\"(%lf, %lf, %lf)\", \n", loc_batch_con->x[j], loc_batch_con->y[j], data->h_obs_conditioning[cs + j]);
            //     }
            // }

            // printLocations(cs, loc_batch_con);
            if (data->time_flag && data->kernel == 4)
            {
                //*h_Cov_conditioning: \sigma_{22}
                core_dcmg_spacetime_matern(h_Cov_conditioning + i * cs * cs,
                                           cs, cs,
                                           loc_batch_con,
                                           loc_batch_con, localtheta, data->distance_metric);
                // *h_Cov_cross: \sigma_{21}
                // printLocations(data->batchNum[i], loc_batch);
                core_dcmg_spacetime_matern(h_Cov_cross + cs * batchNumAccum[i],
                                           cs, batchNum[i],
                                           loc_batch_con,
                                           loc_batch, localtheta, data->distance_metric);
            }
            else
            {
                //*h_Cov_conditioning: \sigma_{22}
                core_dcmg(h_Cov_conditioning + i * cs * cs,
                          cs, cs,
                          loc_batch_con,
                          loc_batch_con, localtheta, data->distance_metric,
                          z_flag, data->dist_scale);
                // *h_Cov_cross: \sigma_{21}
                // printLocations(data->batchNum[i], loc_batch);
                core_dcmg(h_Cov_cross + cs * batchNumAccum[i],
                          cs, batchNum[i],
                          loc_batch_con,
                          loc_batch, localtheta, data->distance_metric,
                          z_flag, data->dist_scale);
            }
            // printf("The conditioning covariance matrix.\n");
            // printMatrixCPU(cs, batchNum[i], h_Cov_cross + cs * batchNum[i], h_ldacon[0], i);
            free(loc_batch);
            free(loc_batch_con);
        }
    }

    //-----------------------------------------------------------//
    //------------------Memory set/get/...-----------------------//
    //-----------------------------------------------------------//
    double *h_Cov_tmp, *d_Cov_tmp;
    // copy the (conditioning) observations, which is overwritten
    // for each iterations
    magma_dcopy(h_lddacon[0] * batchCount, d_obs_conditioning, 1, d_obs_conditioning_copy, 1, queue);
    magma_dcopy(total_size_dev_obs, d_obs, 1, d_obs_copy, 1, queue);
    // printMatrixGPU(h_lda[1], 1, h_obs_array_copy[1], h_ldda[1], 1);
    // printMatrixGPU(h_ldacon[1], 1, h_obs_conditioning_array_copy[1], h_lddacon[1], 1);

    h_Cov_tmp = h_Cov;
    d_Cov_tmp = d_Cov;
    for (int i = 0; i < batchCount; i++)
    {
        magma_dsetmatrix(batchNum[i], batchNum[i],
                         h_Cov_tmp, h_lda[i],
                         d_Cov_tmp, h_ldda[i],
                         queue);
        h_Cov_tmp += batchNum[i] * h_lda[i];
        d_Cov_tmp += batchNum[i] * h_ldda[i];
    }
    magma_setvector(batchCount, sizeof(int), hinfo_magma, 1, dinfo_magma, 1, queue);
    if (data->vecchia && cs > 0)
    {
        // printf("[info] The vecchia offset is starting now!\n");
        // printMatrixCPU(cs, cs, data->h_Cov_first, cs, 0);
        double *h_Cov_conditioning_tmp, *d_Cov_conditioning_tmp;
        double *h_Cov_cross_tmp, *d_Cov_cross_tmp;
        double *h_obs_tmp, *d_obs_tmp;

        h_Cov_conditioning_tmp = h_Cov_conditioning;
        d_Cov_conditioning_tmp = d_Cov_conditioning;
        h_Cov_cross_tmp = h_Cov_cross;
        d_Cov_cross_tmp = d_Cov_cross;
        for (int i = 0; i < batchCount; i++)
        {
            magma_dsetmatrix(cs, cs,
                             h_Cov_conditioning_tmp, h_ldacon[i],
                             d_Cov_conditioning_tmp, h_lddacon[i],
                             queue);
            // printMatrixGPU(cs, batchNum[i], d_Cov_cross_tmp, h_ldda[i], i);
            // printMatrixCPU(cs, batchNum[i], h_Cov_cross_tmp, h_lda[i], i);
            magma_dsetmatrix(cs, batchNum[i],
                             h_Cov_cross_tmp, h_ldacon[i],
                             d_Cov_cross_tmp, h_lddacon[i],
                             queue);
            // printMatrixGPU(cs, cs, h_Cov_conditioning_array[i], h_lddacon[0], i);
            // printMatrixGPU(cs, batchNum[i], d_Cov_cross_tmp, h_ldda[i], i);
            h_Cov_conditioning_tmp += h_ldacon[i] * cs;
            d_Cov_conditioning_tmp += h_lddacon[i] * cs;
            h_Cov_cross_tmp += h_ldacon[i] * batchNum[i];
            d_Cov_cross_tmp += h_lddacon[i] * batchNum[i];
        }
        // sigma_conditioning
        // sigma_cross
        // z_conditioning
    }

    //-----------------------------------------------------//
    //------------------Conditioning-----------------------//
    //-----------------------------------------------------//
    if (data->vecchia && cs > 0)
    {

        // vbatch return value
        int info;
        /*
        cholesky decomposition
        */
        // for (int i = 0; i < batchCount; i++)
        // {
        //     // printMatrixGPU(h_ldacon[i], h_lda[i], h_Cov_cross_array[i], h_ldda[i], i);
        //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_conditioning_array[i], h_ldda[i], i);
        // }
        // printf("[info] Starting Cholesky decomposition. \n");

        // gpu_time = magma_sync_wtime(queue);

        info = magma_dpotrf_vbatched(
            MagmaLower, d_ldacon,
            d_Cov_conditioning_array, d_lddacon,
            dinfo_magma, batchCount,
            queue);

        // gpu_time = magma_sync_wtime(queue) - gpu_time;
        // gpu_perf = gflops / gpu_time;

        // for (int i = 0; i < batchCount; i++)
        // {
        //     printMatrixGPU(h_ldacon[i], h_lda[i], h_Cov_cross_array[i], h_ldda[i], i);
        //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_conditioning_array[i], h_ldda[i], i);
        // }
        // printf("[info] Finished Cholesky decomposition. \n");
        /*
        triangular solution: L \Sigma_offset <- \Sigma_old && L z_offset <- z_old
        */
        // printf("[info] Starting triangular solver. \n");
        // for (int i = 1; i < batchCount; i++)
        // {
        //     printMatrixGPU(h_ldacon[i], h_lda[i], h_Cov_cross_array[i], h_ldda[i], i);
        //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_conditioning_array[i], h_ldda[i], i);
        // }
        // d_obs_conditioning keeps the same for each iteration
        magmablas_dtrsm_vbatched(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
            d_ldacon, d_lda, 1.,
            d_Cov_conditioning_array, d_lddacon,
            d_Cov_cross_array, d_lddacon,
            batchCount, queue);
        // for (int i = 0; i < 1; i++) // batchCount
        // {
        //     // printMatrixGPU(h_ldacon[i], h_ldacon[i], h_Cov_conditioning_array[i], h_lddacon[i], i);
        //     printMatrixGPU(h_ldacon[i], 1, h_obs_conditioning_array_copy[i], h_lddacon[i], i);
        // }
        // printMatrixGPU(h_ldacon[1], 1, h_obs_conditioning_array_copy[1], h_lddacon[1], 1);
        magmablas_dtrsm_vbatched(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
            d_ldacon, d_const1, 1.,
            d_Cov_conditioning_array, d_lddacon,
            d_obs_conditioning_array_copy, d_lddacon,
            batchCount, queue);
        // for (int i = 0; i < 1; i++) // batchCount
        //     printMatrixGPU(h_ldacon[i], 1, h_obs_conditioning_array_copy[i], h_lddacon[i], i);
        // printMatrixGPU(h_ldacon[1], 1, h_obs_conditioning_array_copy[1], h_lddacon[1], 1);

        // printf("[info] Finished triangular solver. \n");
        /*
        GEMM and GEMV: \Sigma_offset^T %*% \Sigma_offset and \Sigma_offset^T %*% z_offset
        */
        // // \Sigma_offset^T %*% \Sigma_offset
        // for (int i = 1; i < batchCount; i++)
        // {
        //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_cross_array[i],  h_ldda[i], i);
        //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_offset_array[i],  h_ldda[i], i);
        // }
        magmablas_dgemm_vbatched(MagmaTrans, MagmaNoTrans,
                                 d_lda, d_lda, d_ldacon,
                                 1,
                                 d_Cov_cross_array, d_lddacon,
                                 d_Cov_cross_array, d_lddacon,
                                 0,
                                 d_Cov_offset_array, d_ldda,
                                 batchCount,
                                 queue);

        // for (int i = 1; i < 10; i++)
        // {
        //     // printMatrixGPU(h_lda[i], h_lda[i], h_Cov_cross_array[i],  h_ldda[i], i);
        //     // printMatrixGPU(h_lda[i], h_lda[i], h_Cov_offset_array[i],  h_ldda[i], i);
        //     printVecGPU(h_ldacon[i], 1, h_obs_conditioning_array_copy[i], h_lddacon[i], i);
        // }
        // \Sigma_offset^T %*% z_offset
        // GEMV -> GEMM (GEMV is supposed to be better, but there is unknown issues with the API)
        magmablas_dgemm_vbatched(MagmaTrans, MagmaNoTrans,
                                 d_lda, d_const1, d_ldacon,
                                 1,
                                 d_Cov_cross_array, d_lddacon,
                                 d_obs_conditioning_array_copy, d_lddacon,
                                 0,
                                 d_mu_offset_array, d_ldda,
                                 batchCount,
                                 queue);
        // for (int i = 1; i < 10; i++)
        // {
        //     printVecGPU(h_lda[i], 1, h_mu_offset_array[i], h_ldda[i], i);
        // }
        /*TODO non-data->strided*/
        // printf("[info] Finished GEMM and GEMV. \n");
        /*
        GEAD: \Sigma_new <- \Sigma - \Sigma_offset && \mu_new <- \mu - \mu_offset (not necessary)
        */
        /*
            in geam, the original value can be overwritten
             Cov22'<- Cov22 - Cov12^T inv(Cov11) Cov12
             Cov22: d_obs
             Cov22': d_obs
             Cov12^T inv(Cov11) Cov12: d_mu_offset
        */
        /*
            in geam, the original value can be overwritten
            y'<- y - mu
            y: d_obs
            y': d_obs
            mu: d_mu_offset
        */
        // for (int i = 0; i < batchCount; i++)
        // {
        //     printf("%dth", i);
        //     printMatrixGPU(h_lda[i], h_lda[i], d_Cov_array[i], h_ldda[i]);
        // }
        for (size_t i = 1; i < batchCount; ++i)
        {

            // printMatrixGPU(h_lda[i], h_lda[i], h_Cov_array[i], h_ldda[i], i);
            // printMatrixGPU(h_lda[i], h_lda[i], h_Cov_offset_array[i], h_ldda[i], i);
            // if(h_lda[i] > h_ldda[i] || 1 > h_ldda[i]){
            //         printf("Error: leading dimension lda (%d) is less than the number of rows (%d).\n", h_ldda[i], h_lda[i]);
            // }
            magmablas_dgeadd(h_lda[i], h_lda[i],
                             -1.,
                             h_Cov_offset_array[i], h_ldda[i], // d_ldda[i]
                             h_Cov_array[i], h_ldda[i],
                             queue);
            // printf("The results before TRSM \n");
            // printMatrixGPU(h_lda[i], h_lda[i], h_Cov_array[i], h_ldda[i], i);
            // if (i < 10)
            //     printMatrixGPU(h_lda[i], 1, h_obs_array_copy[i], h_ldda[i], i);
            magmablas_dgeadd(h_lda[i], 1,
                             -1,
                             h_mu_offset_array[i], h_ldda[i],
                             h_obs_array_copy[i], h_ldda[i],
                             queue);
            // if (i < 10)
            //     printMatrixGPU(h_lda[i], 1, h_obs_array_copy[i], h_ldda[i], i);
        }
    }
    // */conditioning part, \sigma_{12} inv (\sigma_{22}) \sigma_{21}
    // printf("[info] The vecchia offset is finished!  \n");
    // printf("[Info] The time for vecchia offset is %lf seconds \n", vecchia_time);
    // // printf("[info] Independent computing is starting now! \n");

    //-----------------------------------------------------//
    //------------------independent blocks-----------------------//
    //-----------------------------------------------------//
    // intermidiate results
    double *logdet_result_h = data->logdet_result_h;
    double *norm2_result_h = data->norm2_result_h;
    int info = 0;        // debug for potrf
    double llk = 0;      // log-likelihood
    double _llk_tmp = 0; // debug for log-likelihood

    // cholesky
    info = magma_dpotrf_vbatched(
        MagmaLower, d_batchNum,
        d_Cov_array, d_ldda,
        dinfo_magma, batchCount,
        queue);
    // // triangular solution: L Z_new <- Z_old
    // for (int i = 0; i < 10; i++)
    // {
    //     printMatrixGPU(h_lda[i], h_lda[i], h_Cov_array[i], h_ldda[i], i);
    //     printMatrixGPU(h_lda[i], 1, h_obs_array_copy[i], h_ldda[i], i);
    // }

    magmablas_dtrsm_vbatched(
        MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
        d_batchNum, d_const1, 1.,
        d_Cov_array, d_ldda,
        d_obs_array_copy, d_ldda,
        batchCount, queue);
    // printMatrixGPU(h_lda[1], 1, h_obs_array_copy[1], h_ldda[1], 1);
    for (int i = 0; i < batchCount; ++i)
    {
        // determinant
        core_Xlogdet<T>(h_Cov_array[i], //
                        batchNum[i], h_ldda[i],
                        &(logdet_result_h[i]));
        // Dot scalar Z_new^T Z_new
        norm2_result_h[i] = magma_dnrm2(h_lda[i],
                                        h_obs_array_copy[i],
                                        1, queue);
    }

    for (int k = 0; k < batchCount; k++)
    {
        _llk_tmp = -(norm2_result_h[k] * norm2_result_h[k] + logdet_result_h[k] + batchNum[k] * log(2 * PI)) * 0.5;
        llk += _llk_tmp;
        // fprintf(stderr, "%dth location is %lf %lf \n", k, data->locations_new->x[cs + k], data->locations_new->y[cs + k]);
        // if (k == 1) {   
        //     fprintf(stderr, "%dth log determinant is % lf\n", k, logdet_result_h[k]);
        //     fprintf(stderr, "%dth dot product is % lf\n", k, norm2_result_h[k] * norm2_result_h[k]);
        //     fprintf(stderr, "%dth pi is % lf\n", k, batchNum[k] * log(2 * PI));
        //     fprintf(stderr, "%dth log likelihood is % lf\n", k, _llk_tmp);
        //     fprintf(stderr, "-------------------------------------\n");
        // }
        // if (k == 10)
        //     break;
    }
    // printf("[info] Independent computing is finished! \n");
    // printf("[Info] The time for independent computing is %lf seconds\n", indep_time);
    // printf("[Info] The time for LLH is %lf seconds\n", indep_time + vecchia_time);
    // // printf("(Estimated) Sigma: %lf beta:  %lf  nu: %lf\n", localtheta[0], localtheta[1], localtheta[2]);
    // printf("Log likelihood is %lf \n", llk);
    if (data->perf != 1)
    {
        if (data->kernel == 1 || data->kernel == 2)
        {
            printf("%dth Model Parameters (Variance, range, smoothness): (%1.8lf, %1.8lf, %1.8lf) -> Loglik: %.18lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2], llk);
        }
        else if (data->kernel == 3)
        {
            printf("%dth Model Parameters (Variance, range, smoothness, nugget): (%1.8lf, %1.8lf, %1.8lf, %1.8lf) -> Loglik: %.18lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2], localtheta[3], llk);
        }
        else if (data->kernel == 4)
        {
            printf("%dth Model Parameters (Variance, beta, nu, beta_time, nu_time, sep): (%lf, %lf, %lf, %lf, %lf, %lf) -> Loglik: %lf \n",
                   data->iterations, localtheta[0], localtheta[1], localtheta[2],
                   localtheta[3], localtheta[4], localtheta[5], llk);
        }
    }
    data->iterations += 1;
    // data->vecchia_time_total += (indep_time + vecchia_time);
    return llk;
}

#endif