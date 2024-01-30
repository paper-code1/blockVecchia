#ifndef LLG_H
#define LLG_H

#include <cmath>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#ifdef __cplusplus
extern "C"
{
#endif

  // to combine the first clusters together and
  // then the out is feed for the new locations and
  // obeservations for batch operations
  void cluster_combine(int *firstClusterCount, int *firstClusterSize, int num_loc, int *clusterid, int *permIndex, int *clusterNum)
  {
    // combine the first clusters together
    // when the first proper cluster is larger than 1
    // e.g.: cluster 1 and cluster 2 -> cluster 1
    if (firstClusterCount[0] > 1)
    {
      // permIndex[i]: current cluster number
      for (int i = 1; i < firstClusterCount[0]; i++)
      {
        // the number of cluster 1, 2 (0 is default)
        int _nc = 0;
        for (int k = 0; k < num_loc; k++)
        {
          if (clusterid[k] == permIndex[i])
          {
            clusterid[k] = permIndex[0];
            _nc++;
          }
        }
        assert(_nc == clusterNum[permIndex[i]]);
        permIndex[i] = permIndex[0];
        clusterNum[permIndex[i]] = firstClusterSize[0];
      }
    }
  }

  void cluster_to_batch(int num_loc, int batchCount, int *batchNum, int *batchNumAccum, int *batchIndex, location *locations, double *h_obs, location *locations_new, double *h_obs_new, int *clusterid, location *locsCentroid)
  {
    // reconstruct the new locations and observations
    for (int i = 0; i < batchCount; ++i)
    {
      // batchIndex[i] is the current cluster, it has batchNum[i] locations 
      int _id = 0;
      for (int j = 0; j < batchNum[i]; ++j)
      {
        while (true)
        {
          if ((clusterid[_id] == batchIndex[i]) || _id >= num_loc)
            break;
          else
            _id++;
        }
        if (_id < num_loc)
        {
          locations_new->x[batchNumAccum[i] + j] = locations->x[_id];
          locations_new->y[batchNumAccum[i] + j] = locations->y[_id];
          h_obs_new[batchNumAccum[i] + j] = h_obs[_id];
          _id++;
        }
      }
    }
  }

  // check the first batch size
  void firstbatch(int nclusters, int *clusterNum, int *permIndex, int *firstClusterSize, int *firstClusterCount, int cs)
  {
    int size_ft = 0;
    for (int i = 0; i < nclusters; ++i)
    {
      size_ft += clusterNum[permIndex[i]];
      if (size_ft >= cs)
      {
        firstClusterSize[0] = size_ft;
        firstClusterCount[0] = i + 1;
        break;
      }
    }
  }

  /*
  clusters reordering index
  */
  bool hasDuplicates(int *array, int length)
  {
    for (int i = 0; i < length - 1; i++)
    {
      for (int j = i + 1; j < length; j++)
      {
        if (array[i] == array[j])
        {
          return true;
        }
      }
    }
    return false;
  }
  void reorderIndex(location *locsCentroid, double **centroid, int *clusterReordering, int nclusters)
  {
    for (int i = 0; i < nclusters; ++i)
    {
      double _dist_min = 999999999999;
      double _dist_temp = 0;
      for (int j = 0; j < nclusters; ++j)
      {
        _dist_temp = sqrt(pow(locsCentroid->x[i] - centroid[j][0], 2) +
                          pow(locsCentroid->y[i] - centroid[j][1], 2));
        if (_dist_temp < _dist_min)
        {
          clusterReordering[i] = j;
          _dist_min = _dist_temp;
        }
      }
    }
    // check if there is same orders
    if (hasDuplicates(clusterReordering, nclusters))
    {
      fprintf(stderr, "Your reordering has unknown issues, please check it");
      exit(-1);
    }
  }

  /*
  clustering for the locations
  */
  // Function to print the results
  void printClusterResults(int nclusters, int num_loc, int *clusterid, double **cdata, int *cnum, int *max_num_cluster)
  {
    printf("Cluster results:\n");
    for (int i = 0; i < nclusters; ++i)
    {
      printf("Cluster %d: (%lf, %lf) %d locations", i, cdata[i][0], cdata[i][1], cnum[i]);
      printf("\n");
    }
    printf("Max neighbors %d \n", max_num_cluster[0]);
  }

  // Function to print the results and count the number of points in each cluster
  void ClusterCounts(int nclusters, int num_loc, int *clusterid, int *cnum, int *max_num_cluster, int *min_num_cluster)
  {
    for (int i = 0; i < num_loc; ++i)
    {
      cnum[clusterid[i]]++; // Increment count for the corresponding cluster
    }
    min_num_cluster[0] = 9999;
    max_num_cluster[0] = -1;
    for (int i = 0; i < nclusters; ++i)
    {
      if (max_num_cluster[0] < cnum[i])
        max_num_cluster[0] = cnum[i];
      if (min_num_cluster[0] > cnum[i])
        min_num_cluster[0] = cnum[i];
    }
  }

  void clustering_2D(size_t nclusters, int num_loc,
                     location *locations,
                     int *clusterid, double **cdata,
                     int *cnum, int *max_num_cluster, int seed)
  {
    // 2D locations
    int dim = 2;
    srand(seed);
    // Allocate and initialize data, mask, and weight arrays
    double **data = (double **)malloc(num_loc * sizeof(double *));
    int **mask = (int **)malloc(num_loc * sizeof(int *));
    int **cmask = (int **)malloc(nclusters * sizeof(int *));
    double *weight = (double *)malloc(dim * sizeof(double));

    for (int i = 0; i < num_loc; ++i)
    {
      // transform the locations into pointer to pointer array
      data[i] = (double *)malloc(dim * sizeof(double));
      mask[i] = (int *)malloc(dim * sizeof(int));
      data[i][0] = locations->x[i];
      data[i][1] = locations->y[i];
      mask[i][0] = 1;
      mask[i][1] = 1; // No missing data
      // init the cluster points
      if (i < nclusters)
      {
        cmask[i] = (int *)malloc(dim * sizeof(int));
        for (int j = 0; j < dim; ++j)
        {
          cdata[i][j] = -1;
          cmask[i][j] = 1; // No missing data
        }
      }
    }

    // Initialize weights to 1
    for (int j = 0; j < dim; ++j)
      weight[j] = 1.0;

    // Other parameters
    int transpose = 0;
    int npass = 0;
    char method = 'a';
    char dist = 'e'; // Assuming 'e' is a valid distance type

    // init the cluster id, no random effect here
    for (int i = 0; i < num_loc; ++i)
      // clusterid[i] = i % nclusters;
      clusterid[i] = rand() % nclusters;
    double error;
    int ifound;
    int *min_num_cluster = (int *)malloc(sizeof(int));

    // Call the kcluster function
    kcluster(nclusters, num_loc, dim, data, mask, weight, transpose, npass, method, dist, clusterid, &error, &ifound);

    getclustercentroids(nclusters, num_loc, dim, data, mask, clusterid, cdata, cmask, transpose, method);

    ClusterCounts(nclusters, num_loc, clusterid, cnum, max_num_cluster, min_num_cluster);

    // Check results (debug mode)
    // printClusterResults(nclusters, num_loc, clusterid, cdata, cnum, max_num_cluster);
    fprintf(stderr, "Average distance to centroid: %f\n", error / num_loc);
    fprintf(stderr, "Smallest cluster: %d\n", min_num_cluster[0]);
    fprintf(stderr, "Largest cluster: %d\n", max_num_cluster[0]);

    // printf("Ifound: %d\n", ifound);

    // Clean up
    for (int i = 0; i < num_loc; ++i)
    {
      free(data[i]);
      free(mask[i]);
    }
    free(data);
    free(mask);
    // free(weight);
  }
#ifdef __cplusplus
}
#endif

/*
determinant for log(det(A)) = log(det(L)det(L^T))
strided version
*/

template <class T>
void core_Xlogdet(T *L, int An, int ldda, T *logdet_result_h)
{
  T *L_h = (T *)malloc(sizeof(T) * An * ldda);
  cudaMemcpy(L_h, L, sizeof(T) * An * ldda, cudaMemcpyDeviceToHost);
  *logdet_result_h = 0;
  for (int i = 0; i < An; i++)
  {
    // printf("%d L diagnal value %lf\n", i, L_h[i + i * ldda]);
    // printf("%d the value is %lf \n", i, * logdet_result_h);
    // printf("%d the value is %p \n", i, logdet_result_h);
    if (L_h[i + i * ldda] > 0)
      *logdet_result_h += log(L_h[i + i * ldda] * L_h[i + i * ldda]);
  }
  // printf("the value is %lf \n", * logdet_result_h);
  // printf("-----------------------------------");
  free(L_h);
}

#endif