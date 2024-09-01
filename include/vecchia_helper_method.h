/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file include/vecchia_helper_method

 *
 *
 *
 * @version 1.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/
#ifndef LLG_H
#define LLG_H

#include <cmath>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <algorithm> // For std::max_element and std::min_element
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
  // to combine the first clusters together and
  // then the out is feed for the new locations and
  // obeservations for batch operations
  void cluster_combine(int *firstClusterCount, int *firstClusterSize, int num_loc, std::vector<Point> &points, int *permIndex, int *clusterNum)
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
          if (points[k].cluster == permIndex[i])
          {
            points[k].cluster = permIndex[0];
            _nc++;
          }
        }
        assert(_nc == clusterNum[permIndex[i]]);
        permIndex[i] = permIndex[0];
        clusterNum[permIndex[i]] = firstClusterSize[0];
      }
    }
  }

  void cluster_to_batch(int num_loc, int batchCount, int *batchNum, int *batchNumAccum, int *batchIndex, location *locations, double *h_obs, location *locations_new, double *h_obs_new, std::vector<Point> &points, location *locsCentroid, bool time_flag)
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
          if ((points[_id].cluster == batchIndex[i]) || _id >= num_loc)
            break;
          else
            _id++;
        }
        if (_id < num_loc)
        {
          locations_new->x[batchNumAccum[i] + j] = locations->x[_id];
          locations_new->y[batchNumAccum[i] + j] = locations->y[_id];
          if (time_flag)
          {
            locations_new->z[batchNumAccum[i] + j] = locations->z[_id];
          }
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
          fprintf(stderr, "Your index has duplicate: index[%d] and index[%d] share %d \n", i, j, array[j]);
          return false;
        }
      }
    }
    return false;
  }
  void reorderIndex(location *locsCentroid, std::vector<Point> &points, int *clusterReordering, int nclusters, bool time_flag)
  {
// in/out: clusterReordering, 0, 1, 2, 3, ...., n -> 10, 2, 3, ...,
#pragma omp parallel for
    for (int i = 0; i < nclusters; ++i)
    {
      double _dist_min = DBL_MAX;
      int min_index = 0; // Track the index of the minimum distance
      for (int j = 0; j < nclusters; ++j)
      {
        double dx = locsCentroid->x[i] - points[j].coordinates[0];
        double dy = locsCentroid->y[i] - points[j].coordinates[1];
        double _dist_temp = dx * dx + dy * dy; // Use squared distance
        if (time_flag){
          double dz = locsCentroid->z[i] - points[j].coordinates[2];
          _dist_temp += dz*dz;
        }
        if (_dist_temp < _dist_min)
        {
          _dist_min = _dist_temp;
          min_index = j;
        }
      }
      clusterReordering[i] = min_index;
    }

    // check if there is same orders
    if (hasDuplicates(clusterReordering, nclusters))
    {
      fprintf(stderr, "Your reordering has unknown issues, please check it\n");
      exit(-1);
    }
  }

  std::vector<Point> convertToPoints(location *&loc, int n, bool time_flag)
  {
    std::vector<Point> points;
    points.reserve(n); // Optimize memory allocation
    for (int i = 0; i < n; ++i)
    {
      Point p;                      // Use the default constructor
      p.coordinates[0] = loc->x[i]; // Set x coordinate
      p.coordinates[1] = loc->y[i]; // Set y coordinate
      if (time_flag)
      {
        p.coordinates[2] = loc->z[i]; // Set z coordinate
      }
      p.cluster = -1;      // Assuming you want to initialize the cluster to -1 or some default value
      points.push_back(p); // Add the point to the vector
    }

    return points;
  }

  // Transfer clusterCounts to an int array
  int *transferClusterCounts(const std::unordered_map<int, int> &clusterCounts)
  {
    int arraySize = 0;
    // Find the maximum cluster ID to determine the size of the array
    int maxClusterId = std::max_element(clusterCounts.begin(), clusterCounts.end(),
                                        [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                                        {
                                          return a.first < b.first;
                                        })
                           ->first;

    arraySize = maxClusterId + 1;           // Array size needs to be maxClusterId + 1 to include the max ID
    int *clusterNum = new int[arraySize](); // Allocate and zero-initialize the array

    // Fill the array with counts
    for (const auto &pair : clusterCounts)
    {
      clusterNum[pair.first] = pair.second;
    }

    return clusterNum;
  }

  // Function to count points in each cluster and find the maximum
  int *countPointsInClusters(const std::vector<Point> &points)
  {
    std::unordered_map<int, int> clusterCounts;

    // Count the number of points in each cluster
    for (const auto &point : points)
    {
      clusterCounts[point.cluster]++;
      // point.print();
    }

    // Find the cluster with the maximum number of points
    auto maxIt = std::max_element(clusterCounts.begin(), clusterCounts.end(),
                                  [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                                  {
                                    return a.second < b.second;
                                  });
    // Find the cluster with the minimum number of points
    auto minIt = std::min_element(clusterCounts.begin(), clusterCounts.end(),
                                  [](const std::pair<int, int> &a, const std::pair<int, int> &b)
                                  {
                                    return a.second < b.second;
                                  });

    // Output the cluster with the maximum number of points
    if (maxIt != clusterCounts.end())
    { // Ensure there are clusters
      std::cout << "Cluster with the most points: " << maxIt->first
                << " (" << maxIt->second << " points)\n";
    }
    else
    {
      std::cout << "No clusters found.\n";
    }

    // Output the cluster with the minimum number of points
    if (minIt != clusterCounts.end())
    { // Ensure there are clusters
      std::cout << "Cluster with the least points: " << minIt->first
                << " (" << minIt->second << " points)\n";
    }
    else
    {
      std::cout << "No clusters found.\n";
    }

    return transferClusterCounts(clusterCounts);
  }
  /*
  determinant for log(det(A)) = log(det(L)det(L^T))
  strided version
  */

#ifdef __cplusplus
}
#endif
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