/**
 * @copyright (c) 2024- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

/**
 * @file include/nearest_neighbor.h

 *
 *
 *
 * @version 1.0.0
 * @author Qilong Pan
 * @date 2024-03-14
 **/

#ifndef NEAREST_NEIGHBOR_H
#define NEAREST_NEIGHBOR_H

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#define pow_e (2.71828182845904)
#define PI (3.141592653589793)
#define earthRadiusKm (6371.0)

// used for nearest neighbor selection
double calEucDistance(double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * This function converts decimal degrees to radians
 * @param deg decimal degree
 */
// static double deg2rad(double deg)
// {
//     return (deg * PI / 180);
// }

/**
 * This function converts radians to decimal degrees
 * @param rad radians
 */
static double rad2deg(double rad)
{
    return (rad * 180 / PI);
}

/**
 * Returns the distance between two points on the Earth.
 * Direct translation from http://en.wikipedia.org/wiki/Haversine_formula
 * @param lat1d Latitude of the first point in degrees
 * @param lon1d Longitude of the first point in degrees
 * @param lat2d Latitude of the second point in degrees
 * @param lon2d Longitude of the second point in degrees
 * @return The distance between the two points in kilometers
 */
static double distanceEarth(double lat1d, double lon1d, double lat2d, double lon2d)
{
    double lat1r, lon1r, lat2r, lon2r, u, v;
    lat1r = deg2rad(lat1d);
    lon1r = deg2rad(lon1d);
    lat2r = deg2rad(lat2d);
    lon2r = deg2rad(lon2d);
    u = sin((lat2r - lat1r) / 2);
    v = sin((lon2r - lon1r) / 2);
    return 2.0 * earthRadiusKm * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

void findNearestPoints(double *h_obs_conditioning, location *locations_con, location *locsCentroid, int *firstClusterCount, double *h_obs_new, location *locations_new, int l0, int l1, int k, int i_block, int distance_metric, bool time_flag)
{

    double *centroid = (double *)calloc(3, sizeof(double));
    centroid[0] = locsCentroid->x[i_block + firstClusterCount[0] - 1];
    centroid[1] = locsCentroid->y[i_block + firstClusterCount[0] - 1];
    if (time_flag)
    {
        centroid[2] = locsCentroid->z[i_block + firstClusterCount[0] - 1];
    }

    double *distances = (double *)malloc(sizeof(double) * (l1 - l0));
    int *indices = (int *)malloc(sizeof(int) * (l1 - l0));

    for (int i = l0; i < l1; i++)
    {
        double distance;
        if (distance_metric == 1)
        {
            distance = distanceEarth(centroid[0], centroid[1], locations_new->x[i], locations_new->y[i]);
        }
        else
        {
            distance = calEucDistance(centroid[0], centroid[1], locations_new->x[i], locations_new->y[i]);
        }
        if (time_flag)
        {
            distance = sqrt(distance * distance + (centroid[2] - locations_new->z[i]) * (centroid[2] - locations_new->z[i]));
        }
        distances[i - l0] = distance;
        indices[i - l0] = i;
    }
    if (k > (l1 - l0))
    {
        printf("Not enough points available.\n");
        k = l1 - l0;
    }

    // Implementing sorting logic (simple selection sort for illustration)
    for (int i = 0; i < k; i++)
    {
        int min_idx = i;
        for (int j = i + 1; j < l1 - l0; j++)
        {
            if (distances[j] < distances[min_idx])
            {
                min_idx = j;
            }
        }
        // Swap the elements
        double temp_dist = distances[i];
        distances[i] = distances[min_idx];
        distances[min_idx] = temp_dist;

        int temp_idx = indices[i];
        indices[i] = indices[min_idx];
        indices[min_idx] = temp_idx;
    }

    for (int i = 0; i < k; i++)
    {
        locations_con->x[i_block * k + i] = locations_new->x[indices[i]];
        locations_con->y[i_block * k + i] = locations_new->y[indices[i]];
        if (time_flag)
        {
            locations_con->z[i_block * k + i] = locations_new->z[indices[i]];
        }
        h_obs_conditioning[i_block * k + i] = h_obs_new[indices[i]];
    }
    // Free allocated memory
    free(distances);
    free(indices);
}

#endif