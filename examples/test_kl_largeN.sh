#!/bin/bash

for iseed in 1 2 3 4
do
    for N in 20000 80000 180000 #20000 80000 120000 160000 200000 240000 
    do 
        for Nbc in 2500
        do
            mkdir -p "./log/20k-kl-bc$Nbc-largeN"
            ############## cluster vecchia ###############
            for Ncs in 1 10 30 60 90 120 150 180 210 240 270 300 330 360 390 420 450
            do
                ./bin/test_dvecchia_batch --ikernel 1.5:0.052537:1.5 --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --vecchia_bc $Nbc --knn --seed $iseed --permutation random 
                ./bin/test_dvecchia_batch --ikernel 1.5:0.052537:1.5 --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --vecchia_bc $Nbc --knn --seed $iseed --permutation mmd
            done
        mv ./log/locs_* "./log/20k-kl-bc$Nbc-largeN"
        done 
    done
done
# ## used for classic vecchia
# # Loop through the lines in the file
# while IFS=' ' read -r sigma beta nu seed; do
#     echo "===================================================="
#     echo "===================================================="
#     echo "sigma: $sigma, beta: $beta, nu: $nu, seed:$seed"
#     ############## classic vecchia ###############
#     for Ncs in 5 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 
#     do
#         for Nbc in 20000
#         do
#             ./bin/test_dvecchia_batch --ikernel $sigma:$beta:$nu --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40 --perf --vecchia_cs $Ncs --vecchia_bc $Nbc --knn --seed $seed
#         done
#     done 
# done < "$filename"
# mkdir ./log/20k-kl-classic
# mv ./log/locs_* ./log/20k-kl-classic