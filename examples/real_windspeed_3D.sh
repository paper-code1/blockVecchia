#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

N=1000000
Ncs=(10 30 60 90 120 180 210) # 10 30 60 90 120 180 210

# block vecchia
for ((index=0; index<${#Ncs[@]}; index++));
do
    for Nbc in 100000 # 10000 50000 100000
    do
    if [ $index -eq 0 ]; then
        ./bin/test_dvecchia_batch --ikernel ?:?:? --vecchia_bc $Nbc --kernel univariate_matern_stationary_no_nugget --num_loc $N --omp_threads 40  --vecchia_cs ${Ncs[0]} --knn --permutation random --tol 5 --lower_bound 0.001 --maxiter 5000 --3D --xy_path ./windspeed_3d/meta_train_1000000 --obs_path ./windspeed_3d/observation_train_1000000
    else
        values=$(sed -n '2p' ./log/locs_${N}_cs_${Ncs[index-1]}_bc_${Nbc}_seed_0_kernel_1.000000:0.100000:0.500000_random | awk '{print $2, $3, $4}')
        values="${values//,}"
        read sigma_pre beta_pre nu_pre <<< "$values"
        echo $sigma_pre:$beta_pre:$nu_pre
        ./bin/test_dvecchia_batch --ikernel ?:?:? --vecchia_bc $Nbc --kernel univariate_matern_stationary_no_nugget  --num_loc $N --omp_threads 40  --vecchia_cs ${Ncs[index]} --knn --permutation random --kernel_init $sigma_pre:$beta_pre:$nu_pre --lower_bound 0.001 --tol 5 --3D --xy_path ./windspeed_3d/meta_train_1000000 --obs_path ./windspeed_3d/observation_train_1000000
    fi
    done
done
