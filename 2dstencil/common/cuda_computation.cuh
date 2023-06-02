#pragma once


#include"temporalconfig.cuh"
#include"iptconfig.cuh"

template<class REAL, int halo, int ipt=ipts<halo,curshape,REAL>::val, int tstep=timesteps<halo,curshape,ipt,REAL>::val>
__global__ void kernel_temporal(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *  __var_4__,REAL *  l2_cache, REAL *  l2_cachetmp, 
  REAL ** buffer, REAL ** caches, 
  int iteration);



template<class REAL, int halo, int ipt=ipts<halo,curshape,REAL>::val, int tstep=timesteps<halo,curshape,ipt,REAL>::val>
__global__ void kernel_temporal_traditional(REAL *__restrict__  input, int width_y, int width_x, 
  REAL *  __var_4__);

