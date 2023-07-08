#pragma once

#include"temporalconfig.cuh"
#include"iptconfig.cuh"

template<int halo, int shape>
struct quesize
{
  static int const que = 1+halo*2;
  static int const smque = 1+halo*2;
  static int const regque = 1+halo;// in afraid of register pressure, shrink the regque size

};


template<int halo>
struct quesize<halo, star_shape>
{
  static int const que = 1+halo;

  static int const smque = 1+halo;
  static int const regque = 1+halo;
};


template<class REAL, int halo, int blockdim=256, 
  int tilex=ipts<halo,curshape,REAL>::tile_x, 
  int ipt=ipts<halo,curshape,REAL>::val, 
  int tiley=ipt*blockdim/tilex,
  int mqsize=timesteps<halo,curshape,ipt,REAL>::val, 
  int smquesize=quesize<halo,curshape>::smque,//halo+1, 
  int regquesize=quesize<halo,curshape>::regque,
  int quesize=quesize<halo,curshape>::que,
  int sizeofsm=smquesize*mqsize+1, int sizeofreg=regquesize*mqsize>
__global__ void kernel3d_temporal(REAL* __restrict__ input, REAL* output,
                                  int height, int width_y, int width_x, 
                                  REAL * l2_cache_i, REAL * l2_cache_o); 



