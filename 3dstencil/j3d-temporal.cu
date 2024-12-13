#include "./config.cuh"

#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include <math.h>
#include <cooperative_groups.h>

#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>

#include "./common/multi_queue.cuh"

namespace cg = cooperative_groups;


// template<class REAL, int halo, int blockdim=256, 
//   int tilex=ipts<halo,curshape,REAL>::tile_x, 
//   int ipt=ipts<halo,curshape,REAL>::val, 
//   int tiley=ipt*blockdim/tilex,
//   int mqsize=timesteps<halo,curshape,ipt,REAL>::val, 
//   int smquesize=halo+1, int regquesize=halo+1 , int quesize=halo+1, 
//   int sizeofsm=smquesize*mqsize+1, int sizeofreg=regquesize*mqsize>
template <class REAL, int halo, int BLOCKDIM, int LOCAL_TILE_X, 
      int LOCAL_ITEM_PER_THREAD,  int LOCAL_TILE_Y, 
       int MQSIZE, 
       int SMQUESIZE, int REGQUESIZE, int QUESIZE,
      int SIZEOFSM, int SIZEOFREG>
__global__ void
kernel3d_temporal(REAL *__restrict__ input,
                  REAL *output,
                  int width_z, int width_y, int width_x,
                  REAL *l2_cache_i, REAL *l2_cache_o)
{
  
static constexpr int const REG_Y_SIZE_MOD = LOCAL_ITEM_PER_THREAD;    

  int gdim_y = (BLOCKDIM / LOCAL_TILE_X);
  const int tile_x_with_halo = LOCAL_TILE_X + 2 * halo;
  const int tile_y_with_halo = LOCAL_TILE_Y + 2 * halo;
  stencilParaT;
  extern __shared__ char sm[];
  REAL *sm_buffer = (REAL *)sm + 1;

  register REAL reg_mque[SIZEOFREG][REG_Y_SIZE_MOD];
    
  /**************/
  /**************/
  // Initialize the shared memory multiqueue indexing
  REAL *sm_mque[SIZEOFSM];
  sm_mque[0] = sm_buffer;
  _Pragma("unroll") for (int hl = 1; hl < SIZEOFSM; hl++)
  {
    sm_mque[hl] = sm_mque[hl - 1] + tile_x_with_halo * tile_y_with_halo;
  }
  /**************/
  /**************/

  const int tid_x = threadIdx.x % LOCAL_TILE_X;
  const int tid_y = threadIdx.x / LOCAL_TILE_X;
  const int index_y = LOCAL_ITEM_PER_THREAD * tid_y;
  const int ps_y = halo;
  const int ps_x = halo;

  const int p_x = blockIdx.x * LOCAL_TILE_X;
  const int c_x = blockIdx.x * LOCAL_TILE_X;
  const int p_y = blockIdx.y * LOCAL_TILE_Y;
  const int c_y = blockIdx.y * LOCAL_TILE_Y;

  int blocksize_z = ((width_z + gridDim.z - 1) / gridDim.z);
  const int p_z = blockIdx.z * (blocksize_z);
  const int p_z_end = p_z + (blocksize_z);

  int step_south = 0;
  int step_north = MQSIZE * gridDim.z * gridDim.y * halo * width_x;
  int step_west = MQSIZE * gridDim.z * gridDim.y * halo * width_x * 2;
  int step_east = MQSIZE * gridDim.z * gridDim.y * halo * width_x * 2 + MQSIZE * gridDim.x * gridDim.z * halo * width_y;

  cg::grid_group gg = cg::this_grid();
  {
    // // int p_y = p_y;
    // smEnqueueAsync<REAL, 1 + halo>(sm_mque + SIZEOFSM - 1 - SMQUESIZE,
    //                                ps_y, ps_x, tile_x_with_halo, tile_y_with_halo,
    //                                input,
    //                                p_z - (QUESIZE) * (MQSIZE - 1), p_y, p_x,
    //                                width_z, width_y, width_x);

    // regEnqueues<REAL, halo, LOCAL_ITEM_PER_THREAD>(reg_mque,
    //                                                input,
    //                                                p_z -  (QUESIZE) * (MQSIZE - 1) - halo, index_y + p_y, p_x + tid_x,
    //                                                width_z, width_y, width_x);
    // __pipeline_wait_prior(0);
    // __syncthreads();

    for (int global_z = p_z -  (QUESIZE) * max(1,(MQSIZE - 1))-halo-1; global_z < p_z_end + (QUESIZE) * (MQSIZE - 1); global_z += 1)
    {
      // preload next step / next QUEUE TILE 
      if (global_z < p_z_end +  (QUESIZE) * (MQSIZE - 1))
      {
        smEnqueueAsync<REAL, 1>(sm_mque + SIZEOFSM - 1,
                                ps_y, ps_x, tile_x_with_halo, tile_y_with_halo,
                                input,
                                global_z + QUESIZE, p_y, p_x,
                                width_z, width_y, width_x);
      }

      // might not be able to unroll
      REAL sum[MQSIZE][LOCAL_ITEM_PER_THREAD];

      _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
      {
        init_reg_array<REAL, LOCAL_ITEM_PER_THREAD>(sum[MQSIZE - step], 0);

        regEnqueue<REAL, LOCAL_ITEM_PER_THREAD>(
            reg_mque[SIZEOFREG - step * (REGQUESIZE) + halo],
            sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 1],
            ps_y + index_y, ps_x + tid_x,
            tile_y_with_halo, tile_x_with_halo);

        // main computation
        computation<REAL, LOCAL_ITEM_PER_THREAD, halo, REG_Y_SIZE_MOD>(sum[MQSIZE - step],
                                                                       sm_mque + SIZEOFSM - (step) * (SMQUESIZE) - 1,
                                                                       ps_y + index_y, tile_x_with_halo, tid_x + ps_x,
                                                                       reg_mque + SIZEOFREG - step * REGQUESIZE,
                                                                       west,east,north,south,top,bottom,center);
      }
      // star version can use multi-buffer to remove the necessarity of two sync
      __syncthreads();
      // lazy update the result
      smEnqueues<REAL, MQSIZE,1, SIZEOFSM, LOCAL_ITEM_PER_THREAD, QUESIZE>(
          sm_mque, ps_y + index_y, ps_x + tid_x, tile_x_with_halo, tile_y_with_halo, sum);

     
      // PERKS-like handling tb dependency
       //   // south
      if (tid_y == 0)
      {
        _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
        {
          _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
          {
            l2_cache_o[step_south +
                       ((step - 1) * gridDim.z + blockIdx.z) * gridDim.y * halo * width_x +
                       (blockIdx.y * halo + l_y) * width_x +
                       c_x + tid_x] = sum[MQSIZE - step][l_y];
          }
        }
      }
      // //north
      else if (tid_y == gdim_y - 1)
      {
        _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
        {
          _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
          {
            l2_cache_o[step_north +
                       ((step - 1) * gridDim.z + blockIdx.z) * gridDim.y * halo * width_x +
                       (blockIdx.y * halo + l_y) * width_x +
                       c_x + tid_x] = sum[MQSIZE - step][LOCAL_ITEM_PER_THREAD - halo + l_y];
          }
        }
      }

      init_reg_array<REAL, LOCAL_ITEM_PER_THREAD>(sum[0], 0);

      regEnqueue<REAL, LOCAL_ITEM_PER_THREAD>(
          reg_mque[halo],
          sm_mque[0],
          ps_y + index_y, ps_x + tid_x,
          tile_y_with_halo, tile_x_with_halo);

      // main computation
      computation<REAL, LOCAL_ITEM_PER_THREAD, halo, REG_Y_SIZE_MOD>(sum[0],
                                                                     sm_mque,
                                                                     ps_y + index_y, tile_x_with_halo, tid_x + ps_x,
                                                                     reg_mque,
                                                                     west,east,north,south,top,bottom,center);

      // star version can use multi-buffer to remove the necessarity of two sync
      int global_z2 = global_z - (QUESIZE) * (MQSIZE - 1);
      if (global_z2 >= p_z && global_z2 < p_z_end && global_z2 < width_z)
      {
        store<REAL, LOCAL_ITEM_PER_THREAD>(output,
                                           global_z2, p_y + index_y, p_x + tid_x, width_z, width_y, width_x, sum[0]);
      }

      // only need to consider boundary
      // west
      __syncthreads();
      // PERKS-like handling tb dependency
      if (tid_y == 1)
      {
        if (tid_x < LOCAL_TILE_Y)
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
            {
              l2_cache_o[step_west +
                         ((step - 1) * gridDim.z + blockIdx.z) * gridDim.x * halo * width_y +
                         (blockIdx.x * halo + l_x) * width_y +
                         c_y + tid_x] =
                  sm_mque
                      [SIZEOFSM - (step) * (SMQUESIZE) - 1]
                      [(ps_y + tid_x) * tile_x_with_halo + ps_x + l_x];
            }
          }
        }
      }
      // //east
      else if (tid_y == 2)
      {
        if (tid_x < LOCAL_TILE_Y)
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
            {
              l2_cache_o[step_east +
                         ((step - 1) * gridDim.z + blockIdx.z) * gridDim.x * halo * width_y +
                         (blockIdx.x * halo + l_x) * width_y +
                         c_y + tid_x] =
                  sm_mque
                      [SIZEOFSM - (step) * (SMQUESIZE) - 1]
                      [(ps_y + tid_x) * tile_x_with_halo + l_x + LOCAL_TILE_X];
            }
          }
        }
      }
      // }
      gg.sync();

      smShuffle<REAL, SIZEOFSM, 1>(sm_mque);
      regShuffle<REAL, MQSIZE, SIZEOFREG, REGQUESIZE,REG_Y_SIZE_MOD,  1, halo>(reg_mque);

      REAL *tmp_ptr = l2_cache_i;
      l2_cache_i = l2_cache_o;
      l2_cache_o = tmp_ptr;
      __pipeline_wait_prior(0);
      // PERKS-like handling tb dependency
      // //south
      if (tid_y == 0)
      {
        if (blockIdx.y == 0)
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
            {
              sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y) + tid_x + ps_x] =
                  (sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (ps_y) + tid_x + ps_x]);
            }
          }
        }
        else
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
            {
              __pipeline_memcpy_async(sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2] + tile_x_with_halo * (l_y) + tid_x + ps_x,
                                      l2_cache_i +
                                          step_north +
                                          (((step - 1) * gridDim.z + blockIdx.z)) * gridDim.y * halo * width_x +
                                          ((blockIdx.y - 1) * halo + l_y) * width_x +
                                          c_x + tid_x,
                                      sizeof(REAL));
            }
          }
        }
      }
      // north
      else if (tid_y == 1)
      {
        if (blockIdx.y == gridDim.y - 1)
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
            {
              sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y + LOCAL_TILE_Y + ps_y) + tid_x + ps_x] =
                  (sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (LOCAL_TILE_Y + ps_y) + tid_x + ps_x]);
            }
          }
        }
        else
        {
          _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
          {
            _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
            {
              __pipeline_memcpy_async(sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2] + tile_x_with_halo * (l_y + LOCAL_TILE_Y + ps_y) + tid_x + ps_x,
                                      l2_cache_i +
                                          step_south +
                                          (((step - 1) * gridDim.z + blockIdx.z)) * gridDim.y * halo * width_x +
                                          ((blockIdx.y + 1) * halo + l_y) * width_x +
                                          c_x + tid_x,
                                      sizeof(REAL));
            }
          }
        }
      }
      // west
      else if (tid_y == 2)
      {
        if (tid_x < LOCAL_TILE_Y)
        {
          int l_y = tid_x;
          if (blockIdx.x == 0)
          {
            _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
            {
              _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
              {
                sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y + ps_y) + l_x] =
                    (sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y + ps_y)]);
              }
            }
          }
          else
          {
            _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
            {
              _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
              {
                __pipeline_memcpy_async(sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2] + tile_x_with_halo * (l_y + ps_y) + l_x,
                                        l2_cache_i +
                                            step_east +
                                            ((step - 1) * gridDim.z + blockIdx.z) * gridDim.x * halo * width_y +
                                            ((blockIdx.x - 1) * halo + l_x) * width_y +
                                            c_y + l_y,
                                        sizeof(REAL));
              }
            }
          }
        }
      }
      // east
      else if (tid_y == 3)
      {
        if (tid_x < LOCAL_TILE_Y)
        {
          int l_y = tid_x;
          if (blockIdx.x == gridDim.x - 1)
          {
            _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
            {
              _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
              {
                sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y + ps_y) + l_x + LOCAL_TILE_X + ps_x] =
                    (sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2][tile_x_with_halo * (l_y + ps_y) + LOCAL_TILE_X + ps_x]);
              }
            }
          }
          else
          {
            _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
            {
              _Pragma("unroll") for (int l_x = 0; l_x < halo; l_x++)
              {
                __pipeline_memcpy_async(sm_mque[SIZEOFSM - (step) * (SMQUESIZE) - 2] + tile_x_with_halo * (l_y + ps_y) + l_x + LOCAL_TILE_X + ps_x,
                                        l2_cache_i +
                                            step_west +
                                            ((step - 1) * gridDim.z + blockIdx.z) * gridDim.x * halo * width_y +
                                            ((blockIdx.x + 1) * halo + l_x) * width_y +
                                            c_y + l_y,
                                        sizeof(REAL));
              }
            }
          }
        }
      }
    }
  }

}

template __global__ void kernel3d_temporal<double, HALO>(double *__restrict__, double *, int, int, int, double *, double *);
template __global__ void kernel3d_temporal<float, HALO>(float *__restrict__, float *, int, int, int, float *, float *);
