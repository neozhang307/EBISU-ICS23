#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>

#include "config.cuh"
#include "./common/cuda_computation_back.cuh"
#include "./common/cuda_common.cuh"

namespace cg = cooperative_groups;
template <class REAL, int halo, int LOCAL_TILE_Y, int LOCAL_DEPTH>
__global__ void kernel_temporal_traditional(REAL *__restrict__ input, int width_y, int width_x,
                                            REAL *__var_4__)
{


  stencilParaT;
  // const REAL filter[2*halo+1][2*halo+1];

  extern __shared__ char sm[];

  REAL *sm_space = (REAL *)sm + 1;
  REAL *sm_rbuffers = sm_space;


  register REAL r_smbuffer[LOCAL_DEPTH][2 * halo + 1][2 * halo + LOCAL_TILE_Y];


  const int tid = threadIdx.x;
  const int ps_x = halo;

  const int blocksize_x = blockDim.x;
  const int valid_blocksize_x = blocksize_x - LOCAL_DEPTH * halo * 2;
  const int p_x_real = blockIdx.x * valid_blocksize_x - LOCAL_DEPTH * halo;

  const int tile_x_with_halo = blocksize_x;

  int blocksize_y = (width_y + gridDim.y - 1) / gridDim.y;

  const int p_y = blockIdx.y * (blocksize_y);
  const int p_y_end = p_y + (blocksize_y);

  const int p_y_real = (p_y - 2 * halo * LOCAL_DEPTH);                          //-((halo*LOCAL_DEPTH+LOCAL_TILE_Y-1)/LOCAL_TILE_Y)*LOCAL_TILE_Y-halo;//(LOCAL_TILE_Y)*(LOCAL_DEPTH-1);//(p_y - LOCAL_DEPTH*(halo)*2);
  const int p_y_real_end = p_y_end + (halo + LOCAL_TILE_Y) * (LOCAL_DEPTH - 1); //(p_y_end + LOCAL_DEPTH*(halo+LOCAL_TILE_Y));


  const int sizeof_rbuffer = halo * 2 + LOCAL_TILE_Y;


  int sm_range_y_base = (LOCAL_TILE_Y * 2 + halo*2) + (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1);
  int sm_range_y = __powf(2, ceil(__log2f(sm_range_y_base)));

  {
    int local_x = tid;
    int basesm_y[LOCAL_DEPTH];

    basesm_y[LOCAL_DEPTH - 1] = 0;
    _Pragma("unroll") for (int i = LOCAL_DEPTH - 2; i >= 0; i--)
    {
      basesm_y[i] = (basesm_y[i + 1] + (sm_range_y - (halo + LOCAL_TILE_Y))) & (sm_range_y - 1);
    }

    _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
    {
      int l_global_y;
      {
        l_global_y = (MAX(p_y_real + l_y, 0));
        l_global_y = (MIN(l_global_y, width_y - 1));
      }

      __pipeline_memcpy_async(sm_rbuffers + (l_y + basesm_y[LOCAL_DEPTH - 1]) * tile_x_with_halo + tid + ps_x,
                              input + (l_global_y)*width_x + MIN(MAX(p_x_real + tid, 0), width_x - 1) // MAX(p_x-halo+tid,0)
                              ,
                              sizeof(REAL));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    sm2regsv4<REAL, sizeof_rbuffer, halo, isBox>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                              0, sm_range_y,
                                              ps_x, tid, // rind_x,
                                              tile_x_with_halo);
    __syncthreads();
    _Pragma("unroll") for (int l_y = 0; l_y < halo + LOCAL_TILE_Y; l_y++)
    {
      int l_global_y;
      {
        l_global_y = (MAX(p_y_real + l_y + halo, 0));
        l_global_y = (MIN(l_global_y, width_y - 1));
      }

      __pipeline_memcpy_async(sm_rbuffers + (l_y + basesm_y[LOCAL_DEPTH - 1]) * tile_x_with_halo + tid + ps_x,
                              input + (l_global_y)*width_x + MIN(MAX(p_x_real + tid, 0), width_x - 1) // MAX(p_x-halo+tid,0)
                              ,
                              sizeof(REAL));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    sm2regsv4<REAL, sizeof_rbuffer, halo, isBox>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                              0, sm_range_y,
                                              ps_x, tid, // rind_x,
                                              tile_x_with_halo, halo);
    // computation of register space
    for (int global_y = p_y_real + halo; global_y < p_y_real_end; global_y += LOCAL_TILE_Y)
    {
      // preload next slice
      if (global_y + LOCAL_TILE_Y < p_y_real_end)
      {

        _Pragma("unroll") for (int l_y = 0; l_y < LOCAL_TILE_Y; l_y++)
        {
          int l_global_y;
          l_global_y = (MAX(global_y + LOCAL_TILE_Y + halo + l_y, 0));
          l_global_y = (MIN(l_global_y, width_y - 1));

          __pipeline_memcpy_async(sm_rbuffers + ((l_y + basesm_y[LOCAL_DEPTH - 1] + LOCAL_TILE_Y + halo) & (sm_range_y - 1)) * tile_x_with_halo + tid + ps_x,
                                  input + (l_global_y)*width_x + MIN(MAX(p_x_real + tid, 0), width_x - 1), sizeof(REAL));
        }
      }
      __pipeline_commit();

      // //loading middle slice
      // //computation of middle slice
      REAL sum[LOCAL_DEPTH][LOCAL_TILE_Y];

      _Pragma("unroll") for (int step = 1; step < LOCAL_DEPTH; step++)
      {
        sm2regsv4<REAL, sizeof_rbuffer, LOCAL_TILE_Y, isBox>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - step)],
                                                          halo + basesm_y[LOCAL_DEPTH - step], sm_range_y,
                                                          ps_x, tid, // rind_x,
                                                          tile_x_with_halo, 2 * halo);

        init_reg_array<REAL, LOCAL_TILE_Y>(sum[step], 0);
        // main computation
        computation_box<REAL, LOCAL_TILE_Y, halo>(sum[step],
                                               sm_rbuffers, basesm_y[LOCAL_DEPTH - step], sm_range_y,
                                               local_x + ps_x, tile_x_with_halo,
                                               r_smbuffer[LOCAL_DEPTH - step], halo,
                                               stencilParaInput);
      }
      __syncthreads();
      sm2regsv4<REAL, sizeof_rbuffer, LOCAL_TILE_Y, isBox>(sm_rbuffers, r_smbuffer[(0)],
                                                        halo + basesm_y[0], sm_range_y,
                                                        ps_x, tid, // rind_x,
                                                        tile_x_with_halo, 2 * halo);

      _Pragma("unroll") for (int step = 1; step < LOCAL_DEPTH; step++)
      {
        _Pragma("unroll") for (int l_y = 0; l_y < LOCAL_TILE_Y; l_y++)
        {
          sm_rbuffers[((l_y + basesm_y[LOCAL_DEPTH - step]) & (sm_range_y - 1)) * tile_x_with_halo + ps_x + tid] //+((tid+rind_x)&(blockDim.x-1))]
              = sum[step][l_y];
        }
      }

      init_reg_array<REAL, LOCAL_TILE_Y>(sum[0], 0);
      // main computation

      computation_box<REAL, LOCAL_TILE_Y, halo>(sum[0],
                                             sm_rbuffers, basesm_y[0], sm_range_y,
                                             local_x + ps_x,
                                             tile_x_with_halo,
                                             r_smbuffer[0], halo, stencilParaInput);
      // store to global
      if (tid >= halo * (LOCAL_DEPTH) && tid < valid_blocksize_x + halo * (LOCAL_DEPTH))
      {
        _Pragma("unroll") for (int l_y = 0; l_y < LOCAL_TILE_Y; l_y++)
        {
          int l_global_y = global_y + l_y - (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1);
          int l_global_x = p_x_real + tid; //+rind_x;
          {
            if (l_global_y >= p_y_end |
                l_global_y >= width_y |
                l_global_x >= width_x)
              break;
            if (l_global_y < p_y)
            {
              continue;
            }
          }
          __var_4__[(l_global_y)*width_x + l_global_x] = sum[0][l_y];
        }
      }
      // some data in shared memroy can be used in next tiling.
      _Pragma("unroll") for (int step = 0; step < LOCAL_DEPTH - 1; step++)
      {

        regs2regs<REAL, sizeof_rbuffer, sizeof_rbuffer, LOCAL_TILE_Y + 2 * halo, isBOX>(r_smbuffer[step], r_smbuffer[step], LOCAL_TILE_Y, 0);
      }
      {
        regs2regs<REAL, sizeof_rbuffer, sizeof_rbuffer, 2 * halo, isBOX>(r_smbuffer[LOCAL_DEPTH - 1], r_smbuffer[LOCAL_DEPTH - 1], LOCAL_TILE_Y, 0);
      }
      _Pragma("unroll") for (int i = 0; i < LOCAL_DEPTH; i++)
      {
        basesm_y[i] = (basesm_y[i] + LOCAL_TILE_Y) & (sm_range_y - 1);
      }
      __pipeline_wait_prior(0);
      __syncthreads();
    }
  }

}

template __global__ void kernel_temporal_traditional<double, HALO>(double *__restrict__, int, int, double *);
template __global__ void kernel_temporal_traditional<float, HALO>(float *__restrict__, int, int, float *);
