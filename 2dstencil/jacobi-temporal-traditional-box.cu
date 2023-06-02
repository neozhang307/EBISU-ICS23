#include <math.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>

#include "config.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include "./common/multi_queue.cuh"

namespace cg = cooperative_groups;
template <class REAL, int halo, int LOCAL_TILE_Y, int LOCAL_DEPTH>
__global__ void kernel_temporal_traditional(REAL *__restrict__ input, int width_y, int width_x,
                                            REAL *__var_4__)
{

  stencilParaT;
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

  int local_x = tid;

  /**************/
  /**************/
  // Initialize the shared memory multiqueue indexing
  // Calculate the index ranging, to do cycular need to get a range that is 2^n
  int sm_range_y_base = (LOCAL_TILE_Y * 2 + halo) + (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1);
  int sm_range_y = __powf(2, ceil(__log2f(sm_range_y_base)));
  int basesm_y[LOCAL_DEPTH];
  basesm_y[LOCAL_DEPTH - 1] = 0;
  _Pragma("unroll") for (int i = LOCAL_DEPTH - 2; i >= 0; i--)
  {
    basesm_y[i] = (basesm_y[i + 1] + (sm_range_y - (halo + LOCAL_TILE_Y))) & (sm_range_y - 1);
  }
  /**************/
  /**************/
  {
    smEnqueueAsync<REAL, halo>(sm_rbuffers, basesm_y[LOCAL_DEPTH - 1], tid + ps_x, sm_range_y, tile_x_with_halo,
                               input, p_y_real, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
    __pipeline_wait_prior(0);
    __syncthreads();

    regsEnqueues<REAL, sizeof_rbuffer, LOCAL_TILE_Y, isBox, 0>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                                 0, sm_range_y,
                                                 ps_x, tid, // rind_x,
                                                 tile_x_with_halo);
    __syncthreads();

    smEnqueueAsync<REAL, halo + LOCAL_TILE_Y>(sm_rbuffers, basesm_y[LOCAL_DEPTH - 1], tid + ps_x, sm_range_y, tile_x_with_halo,
                                              input, p_y_real + halo, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
    __pipeline_wait_prior(0);
    __syncthreads();
    regsEnqueues<REAL, sizeof_rbuffer, LOCAL_TILE_Y, isBox, halo>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                                 0, sm_range_y,
                                                 ps_x, tid, // rind_x,
                                                 tile_x_with_halo);
    // computation of register space
    for (int global_y = p_y_real + halo; global_y < p_y_real_end; global_y += LOCAL_TILE_Y)
    {
      // preload next slice
      if (global_y + LOCAL_TILE_Y < p_y_real_end)
      {
        smEnqueueAsync<REAL, LOCAL_TILE_Y>(sm_rbuffers, basesm_y[LOCAL_DEPTH - 1] + LOCAL_TILE_Y + halo, tid + ps_x, sm_range_y, tile_x_with_halo,
                                         input, global_y + LOCAL_TILE_Y + halo, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
      }

      REAL sum[LOCAL_DEPTH][LOCAL_TILE_Y];

      _Pragma("unroll") for (int step = 1; step <= LOCAL_DEPTH; step++)
      {
        //due to register pressure, no reigster lazy streaming
        regsEnqueues<REAL, sizeof_rbuffer, LOCAL_TILE_Y, isBox, 2 * halo>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - step)],
                                                             halo + basesm_y[LOCAL_DEPTH - step], sm_range_y,
                                                             ps_x, tid, // rind_x,
                                                             tile_x_with_halo);
        init_reg_array<REAL, LOCAL_TILE_Y>(sum[LOCAL_DEPTH - step], 0);
        // main computation
        computation_box<REAL, LOCAL_TILE_Y, halo>(sum[LOCAL_DEPTH - step],
                                                  sm_rbuffers, basesm_y[LOCAL_DEPTH - step], sm_range_y,
                                                  local_x + ps_x, tile_x_with_halo,
                                                  r_smbuffer[LOCAL_DEPTH - step], halo,
                                                  filter);
      }
      // lazy streaming can not further reduce synchronization
      // because can not further apply register lazy streaming due to possibily register pressure
      __syncthreads();
      smEnqueues<REAL, LOCAL_DEPTH, LOCAL_TILE_Y, LOCAL_TILE_Y>(sm_rbuffers, basesm_y, 0, tid + ps_x, sm_range_y, tile_x_with_halo, sum);
      // shuffle
      {
        regShuffle<REAL, LOCAL_DEPTH, 2 * halo +1, 2*halo+ LOCAL_TILE_Y , 2 * halo +1, LOCAL_TILE_Y>(r_smbuffer);
        smShuffle<REAL, LOCAL_DEPTH, LOCAL_TILE_Y>(basesm_y, sm_range_y);
      }
      // store to global
      if (tid >= halo * (LOCAL_DEPTH) && tid < valid_blocksize_x + halo * (LOCAL_DEPTH))
      {
        store<REAL, LOCAL_TILE_Y>(__var_4__, sum[0],
                                  width_x, width_y,
                                  p_y, p_y_end,
                                  p_x_real + tid, // + rind_x,
                                  global_y - (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1),
                                  LOCAL_TILE_Y);
      }
      
      __pipeline_wait_prior(0);
      __syncthreads();
    }
  }
}

template __global__ void kernel_temporal_traditional<double, HALO>(double *__restrict__, int, int, double *);
template __global__ void kernel_temporal_traditional<float, HALO>(float *__restrict__, int, int, float *);
