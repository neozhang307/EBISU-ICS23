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
  // stencilParaT;
  const REAL west[2] = {12.0 / 118, 9.0 / 118};
  const REAL east[2] = {12.0 / 118, 9.0 / 118};
  const REAL north[2] = {5.0 / 118, 7.0 / 118};
  const REAL south[2] = {5.0 / 118, 7.0 / 118};
  const REAL center = 15.0 / 118;

  extern __shared__ char sm[];

  REAL *sm_space = (REAL *)sm + 1;
  REAL *sm_rbuffers = sm_space;

  register REAL r_smbuffer[LOCAL_DEPTH][2 * halo + LOCAL_TILE_Y * 2];

  const int tid = threadIdx.x;
  const int ps_x = 0;
  const int rind_x = halo * (LOCAL_DEPTH);

  const int blocksize_x = blockDim.x;
  const int valid_blocksize_x = blocksize_x - LOCAL_DEPTH * halo * 2;
  const int p_x_real = blockIdx.x * valid_blocksize_x - LOCAL_DEPTH * halo;

  const int tile_x_with_halo = blocksize_x;

  int blocksize_y = (width_y + gridDim.y - 1) / gridDim.y;

  const int p_y = blockIdx.y * (blocksize_y);
  const int p_y_end = min(p_y + (blocksize_y), width_y - 1);

  const int p_y_real = (p_y - 2 * halo * LOCAL_DEPTH);
  const int p_y_real_end = p_y_end + (halo + LOCAL_TILE_Y) * (LOCAL_DEPTH - 1);

  const int sizeof_rbuffer = halo * 2 + LOCAL_TILE_Y * 2;

  int local_x = tid;

  /**************/
  /**************/
  // Initialize the shared memory multiqueue indexing
  int sm_mqueue_index[LOCAL_DEPTH];
  // Calculate the index ranging, to do cycular need to get a range that is 2^n
  int sm_range_y_base = (LOCAL_TILE_Y * 2 + halo) + (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1);
  int sm_range_y = __powf(2, ceil(__log2f(sm_range_y_base)));
  sm_mqueue_index[LOCAL_DEPTH - 1] = 0;
  _Pragma("unroll") for (int i = LOCAL_DEPTH - 2; i >= 0; i--)
  {
    sm_mqueue_index[i] = (sm_mqueue_index[i + 1] + (sm_range_y - (halo + LOCAL_TILE_Y))) & (sm_range_y - 1);
  }
  /**************/
  /**************/
  {
    // initial enqueue
    smEnqueueAsync<REAL, halo>(sm_rbuffers, sm_mqueue_index[LOCAL_DEPTH - 1], tid + ps_x, sm_range_y, tile_x_with_halo,
                               input, p_y_real, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
    __pipeline_wait_prior(0);
    __syncthreads();

    regEnqueue<REAL, sizeof_rbuffer, halo, 0>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                              0, sm_range_y,
                                              ps_x, (tid + rind_x) & (blockDim.x - 1),
                                              tile_x_with_halo);

    __syncthreads();
    smEnqueueAsync<REAL, halo + LOCAL_TILE_Y>(sm_rbuffers, sm_mqueue_index[LOCAL_DEPTH - 1], tid + ps_x, sm_range_y, tile_x_with_halo,
                                              input, p_y_real + halo, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
    __pipeline_wait_prior(0);
    __syncthreads();

    regEnqueue<REAL, sizeof_rbuffer, halo, halo>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                                 0, sm_range_y,
                                                 ps_x, (tid + rind_x) & (blockDim.x - 1),
                                                 tile_x_with_halo);
  }
  int global_y = p_y_real + halo;
  // computation of register space
  for (; global_y < p_y_real_end; global_y += LOCAL_TILE_Y)
  {
    // Prefetch next tile
    if (global_y + LOCAL_TILE_Y < p_y_real_end)
    {
      smEnqueueAsync<REAL, LOCAL_TILE_Y>(sm_rbuffers, sm_mqueue_index[LOCAL_DEPTH - 1] + LOCAL_TILE_Y + halo, tid + ps_x, sm_range_y, tile_x_with_halo,
                                         input, global_y + LOCAL_TILE_Y + halo, MIN(MAX(p_x_real + tid, 0), width_x - 1), width_y, width_x);
    }

    REAL sum[LOCAL_DEPTH][LOCAL_TILE_Y];
    // Enqueue current tile to register (it is already loaded in shared memroy)
    regEnqueue<REAL, sizeof_rbuffer, LOCAL_TILE_Y, 2 * halo>(sm_rbuffers, r_smbuffer[(LOCAL_DEPTH - 1)],
                                                             halo + sm_mqueue_index[LOCAL_DEPTH - 1], sm_range_y,
                                                             ps_x, (tid + rind_x) & (blockDim.x - 1),
                                                             tile_x_with_halo);
    _Pragma("unroll") for (int step = 0; step < LOCAL_DEPTH; step++)
    {
      init_reg_array<REAL, LOCAL_TILE_Y>(sum[step], 0);
      // main stencil computation
      compute<REAL, LOCAL_TILE_Y, halo>(sum[step],
                                        sm_rbuffers, sm_mqueue_index[step], sm_range_y,
                                        0, (local_x + rind_x), tile_x_with_halo, blockDim.x,
                                        r_smbuffer[step], halo,
                                        west, east, north, south, center);
    }
    // Lazy enqueue for next tiling in the following time steps.
    regEnqueues<REAL, LOCAL_DEPTH, LOCAL_TILE_Y * 2 + 2 * halo, LOCAL_TILE_Y, LOCAL_TILE_Y + 2 * halo, 0, LOCAL_TILE_Y>(r_smbuffer, sum);
    smEnqueues<REAL, LOCAL_DEPTH, LOCAL_TILE_Y, LOCAL_TILE_Y>(sm_rbuffers, sm_mqueue_index, 0, ((tid + rind_x) & (blockDim.x - 1)) + ps_x, sm_range_y, tile_x_with_halo, sum);
    // shuffle
    {
      regShuffle<REAL, LOCAL_DEPTH, 2 * halo + LOCAL_TILE_Y * 2, LOCAL_TILE_Y, LOCAL_TILE_Y + 2 * halo, 2 * halo>(r_smbuffer);
      smShuffle<REAL, LOCAL_DEPTH, LOCAL_TILE_Y>(sm_mqueue_index, sm_range_y);
    }
    // store
    if (tid < valid_blocksize_x)
    {
      store<REAL, LOCAL_TILE_Y>(__var_4__, sum[0],
                                width_x, width_y,
                                p_y, p_y_end,
                                p_x_real + tid + rind_x,
                                global_y - (LOCAL_TILE_Y + halo) * (LOCAL_DEPTH - 1),
                                LOCAL_TILE_Y);
    }
    // sync the prefetch
    __pipeline_wait_prior(0);
    // lazy streaming can reduce the amount of synchronization to only one synchronization.
    __syncthreads();
  }
}

template __global__ void kernel_temporal_traditional<double, HALO>(double *__restrict__, int, int, double *);
template __global__ void kernel_temporal_traditional<float, HALO>(float *__restrict__, int, int, float *);
