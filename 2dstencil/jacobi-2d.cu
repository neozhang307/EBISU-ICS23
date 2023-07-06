#include <cuda.h>
#include <cooperative_groups.h>

#include "cuda_profiler_api.h"
#include "stdio.h"
#include "stdio.h"
#include "assert.h"
#include "config.cuh"

#include "./common/jacobi_cuda.cuh"
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"

#include "../share/printHelper.hpp"
#include "../share/launchHelper.cuh"




namespace cg = cooperative_groups;

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }                                                                                  \
  }

// get code running environment
__global__ void printptx(int *result)
{
  result[0] = PERKS_ARCH;
}
void host_printptx(int &result)
{
  int *d_r;
  cudaMalloc((void **)&d_r, sizeof(int));
  printptx<<<1, 1>>>(d_r);
  cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(d_r);
}

template <class REAL>
void getExperimentSetting(int *iteration, int *widthy, int *widthx, int bdimx)
{

  constexpr int RTILE_Y = (ipts<HALO, curshape, REAL>::val);
  iteration[0] = timesteps<HALO, curshape, RTILE_Y, REAL>::val;
  widthy[0] = widthx[0] = (bdimx - 2 * iteration[0] * HALO) * 36;

}
template void getExperimentSetting<double>(int *, int *, int *, int);
template void getExperimentSetting<float>(int *, int *, int *, int);


/// @brief 
/// @tparam REAL 
/// @param i 
/// @param width_x 
/// @param width_y 
/// @param o 
template <class REAL>
__forceinline__ void exchangeio(REAL*&i, int width_x, int width_y, REAL*&o)
{
  REAL *tmp = i;
  i = o;
  o = tmp;
}

template <class REAL>
int jacobi_iterative(REAL *h_input, int width_y, int width_x, REAL *__var_0__,
                     int bdimx, int blkpsm, int iteration,
                     bool usewarmup, bool verbose)
{
  /*************************************/
  // initialization
  constexpr int RTILE_Y = (ipts<HALO, curshape, REAL>::val);
  constexpr int TSTEP = timesteps<HALO, curshape, RTILE_Y, REAL>::val;
  bdimx = ((bdimx == 128) ? 128 : 256);
  int ptx;
  host_printptx(ptx);
  if (blkpsm <= 0)blkpsm = 100;
  auto execute_kernel = kernel_temporal_traditional<REAL, HALO>;

  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  // initialization input and output space
  REAL *input;
  cudaMalloc(&input, sizeof(REAL) * ((width_y - 0) * (width_x - 0)));
  cudaMemcpy(input, h_input, sizeof(REAL) * ((width_y - 0) * (width_x - 0)), cudaMemcpyHostToDevice);
  REAL *__var_1__;
  cudaMalloc(&__var_1__, sizeof(REAL) * ((width_y - 0) * (width_x - 0)));
  REAL *__var_2__;
  cudaMalloc(&__var_2__, sizeof(REAL) * ((width_y - 0) * (width_x - 0)));

  // initialize shared memory
  int maxSharedMemory;
  cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  // could not use all share memory in a100. so set it in default.
  int SharedMemoryUsed = maxSharedMemory - 1024;
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);

  size_t executeSM = 0;
  int basic_smrange = pow(2, ceil(log((RTILE_Y + HALO) * 2) / log(2)));
  int basic_sm_space = basic_smrange * (bdimx + 2 * HALO) + 1;
  size_t sharememory_basic = (basic_sm_space) * sizeof(REAL);
  executeSM = sharememory_basic;

  int smrange = pow(2, ceil(log((RTILE_Y * 2 + HALO ) + (RTILE_Y + HALO) * (TSTEP - 1)) / log(2)));

  executeSM = smrange * sizeof(REAL) * (bdimx + 2 * isBox) + sizeof(REAL);

  int numBlocksPerSm_current = 1000;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm_current, execute_kernel, bdimx, executeSM);
  cudaDeviceSynchronize();
  cudaCheckError();

  if (blkpsm != 0)
  {
    numBlocksPerSm_current = min(numBlocksPerSm_current, blkpsm);
  }

  dim3 block_dim(bdimx);
  int valid_dimx = bdimx - 2 * HALO * (TSTEP);
  int gimx = (width_x + valid_dimx - 1) / (valid_dimx);
  dim3 grid_dim(gimx, sm_count * numBlocksPerSm_current / (gimx));
  dim3 executeBlockDim = block_dim;
  dim3 executeGridDim = grid_dim;

  LaunchHelper<false> myLauncher=LaunchHelper<false>();
  if (usewarmup)
  {
    myLauncher.warmup(execute_kernel, exchangeio<REAL>,executeGridDim, executeBlockDim, executeSM,0,
      __var_1__, width_y, width_x, __var_2__);
  }

  cudaEvent_t _forma_timer_start_, _forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_, 0);

  {
    myLauncher.launch(execute_kernel, executeGridDim, executeBlockDim, executeSM,0,
      input, width_y, width_x, __var_2__);

    for (int i = TSTEP; i < iteration; i += TSTEP)
    {
      myLauncher.launch(execute_kernel, executeGridDim, executeBlockDim, executeSM,0,
        __var_2__, width_y, width_x, __var_1__);
      REAL *tmp = __var_2__;
      __var_2__ = __var_1__;
      __var_1__ = tmp;
    }
  }


  cudaEventRecord(_forma_timer_stop_, 0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, _forma_timer_start_, _forma_timer_stop_);

  PrintHelper myPrinter((int)verbose);
  myPrinter.PrintPtx(ptx);
  myPrinter.PrintDataType(sizeof(REAL));
  // myPrinter.PrintAsync(async);
  myPrinter.PrintDomain(width_x, width_y);
  myPrinter.PrintIteration(iteration);
  myPrinter.PrintBlockDim(executeBlockDim.x);
  myPrinter.PrintValidThreadBlockTiling(valid_dimx);
  myPrinter.PrintGridDim(executeGridDim.x, executeGridDim.y, executeGridDim.z);
  myPrinter.PrintBlockPerSM((double)(executeGridDim.x) * (executeGridDim.y) / sm_count,
                            numBlocksPerSm_current);
  myPrinter.PrintSharedMemory(SharedMemoryUsed / 1024.0);
  myPrinter.PrintSmRange(smrange, (RTILE_Y * 2 + HALO) + (RTILE_Y + HALO) * (TSTEP - 1));
  myPrinter.PrintPerformance(width_x, width_y, iteration, HALO, FPC, elapsedTime);
  myPrinter.PirntFinish();

  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);

// finalization
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL) * ((width_y - 0) * (width_x - 0)), cudaMemcpyDeviceToHost);

  /*Kernel Launch End */
  /* Host Free Begin */
  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);

  return 0;
  return 0;
}

template int jacobi_iterative<float>(float *h_input, int width_y, int width_x, float *__var_0__,
                                     int bdimx, int blkpsm, int iteration,
                                     bool usewarmup, bool verbose);
template int jacobi_iterative<double>(double *h_input, int width_y, int width_x, double *__var_0__,
                                      int bdimx, int blkpsm, int iteration,
                                      bool usewarmup, bool verbose);