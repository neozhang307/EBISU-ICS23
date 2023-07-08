

#include <cuda.h>
#include "stdio.h"
#include <cooperative_groups.h>
#include "stdio.h"
#include "assert.h"
#include "config.cuh"

#include "./common/jacobi_cuda.cuh"
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#include "./common/temporalconfig.cuh"


#include "../share/printHelper.hpp"
#include "../share/launchHelper.cuh"



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
}

template <class REAL>
void getExperimentSetting(int *iteration, int *height, int *widthy, int *widthx, int bdimx)
{
  static constexpr int const ITEM_PER_THREAD = (ipts<HALO, curshape, REAL>::val);
  int TSTEP = timesteps<HALO, curshape, ITEM_PER_THREAD, REAL>::val;
  int TILE_X = ipts<HALO, curshape, REAL>::tile_x;
  int TILE_Y = ipts<HALO, curshape, REAL>::val * bdimx / TILE_X;
  iteration[0] = TSTEP;
  widthx[0] = 12 * TILE_X;
  widthy[0] = 9 * TILE_Y;
  height[0] = 2560;
}

template void getExperimentSetting<double>(int *, int *, int *, int *, int);
template void getExperimentSetting<float>(int *, int *, int *, int *, int);

template <class REAL>
int j3d_iterative(REAL *h_input,
                  int height, int width_y, int width_x,
                  REAL *__var_0__,
                  int global_bdimx,
                  int blkpsm,
                  int iteration,
                  bool usewarmup,
                  bool verbose)
{
  static constexpr int const ITEM_PER_THREAD = (ipts<HALO, curshape, REAL>::val);
  LaunchHelper<true> myLauncher = LaunchHelper<true>();
  int TSTEP = timesteps<HALO, curshape, ITEM_PER_THREAD, REAL>::val;
  const int LOCAL_ITEM_PER_THREAD = ITEM_PER_THREAD;
  global_bdimx = global_bdimx == 128 ? 128 : 256;

  int TILE_X = ipts<HALO, curshape, REAL>::tile_x;

  int TILE_Y = LOCAL_ITEM_PER_THREAD * global_bdimx / TILE_X;
  assert(TILE_Y <= TILE_X);

  if (blkpsm <= 0) blkpsm = 100;

  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

  int ptx;
  host_printptx(ptx);

  auto execute_kernel = (kernel3d_temporal<REAL, HALO>);

  // shared memory related
  size_t executeSM = 0;
  int smrange = (quesize<HALO, curshape>::smque) * TSTEP + (curshape==1);
  // int smrange = (HALO+1) * TSTEP + 1;
  int basic_sm_space = ((TILE_Y + 2 * HALO) * (TILE_X + HALO * 2) * smrange )* sizeof(REAL);

  executeSM = basic_sm_space;

  int maxSharedMemory;
  cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  int SharedMemoryUsed = maxSharedMemory - 1024;
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);

  int numBlocksPerSm_current = 100;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm_current, execute_kernel, global_bdimx, executeSM);
  cudaCheckError();
  if (blkpsm <= 0)
    blkpsm = numBlocksPerSm_current;
  numBlocksPerSm_current = min(blkpsm, numBlocksPerSm_current);

  dim3 block_dim_3(global_bdimx, 1, 1);
  int gdimx = width_x / TILE_X;
  int gdimy = width_y / TILE_Y;
  int gdimz = MAX(1, sm_count * numBlocksPerSm_current / gdimx / gdimy);
  dim3 grid_dim_3(gdimx, gdimy, gdimz);
  dim3 executeBlockDim = block_dim_3;
  dim3 executeGridDim = grid_dim_3;
  
  if (numBlocksPerSm_current == 0)
  {
    return -1;
  }
  if (executeGridDim.z * (2 * HALO + 1) > height)
  {
    return -2;
  }

  REAL *input;
  cudaMalloc(&input, sizeof(REAL) * (height * width_x * width_y));
  Check_CUDA_Error("Allocation Error!! : input\n");

  cudaGetLastError();
  cudaMemcpy(input, h_input, sizeof(REAL) * (height * width_x * width_y), cudaMemcpyHostToDevice);
  REAL *__var_1__;
  cudaMalloc(&__var_1__, sizeof(REAL) * (height * width_x * width_y));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL *__var_2__;
  cudaMalloc(&__var_2__, sizeof(REAL) * (height * width_x * width_y));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");

  // L2 cache persistent might be useful
  size_t L2_utage = (TSTEP * executeGridDim.z * executeGridDim.y * HALO * width_x * 2 + TSTEP * executeGridDim.x * executeGridDim.z * HALO * width_y * 2);
  REAL *l2_cache;
  cudaMalloc(&l2_cache, L2_utage * sizeof(REAL) * 2);
  REAL *l2_cache1 = l2_cache;
  REAL *l2_cache2 = l2_cache + L2_utage;
  size_t inner_window_size = 30 * 1024 * 1024;
  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(l2_cache);                   // Global Memory data pointer
  stream_attribute.accessPolicyWindow.num_bytes = min(inner_window_size, L2_utage * sizeof(REAL) * 2); // Number of bytes for persistence access
  stream_attribute.accessPolicyWindow.hitRatio = 1;                                                    // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;                          // Persistence Property
  stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

  cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
  cudaCtxResetPersistingL2Cache();
  cudaStreamSynchronize(0);

  cudaCheckError();

  if (usewarmup)
  {
    myLauncher.warmup(execute_kernel, executeGridDim, executeBlockDim, executeSM, 0,
                      __var_2__, __var_1__,
                      height, width_y, width_x,
                      l2_cache1, l2_cache2);
  }

  cudaEvent_t _forma_timer_start_, _forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_, 0);

  myLauncher.launch(execute_kernel, executeGridDim, executeBlockDim, executeSM, 0,
                    input, __var_2__,
                    height, width_y, width_x,
                    l2_cache1, l2_cache2);

  for (int i = TSTEP; i < iteration; i += TSTEP)
  {
    myLauncher.launch(execute_kernel, executeGridDim, executeBlockDim, executeSM, 0,
                      __var_2__, __var_1__,
                      height, width_y, width_x,
                      l2_cache1, l2_cache2);

    REAL *tmp = __var_2__;
    __var_2__ = __var_1__;
    __var_1__ = tmp;
  }

  cudaEventRecord(_forma_timer_stop_, 0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, _forma_timer_start_, _forma_timer_stop_);

  PrintHelper myPrinter((int)verbose);
  myPrinter.PrintPtx(ptx);
  myPrinter.PrintDataType(sizeof(REAL));
  myPrinter.PrintDomain(width_x, width_y, height);
  myPrinter.PrintIteration(iteration);
  myPrinter.PrintIteration(TSTEP);

  myPrinter.PrintBlockDim(executeBlockDim.x);
  myPrinter.PrintILP(LOCAL_ITEM_PER_THREAD);
  myPrinter.PrintGridDim(executeGridDim.x, executeGridDim.y, executeGridDim.z);
  myPrinter.PrintBlockPerSM((double)(executeGridDim.x) * (executeGridDim.y) / sm_count,
                            numBlocksPerSm_current);
  myPrinter.PrintSharedMemory((double)executeSM / 1024.0, SharedMemoryUsed / 1024.0);
  myPrinter.PrintSmRange(smrange, smrange);
  myPrinter.PrintPerformance(width_x, width_y, height, iteration, HALO, FPC, elapsedTime);
  myPrinter.PirntFinish();

  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);

  cudaDeviceSynchronize();
  cudaCheckError();

  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL) * height * width_x * width_y, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);
  cudaFree(l2_cache);

  return 0;
}

template int j3d_iterative<float>(float *h_input,
                                  int height, int width_y, int width_x,
                                  float *__var_0__,
                                  int global_bdimx,
                                  int blkpsm,
                                  int iteration,
                                  bool usewarmup,
                                  bool verbose);

template int j3d_iterative<double>(double *h_input,
                                   int height, int width_y, int width_x,
                                   double *__var_0__,
                                   int global_bdimx,
                                   int blkpsm,
                                   int iteration,
                                   bool usewarmup,
                                   bool verbose);
