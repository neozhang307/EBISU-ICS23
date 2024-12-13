// #include "./common/common.hpp"
// #include <cooperative_groups.h>
// #include <cuda.h>
// #include "stdio.h"
// #include "./common/cuda_computation.cuh"
// #include "./common/cuda_common.cuh"
// #include "./common/types.hpp"
#ifdef GEN
#include "./genconfig.cuh"
#endif


#ifdef _TIMER_
  #include "cuda_profiler_api.h"
#endif
#include <cuda.h>
#include "stdio.h"
#include <cooperative_groups.h>
#include "stdio.h"
#include "assert.h"
#include "config.cuh" 
#include "./common/jacobi_cuda.cuh"
#include "./common/types.hpp"
#include "./common/cuda_common.cuh"
#include "./common/cuda_computation.cuh"
#if defined(TEMPORAL)||defined(TRATEMPORAL)
#include "./common/temporalconfig.cuh"
#endif
// #define TILE_X 256
// #define NAIVE
#if defined(NAIVE)||defined(BASELINE)||defined(BASELINE_CM)||defined(TRATEMPORAL)
  #define TRADITIONLAUNCH
#endif
#if defined(GEN)||defined(PERSISTENT)||defined(GENWR)||defined(TEMPORAL)
  #define PERSISTENTLAUNCH
#endif
#if defined PERSISTENTLAUNCH||defined(BASELINE_CM)||defined(TRATEMPORAL)
  #define PERSISTENTTHREAD
#endif
#if defined(BASELINE)||defined(BASELINE_CM) ||defined(GEN)||defined(GENWR)||defined(PERSISTENT)||defined(TEMPORAL)||defined(TRATEMPORAL)
  #define USEMAXSM
#endif

#if defined(TEMPORAL)||defined(TRATEMPORAL)
  #define ITEM_PER_THREAD (ipts<HALO,curshape,REAL>::val)
#else 
  #define ITEM_PER_THREAD (8)
#endif

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}


__global__ void printptx(int *result)
{
  // printf("code is run in %d\n",PERKS_ARCH);
  result[0]=PERKS_ARCH;
}
void host_printptx(int&result)
{
  int*d_r;
  cudaMalloc((void**)&d_r, sizeof(int));
  printptx<<<1,1>>>(d_r);
  cudaMemcpy(&result, d_r, sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}

template<class REAL>
int getMinWidthY(int width_x, int width_y, int global_bdimx, bool isDoubleTile)
{
  
  int minwidthy1 = j3d_iterative<REAL>(nullptr,
                            100000, width_y, width_x,
                            nullptr, 
                            global_bdimx, 
                            1, 
                            1, 
                            false,
                            false, 
                            0,
                            isDoubleTile,
                            true);

  int minwidthy2 = j3d_iterative<REAL>(nullptr,
                            100000, width_y, width_x,
                            nullptr, 
                            global_bdimx, 
                            2, 
                            1, 
                            false,
                            false, 
                            0,
                            isDoubleTile,
                            true);
  int minwidthy3 = j3d_iterative<REAL>(nullptr,
                            100000, width_y, width_x,
                            nullptr, 
                            global_bdimx, 
                            1, 
                            1, 
                            true,
                            false, 
                            0,
                            isDoubleTile,
                            true);
  int minwidthy4 = j3d_iterative<REAL>(nullptr,
                            100000, width_y, width_x,
                            nullptr, 
                            global_bdimx, 
                            2, 
                            1, 
                            true,
                            false, 
                            0,
                            isDoubleTile,
                            true);

  int result = max(minwidthy1,minwidthy2);
  result = max(result,minwidthy3);
  result = max(result,minwidthy4);
  return result;
}

template int getMinWidthY<float>  (int, int, int, bool);
template int getMinWidthY<double> (int, int, int, bool);


template<class REAL>void getExperimentSetting(int* iteration, int*height,int*widthy, int*widthx,int bdimx)
{
  int arch;
  host_printptx(arch);
  int basey=9;
  if(arch==800)
  {
    basey=9;
  }
  else if(arch==900)
  {
    basey=11;
  }

  int TSTEP=timesteps<HALO, curshape,  ITEM_PER_THREAD,  REAL>::val;
  int TILE_X=ipts<HALO,curshape,REAL>::tile_x;
  int TILE_Y=ipts<HALO,curshape,REAL>::val*bdimx/TILE_X;
    iteration[0]=TSTEP;
    widthx[0]=12*(TILE_X-2*TSTEP*HALO);
    widthy[0]=basey*(TILE_Y-2*TSTEP*HALO);
    height[0]=2560;
}
template void getExperimentSetting<double>(int*,int*,int*,int*,int);
template void getExperimentSetting<float> (int*,int*,int*,int*,int);


template<class REAL>
int j3d_iterative(REAL * h_input,
  int height, int width_y, int width_x,
  REAL * __var_0__, 
  int global_bdimx, 
  int blkpsm, 
  int iteration, 
  bool useSM,
  bool usewarmup, 
  int warmupiteration,
  bool isDoubleTile,
  bool getminHeight)
{

#if defined(TEMPORAL)||defined(TRATEMPORAL)
  int TSTEP=timesteps<HALO, curshape,  ITEM_PER_THREAD,  REAL>::val;
  int TEMPSTEP=TSTEP;
#else
  int TSTEP=0;

#endif

  const int LOCAL_ITEM_PER_THREAD=ITEM_PER_THREAD;
  global_bdimx=global_bdimx==128?128:256;
  if(isDoubleTile)
  {
    if(global_bdimx==256)blkpsm=1;
    if(global_bdimx==128)blkpsm=min(blkpsm,2);
  }
  #if defined(TEMPORAL)||defined(TRATEMPORAL)
    int TILE_X=ipts<HALO,curshape,REAL>::tile_x;
  #endif
  int TILE_Y = LOCAL_ITEM_PER_THREAD*global_bdimx/TILE_X;
  assert(TILE_Y<=TILE_X);
  if(blkpsm<=0)blkpsm=100;

  int sm_count;
  cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
#ifndef __PRINT__
  printf("sm_count is %d\n",sm_count);
#endif
  int ptx;
  host_printptx(ptx);
#ifndef __PRINT__
#endif


#ifdef PERSISTENT
  auto execute_kernel = isDoubleTile?(global_bdimx==128?kernel3d_persistent<REAL,HALO,2*ITEM_PER_THREAD,TILE_X,128>
                              :kernel3d_persistent<REAL,HALO,2*ITEM_PER_THREAD,TILE_X,256>):
                                (global_bdimx==128?kernel3d_persistent<REAL,HALO,ITEM_PER_THREAD,TILE_X,128>
                              :kernel3d_persistent<REAL,HALO,ITEM_PER_THREAD,TILE_X,256>);
#endif


  auto execute_kernel =         (kernel3d_temporal_traditional<REAL,HALO,256>);


                 
//shared memory related 
size_t executeSM=0;
#ifndef NAIVE

    // int basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+HALO*2)*(2+HALO*2 )* TSTEP +1)*sizeof(REAL);
    int basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+HALO*2)*((1+HALO*1)* TSTEP+1) +1)*sizeof(REAL);
    #ifdef BASELINE
      basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+HALO*2)*(2+HALO*1+isBOX)+1)*sizeof(REAL);
    #elif defined(BOX)
      // basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+HALO*2)*((2+2*HALO*1)* TSTEP) +1)*sizeof(REAL);
      basic_sm_space=((TILE_Y+2*HALO)*(TILE_X+HALO*2)*((2+HALO*1)* TSTEP) +1)*sizeof(REAL);
    #endif
    executeSM=basic_sm_space;
    // printf("size of sm is %d\n",basic_sm_space);
#endif


  int maxSharedMemory;
  cudaDeviceGetAttribute (&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,0 );
  int SharedMemoryUsed=maxSharedMemory-1024;
  cudaFuncSetAttribute(execute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemoryUsed);




#ifdef PERSISTENTLAUNCH
  int max_sm_flder=0;
#endif

// printf("asdfjalskdfjaskldjfals;");

#if defined(PERSISTENTTHREAD)
  int numBlocksPerSm_current=100;

  #if defined(GEN)
    int reg_folder_z=REG_FOLDER_Z;
  #endif

  #if defined(GEN)||defined(GENWR)

    executeSM+=reg_folder_z*2*HALO*(TILE_Y+TILE_X+2*isBOX);
  #endif 
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm_current, execute_kernel, global_bdimx, executeSM);
  cudaCheckError();
  if(blkpsm<=0)blkpsm=numBlocksPerSm_current;
  numBlocksPerSm_current=min(blkpsm,numBlocksPerSm_current);
  // numBlocksPerSm_current=1;
  dim3 block_dim_3(global_bdimx, 1, 1);
  // dim3 grid_dim_3((width_x+TILE_X+1)/TILE_X-1, width_y/TILE_Y, MIN(height, MAX(1,sm_count*numBlocksPerSm_current/(width_y/TILE_Y*(width_x+TILE_X+1)/(TILE_X-1)))));

#ifndef TRATEMPORAL
  dim3 grid_dim_3((width_x)/TILE_X, width_y/TILE_Y, MIN(height, MAX(1,sm_count*numBlocksPerSm_current/(width_y/TILE_Y*(width_x)/(TILE_X)))));
#else
  int valid_dimx = TILE_Y-2*HALO*(TSTEP);
  int valid_dimy = TILE_Y-2*HALO*(TSTEP);
  int gimx = (width_x+valid_dimx-1)/(valid_dimx);
  int gimy = (width_y+valid_dimy-1)/(valid_dimy);
  dim3 grid_dim_3(gimx, gimy, MIN(height, MAX(1,sm_count*numBlocksPerSm_current/gimy/gimx)));
#endif
  dim3 executeBlockDim=block_dim_3;
  dim3 executeGridDim=grid_dim_3;

  if(numBlocksPerSm_current==0)return -3;
#endif
  int minHeight=0;

  if(executeGridDim.z*(2*HALO+1)>height)return -4;

  if(getminHeight)return (minHeight);

  REAL * input;
  cudaMalloc(&input,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : input\n");

  cudaGetLastError();
  cudaMemcpy(input,h_input,sizeof(REAL)*(height*width_x*width_y), cudaMemcpyHostToDevice);
  REAL * __var_1__;
  cudaMalloc(&__var_1__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_1__\n");
  REAL * __var_2__;
  cudaMalloc(&__var_2__,sizeof(REAL)*(height*width_x*width_y));
  Check_CUDA_Error("Allocation Error!! : __var_2__\n");


  REAL ** h_buffers=(REAL**)malloc(sizeof(REAL*)*(TSTEP+2));
  for(int i=0; i<(TSTEP+1); i++)
  {
    cudaMalloc(&h_buffers[i],sizeof(REAL)*(height*width_x*width_y));
  }
  h_buffers[0]=input;
  h_buffers[TSTEP+1]=__var_2__;

  REAL ** d_buffers;
  cudaMalloc(&d_buffers,sizeof(REAL*)*(TSTEP+2));
  cudaMemcpy(d_buffers,h_buffers,sizeof(REAL*)*(TSTEP+2), cudaMemcpyHostToDevice);

  // size_t L2_utage = width_y*height*sizeof(REAL)*HALO*(width_x/TILE_X)*2+
  //                   width_x*height*sizeof(REAL)*HALO*(width_y/TILE_Y)*2  ;
  size_t L2_utage = 2*(
    TSTEP*executeGridDim.z*executeGridDim.y*HALO*width_x*2+TSTEP*executeGridDim.x*executeGridDim.z*HALO*width_y
  );
  // width_y*height*sizeof(REAL)*HALO*(width_x/TILE_X)*2+
                    // width_x*height*sizeof(REAL)*HALO*(width_y/TILE_Y)*2  ;
  REAL * l2_cache;
  cudaMalloc(&l2_cache,L2_utage);
  REAL * l2_cache1=l2_cache;
  REAL * l2_cache2=l2_cache+(TSTEP*executeGridDim.z*executeGridDim.y*HALO*width_x*2+TSTEP*executeGridDim.x*executeGridDim.z*HALO*width_y);

  // REAL l2perused;
  size_t inner_window_size = 30*1024*1024;
  cudaStreamAttrValue stream_attribute;
  stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(l2_cache);                  // Global Memory data pointer
  stream_attribute.accessPolicyWindow.num_bytes = min(inner_window_size,L2_utage);                                   // Number of bytes for persistence access
  stream_attribute.accessPolicyWindow.hitRatio  = 1;                                             // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;                  // Persistence Property
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  

  cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute); 
  cudaCtxResetPersistingL2Cache();
  cudaStreamSynchronize(0);

  REAL ** h_caches=(REAL**)malloc(sizeof(REAL*)*(TSTEP));
  for(int i=0; i<(TSTEP); i++)
  {
    cudaMalloc(&h_caches[i],L2_utage);
  }

  REAL ** d_caches;
  cudaMalloc(&d_caches,sizeof(REAL*)*(TSTEP));
  cudaMemcpy(d_caches,h_caches,sizeof(REAL*)*(TSTEP), cudaMemcpyHostToDevice);

#ifndef __PRINT__
  printf("l2 cache used is %ld KB : 4096 KB \n",L2_utage/1024);
#endif

  int l_warmupiteration=warmupiteration>0?warmupiteration:1000;

#ifdef PERSISTENTLAUNCH
  int l_iteration=iteration;
  #ifndef TEMPORAL
  void* KernelArgs[] ={(void**)&input,(void*)&__var_2__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void**)&l2_cache1, (void**)&l2_cache2,
    (void*)&l_iteration,(void*)&max_sm_flder};
  // #ifdef __PRINT__  
  void* KernelArgsNULL[] ={(void**)&__var_2__,(void*)&__var_1__,
      (void**)&height,(void**)&width_y,(void*)&width_x,
      (void**)&l2_cache1, (void**)&l2_cache2,
      (void*)&l_warmupiteration,(void*)&max_sm_flder};
  #else
  void* KernelArgs[] ={(void**)&input,(void*)&__var_2__,
    (void**)&height,(void**)&width_y,(void*)&width_x,
    (void**)&l2_cache1, (void**)&l2_cache2,
    (void**)&d_buffers,(void**)&d_caches,
    (void*)&l_iteration};
  // #ifdef __PRINT__  
  void* KernelArgsNULL[] ={(void**)&__var_2__,(void*)&__var_1__,
      (void**)&height,(void**)&width_y,(void*)&width_x,
      (void**)&l2_cache1, (void**)&l2_cache2,
      (void**)&d_buffers,(void**)&d_caches,
      (void*)&l_warmupiteration};      
  #endif
#endif
cudaCheckError();
// bool warmup=false;
if(usewarmup)
{
  cudaEvent_t warstart,warmstop;
  cudaEventCreate(&warstart);
  cudaEventCreate(&warmstop);
  #ifdef TRADITIONLAUNCH
  {
    cudaEventRecord(warstart,0);
    // cudaCheckError();
    for(int i=0; i<l_warmupiteration; i++)
    {
      // execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
      //       (__var_2__, width_y, width_x , __var_1__);
      execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (__var_2__, __var_1__,  height, width_y, width_x);
      REAL* tmp = __var_2__;
      __var_2__=__var_1__;
      __var_1__= tmp;

    } 
    cudaEventRecord(warmstop,0);
    cudaEventSynchronize(warmstop);
    cudaCheckError();
    float warmelapsedTime;
    cudaEventElapsedTime(&warmelapsedTime,warstart,warmstop);
    float nowwarmup=(warmelapsedTime);
    // nowwarmup = max()
    int nowiter=(350+nowwarmup-1)/nowwarmup;

    for(int out=0; out<nowiter; out++)
    {
      for(int i=0; i<l_warmupiteration; i++)
      {
        // execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
              // (__var_2__, width_y, width_x , __var_1__);
        execute_kernel<<<executeGridDim, executeBlockDim, executeSM>>>
          (__var_2__, __var_1__,  height, width_y, width_x);
        REAL* tmp = __var_2__;
        __var_2__=__var_1__;
        __var_1__= tmp;
      }       
    }

  }
  #endif 
  
  #ifdef PERSISTENTLAUNCH
  {
      // double accumulate=0;
      cudaEventRecord(warstart,0);
      cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgsNULL, executeSM,0);
      cudaEventRecord(warmstop,0);
      cudaEventSynchronize(warmstop);
      cudaCheckError();
      float warmelapsedTime;
      cudaEventElapsedTime(&warmelapsedTime,warstart,warmstop);
      int nowwarmup=warmelapsedTime;
      int nowiter=(350+nowwarmup-1)/nowwarmup;
      for(int i=0; i<nowiter; i++)
      {
        cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgsNULL, executeSM,0);
      }
  }
  #endif
}
cudaDeviceSynchronize();

#ifdef _TIMER_
  cudaEvent_t _forma_timer_start_,_forma_timer_stop_;
  cudaEventCreate(&_forma_timer_start_);
  cudaEventCreate(&_forma_timer_stop_);
  cudaEventRecord(_forma_timer_start_,0);
#endif


#ifdef TRADITIONLAUNCH
  execute_kernel<<<executeGridDim, executeBlockDim,executeSM>>>
          (input, __var_2__,  height, width_y, width_x);

  for(int i=TSTEP; i<iteration; i+=TSTEP)
  {
     execute_kernel<<<executeGridDim, executeBlockDim,executeSM>>>
          (__var_2__, __var_1__, height, width_y, width_x);
    REAL* tmp = __var_2__;
    __var_2__=__var_1__;
    __var_1__= tmp;
  }
#endif
#ifdef PERSISTENTLAUNCH
  cudaLaunchCooperativeKernel((void*)execute_kernel, executeGridDim, executeBlockDim, KernelArgs, executeSM,0);
#endif
  cudaDeviceSynchronize();
  cudaCheckError();
#ifdef _TIMER_
  cudaEventRecord(_forma_timer_stop_,0);
  cudaEventSynchronize(_forma_timer_stop_);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime,_forma_timer_start_,_forma_timer_stop_);
#ifndef __PRINT__
  printf("[FORMA] SIZE : %d,%d,%d\n",height,width_y,width_x);
  printf("[FORMA] Computation Time(ms) : %lf\n",elapsedTime);
  printf("[FORMA] Speed(GCells/s) : %lf\n",(REAL)iteration*height*width_x*width_y/ elapsedTime/1000/1000);
  printf("[FORMA] Computation(GFLOPS/s) : %lf\n",(REAL)iteration*height*width_x*width_y*(HALO*2+1)*(HALO*2+1)/ elapsedTime/1000/1000);
  printf("[FORMA] Bandwidth(GB/s) : %lf\n",(REAL)iteration*height*width_x*width_y*sizeof(REAL)*2/ elapsedTime/1000/1000);
#if defined(GEN)||defined(GENWR)  
  printf("[FORMA] rfder : %d\n",reg_folder_z);
#endif
#ifdef PERSISTENTLAUNCH
  printf("[FORMA] sfder : %d\n",max_sm_flder);
  // printf("[FORMA] sm : %f\n",executeSM/1024);
#endif
#else
  // h y x iter TILEX thready=1 gridx gridy latency speed 
  printf("%d\t%d\t",ptx,(int)sizeof(REAL)/4);
  printf("%d\t%d\t%d\t%d\t",height,width_y,width_x,iteration); 
  printf("%d\t%d\t<%d,%d,%d>\t%d\t%d\t",executeBlockDim.x,LOCAL_ITEM_PER_THREAD,
        executeGridDim.x,executeGridDim.y,executeGridDim.z,sm_count,
        (executeGridDim.x)*(executeGridDim.y)*(executeGridDim.z)/sm_count);
  #ifndef NAIVE
  printf("%f\t",(double)basic_sm_space/1024);
  #endif
  printf("%f\t%f\t%lf\t",(double)executeSM/1024,elapsedTime,(REAL)iteration*(height-2*HALO*iteration)*(width_x-2*HALO*iteration)*(width_y-2*HALO*iteration)/ elapsedTime/1000/1000); 
  printf("\n");
#endif
  cudaEventDestroy(_forma_timer_start_);
  cudaEventDestroy(_forma_timer_stop_);
#endif
  cudaDeviceSynchronize();
  cudaCheckError();

  
#if defined(PERSISTENTLAUNCH) 
// || defined(PERSISTENT)
  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);

  if(iteration%(TEMPSTEP*2)==TEMPSTEP)  
  {
    cudaMemcpy(__var_0__, __var_2__, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
  }
  else
  {
    cudaMemcpy(__var_0__, input, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
  }
#else

  cudaMemcpy(__var_0__, __var_2__, sizeof(REAL)*height*width_x*width_y, cudaMemcpyDeviceToHost);
#endif
  cudaDeviceSynchronize();
  cudaCheckError();

  cudaFree(input);
  cudaFree(__var_1__);
  cudaFree(__var_2__);
  cudaFree(l2_cache);
  // cudaFree(l2_cache2);
  for(int i=1; i<TSTEP+1; i++)
  {
    cudaFree(h_buffers[i]);
  }
  for(int i=0; i<TSTEP; i++)
  {
    cudaFree(h_caches[i]);
  }
  cudaFree(d_buffers);
  cudaFree(d_caches);
  free(h_buffers);
  free(h_caches);
  return 0;
}

PERKS_INITIALIZE_ALL_TYPE(PERKS_DECLARE_INITIONIZATION_ITERATIVE);

// template void j3d_iterative<float>(float * h_input, int height, int width_y, int width_x, float * __var_0__, int iteration);
// template void j3d_iterative<double>(float * h_input, int height, int width_y, int width_x, float * __var_0__, int iteration);
