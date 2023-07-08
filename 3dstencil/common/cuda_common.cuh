// #include "cuda.h
#pragma once
#include "stdio.h"

#ifndef CUDACOMMON
#define CUDACOMMON

#ifndef __CUDA_ARCH__
    #define PERKS_ARCH 000
#else
    #if __CUDA_ARCH__==800
        #define PERKS_ARCH 800
    #elif __CUDA_ARCH__==700
        #define PERKS_ARCH 700
    #elif __CUDA_ARCH__==600
        #define PERKS_ARCH 600
    #else
        #error "unsupport"
    #endif
#endif


#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>



extern __host__ __device__ __forceinline__ int MAX(int a, int b) { return a > b ? a : b; }
extern __host__ __device__ __forceinline__ int MIN(int a, int b) { return a < b ? a : b; }
extern __host__ __device__ __forceinline__ int CEIL(int a, int b) { return ( (a) % (b) == 0 ? (a) / (b) :  ( (a) / (b) + 1 ) ); }


#define Check_CUDA_Error(message) \
  do{\
    cudaError_t error = cudaGetLastError();\
    if( error != cudaSuccess ){\
      printf("CUDA-ERROR:%s, %s\n",message,cudaGetErrorString(error) ); \
      exit(-1);\
    }\
  }while(0)

template<class REAL, int SIZE>
__device__ void __forceinline__ init_reg_array(REAL reg_array[SIZE], int val)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<SIZE; l_y++)
  {
    reg_array[l_y]=val;
  }
}


#endif