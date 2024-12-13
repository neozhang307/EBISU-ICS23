#include "./config.cuh"

#include "./common/cuda_computation.cuh"
#include "./common/cuda_common.cuh"
#include "./common/types.hpp"
#include <math.h>

#include <cooperative_groups.h>


#include <cooperative_groups/memcpy_async.h>
#include <cuda_pipeline.h>


namespace cg = cooperative_groups;
// #define isBOX halo
// #define TSTEP (1)

template<class REAL, int halo,  int BLOCKDIM,int LOCAL_TILE_X,int LOCAL_ITEM_PER_THREAD,  int TSTEP>
__global__ void 
kernel3d_temporal_traditional(REAL * __restrict__ input, 
                                REAL *  output, 
                                int width_z, int width_y, int width_x//,
                                // REAL* l2_cache_i, REAL* l2_cache_o
                                )
{
  #define LOCAL_TILE_Y (LOCAL_ITEM_PER_THREAD*BLOCKDIM/LOCAL_TILE_X)
  #define gdim_y (BLOCKDIM/LOCAL_TILE_X)
  #define sizeofsm ((halo*1+1)*(TSTEP)+1)
  #define sizeofreg ((halo+1)*(TSTEP))


  const int tile_x_with_halo=LOCAL_TILE_X+2*halo;
  const int tile_y_with_halo=LOCAL_TILE_Y+2*halo;
  stencilParaT;
  extern __shared__ char sm[];
  REAL* sm_rbuffer = (REAL*)sm+1;

  register REAL r_smbuffer[sizeofreg][REG_Y_SIZE_MOD];

  REAL* smbuffer_buffer_ptr[sizeofsm];
  smbuffer_buffer_ptr[0]=sm_rbuffer;
  #pragma unroll
  for(int hl=1; hl<sizeofsm; hl++)
  {
    smbuffer_buffer_ptr[hl]=smbuffer_buffer_ptr[hl-1]+tile_x_with_halo*tile_y_with_halo;
  }

  const int tid_x = threadIdx.x%LOCAL_TILE_X;
  const int tid_y = threadIdx.x/LOCAL_TILE_X;
  const int index_y = LOCAL_ITEM_PER_THREAD*tid_y;
  const int ps_y = halo;
  const int ps_x = halo;


  const int blocksize_x = LOCAL_TILE_X;
  const int blocksize_y = LOCAL_TILE_Y;
  const int valid_blocksize_x = blocksize_x-TSTEP*halo*2;
  const int p_x_real = blockIdx.x * valid_blocksize_x - TSTEP*halo;
  const int valid_blocksize_y = blocksize_y-TSTEP*halo*2;
  const int p_y_real = blockIdx.y * valid_blocksize_y - TSTEP*halo;

  const int p_x = p_x_real;//blockIdx.x * LOCAL_TILE_X;
  const int p_y = p_y_real;//blockIdx.y * LOCAL_TILE_Y;


  int blocksize_z=((width_z+gridDim.z-1)/gridDim.z);
  const int p_z =  blockIdx.z * (blocksize_z);
  const int p_z_end =  p_z + (blocksize_z); 

  const int p_z_real     = p_z-(halo+1)*(TSTEP-1);//-((halo*LOCAL_STEP+LOCAL_TILE_Y-1)/LOCAL_TILE_Y)*LOCAL_TILE_Y-halo;//(LOCAL_TILE_Y)*(LOCAL_STEP-1);//(p_y - LOCAL_STEP*(halo)*2);
  const int p_z_real_end = p_z_end+(halo+1)*(TSTEP-1); //(p_y_end + LOCAL_STEP*(halo+LOCAL_TILE_Y));

  {
    for(int l_z=0; l_z<halo+1; l_z++)
    {
      for(int lid=threadIdx.x; lid<tile_x_with_halo*tile_y_with_halo; lid+=blockDim.x)
      {
        int l_x=lid%tile_x_with_halo-halo;
        int l_y=lid/tile_x_with_halo-halo;

        // int l_global_z=(MIN(p_z-2*halo*(TSTEP-1)+l_z,width_z-1));
        int l_global_z=(MIN(p_z_real+l_z,width_z-1));
        l_global_z=(MAX(l_global_z,0));

        int l_global_y = (MIN(p_y+l_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));
        int l_global_x = (MIN(p_x+l_x,width_x-1));
          l_global_x = (MAX(l_global_x,0));
        __pipeline_memcpy_async(smbuffer_buffer_ptr[l_z+sizeofsm-1-halo-1]+tile_x_with_halo*(l_y+ps_y)+l_x+ps_x, 
              input+l_global_z*width_x*width_y+l_global_y*width_x+l_global_x , sizeof(REAL));
      } 
    }

    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    _Pragma("unroll")
    for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
    {
      _Pragma("unroll")
      for(int l_z=0; l_z<halo ; l_z++)
      {
        int l_global_z=(MIN(p_z_real+l_z-halo,width_z-1));
        l_global_z=(MAX(l_global_z,0));

        int l_global_y = (MIN(l_y+index_y+p_y,width_y-1));
          l_global_y = (MAX(l_global_y,0));
        int l_global_x = (MIN(p_x+tid_x,width_x-1));
          l_global_x = (MAX(l_global_x,0));

        r_smbuffer[l_z+sizeofreg-(halo+1)][l_y] = input[l_global_z*width_x*width_y+l_global_y*width_x+l_global_x];// smbuffer_buffer_ptr[l_z+sizeofsm-(2*halo+2)][(l_y+ps_y+index_y)*tile_x_with_halo+ps_x+tid_x];//input[(global_y) * width_x + global_x];
      }
    }

    
    for(int global_z=p_z_real; global_z<p_z_real_end; global_z+=1)
    {
      //preload next step
      if(global_z<p_z_end+(halo+1)*(TSTEP-1))
      {

        int l_global_z=(MIN(global_z+halo+1,width_z-1));
        l_global_z=(MAX(l_global_z,0));
        for(int lid=threadIdx.x; lid<tile_x_with_halo*tile_y_with_halo; lid+=blockDim.x)
        {
          int l_x=lid%tile_x_with_halo-halo;
          int l_y=lid/tile_x_with_halo-halo;

          int l_global_y = (MIN(p_y+l_y,width_y-1)); 
            l_global_y = (MAX(l_global_y,0));
          int l_global_x = (MIN(p_x+l_x,width_x-1));
            l_global_x = (MAX(l_global_x,0));

          __pipeline_memcpy_async(smbuffer_buffer_ptr[sizeofsm-1]+tile_x_with_halo*(l_y+ps_y)+l_x+ps_x, 
                input+l_global_z*width_x*width_y+l_global_y*width_x+l_global_x , sizeof(REAL));
        }
      }
      __pipeline_commit();
      // might not be ble to unroll
      // __syncthreads();
      REAL sum[TSTEP][LOCAL_ITEM_PER_THREAD];

        //sm2reg
      //need to unroll
      _Pragma("unroll")
      for(int step=1; step<TSTEP; step++)
      {
        // REAL sum[step][LOCAL_ITEM_PER_THREAD];
        _Pragma("unroll")
        for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
        {
          sum[step][l_y]=0;
        }

        _Pragma("unroll")
        for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
        {
          r_smbuffer[sizeofreg-step*(halo+1)+halo][l_y] = smbuffer_buffer_ptr[sizeofsm-(step)*(1*halo+1)-1][(ps_y+index_y+l_y)*tile_x_with_halo+ps_x+tid_x];
        }
        //main computation
        computation<REAL,LOCAL_ITEM_PER_THREAD,halo,REG_Y_SIZE_MOD>(sum[step],
                                          smbuffer_buffer_ptr+sizeofsm-(step)*(1*halo+1)-1,
                                          ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                          r_smbuffer+sizeofreg-step*(halo+1),
                                          stencilParaInput,isBOX);
      }
        //star version can use multi-buffer to remove the necessarity of two sync
      __syncthreads();
      _Pragma("unroll")
      for(int step=1; step<TSTEP; step++)
      {
        //the following part only onlly need to store boundary
        _Pragma("unroll")
        for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
        {
          smbuffer_buffer_ptr[sizeofsm-(step)*(halo+1)-1][(ps_y+index_y+l_y)*tile_x_with_halo+ps_x+tid_x]=sum[step][l_y];
        }
      }
      // __syncthreads();


      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
        sum[0][l_y]=0;
      }
      _Pragma("unroll")
      for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
      {
          r_smbuffer[halo][l_y] = smbuffer_buffer_ptr[0][(ps_y+index_y+l_y)*tile_x_with_halo+ps_x+tid_x];
      }
      //main computation
      computation<REAL,LOCAL_ITEM_PER_THREAD,halo,REG_Y_SIZE_MOD>(sum[0],
                                        smbuffer_buffer_ptr,
                                        ps_y+index_y, tile_x_with_halo, tid_x+ps_x,
                                        r_smbuffer,
                                        stencilParaInput,isBOX);
      
      //star version can use multi-buffer to remove the necessarity of two sync
      // // reg 2 ptr
      int global_z2=global_z-(halo+1)*(TSTEP-1);
      // int global_z2=global_z-(2*halo+1)*(TSTEP-1);
      if(tid_x>=halo*TSTEP&&tid_x<blocksize_x-halo*TSTEP)
      {
        _Pragma("unroll")
        for(int l_y=0; l_y<LOCAL_ITEM_PER_THREAD; l_y++)
        {
          if(p_x+tid_x>=width_x)break;
          if(p_y+index_y+l_y>=width_y)break;

          if(index_y+l_y>=blocksize_y-halo*TSTEP)break;
          if(index_y+l_y<halo*TSTEP)continue;

          if(global_z2<p_z)break;
          if(global_z2>=p_z_end)break;
          if(global_z2>=width_z)break;
          output   [global_z2*width_x*width_y+(p_y+index_y+l_y)*width_x+p_x+tid_x]=sum[0][l_y];
        }
      }
      


      // __syncthreads();
      // gg.sync();
      
       REAL* tmp = smbuffer_buffer_ptr[0];
      // smswap 
      _Pragma("unroll")
      for(int hl=1; hl<sizeofsm; hl++)
      {
        smbuffer_buffer_ptr[hl-1]=smbuffer_buffer_ptr[hl];
      }
      smbuffer_buffer_ptr[sizeofsm-1]=tmp;
  //  __syncthreads();
      _Pragma("unroll")
      for(int l_z=0; l_z<TSTEP; l_z++)
      {
        _Pragma("unroll")
        for(int l_h=0; l_h<HALO; l_h++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<REG_Y_SIZE_MOD; l_y++)
          {
          
            r_smbuffer[l_z*(halo+1)+l_h][l_y]=r_smbuffer[l_z*(halo+1)+1+l_h][l_y];
          }
        }
      }

      // REAL* tmp_ptr =l2_cache_i;
      // l2_cache_i=l2_cache_o;
      // l2_cache_o=tmp_ptr;

      __pipeline_wait_prior(0);
      __syncthreads();
    }

  }
  #undef LOCAL_TILE_X
  #undef gdim_y
  #undef sizeofsm
  #undef sizeofreg
}

template __global__ void kernel3d_temporal_traditional<double,HALO>(double*__restrict__,double*,int,int,int);
template __global__ void kernel3d_temporal_traditional<float,HALO>(float*__restrict__,float*,int,int,int);
