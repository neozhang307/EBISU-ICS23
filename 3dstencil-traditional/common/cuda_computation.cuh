#pragma once
#define ISINITI (true)
#define NOTINITIAL (false)
#define SYNC (true)
#define NOSYNC (false)

//#undef TILE_Y
// #define USESM

// #ifdef USESM
//   #define USESMSET (true)
// #else
//   #define USESMSET (false)
// #endif



// #define bdimx (256)
// #define bdimx (128)

// #define ITEM_PER_THREAD (8) 
// #ifndef ITEM_PER_THREAD
//   #define ITEM_PER_THREAD (8) 
// #endif
// (TILE_Y/(bdimx/TILE_X))
// #define ITEM_PER_THREAD 8
#if defined(TEMPORAL)||defined(TRATEMPORAL)
#else
#ifndef TILE_X
  #define TILE_X (32)
#endif
#endif
// #define TILE_Y (ITEM_PER_THREAD*bdimx/TILE_X)

// #define NOCACHE_Y (HALO)

#ifndef NOCACHE_Z
#define NOCACHE_Z (HALO)
#endif
// (32)

// #define RTILE_Z (1)

// #ifndef X_FACTOR
// #define X_FACTOR (16)
// #endif
// #ifndef RFOLDER_Z
// #define RFOLDER_Z (3)
// #endif
// #ifndef SFOLDER_Z
// #define SFOLDER_Z (0)
// #endif 
// #define FOLDER_Z (RFOLDER_Z+SFOLDER_Z)
// #define SKIP (1)
// #define SM_X (TILE_X  + 2 * halo)
// #define SM_Y (TILE_Y  + 2 * halo)
// #define SM_H (RTILE_Z + 2 * halo)
// #ifndef RFOLDER_Z
// #define RFOLDER_Z (3)
// #endif


#ifndef BOX
  #define curshape (star_shape)
  #if HALO==1
    #define stencilParaT \
      const REAL center=-1.67f;\
      const REAL west[1]={0.162f};\
      const REAL east[1]={0.161f};\
      const REAL north[1]={0.163f};\
      const REAL south[1]={0.164f};\
      const REAL bottom[1]={0.166f};\
      const REAL top[1]={0.165f};
    #endif
    #if HALO==2
      #define stencilParaT \
        const REAL center=-0.996f;\
        const REAL west[2]={0.083f,0.083f};\
        const REAL east[2]={0.083f,0.083f};\
        const REAL north[2]={0.083f,0.083f};\
        const REAL south[2]={0.083f,0.083f};\
        const REAL bottom[2]={0.083f,0.083f};\
        const REAL top[2]={0.083f,0.083f};
    #endif
    #define isBOX (0)
    #define stencilParaList const REAL west[HALO],const REAL east[HALO],const REAL north[HALO],const REAL south[HALO],const REAL top[HALO], const REAL bottom[HALO], const REAL center
    #define stencilParaInput  west,east,north,south,top,bottom,center
    #define REG_Y_SIZE_MOD (LOCAL_ITEM_PER_THREAD)
#else
  #ifndef TYPE0
    #define curshape (box_shape)
    #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0.5/159, 0.7/159, 0.90/159},\
          {1.2/159, 1.5/159, 1.2/159},\
          {0.9/159, 0.7/159, 0.50/159}\
        },\
        { {0.51/159, 0.71/159, 0.91/159},\
          {1.21/159, 1.51/159, 1.21/159},\
          {0.91/159, 0.71/159, 0.51/159}\
        },\
        { {0.52/159, 0.72/159, 0.920/159},\
          {1.22/159, 1.52/159, 1.22/159},\
          {0.92/159, 0.72/159, 0.520/159}\
        }\
      };
  #else
    #ifdef POISSON
      #define curshape (poisson_shape)
      #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        },\
        { {-0.0833f,        -0.166f,   -0.0833f},\
          {-0.166f, 2.666f,     -0.166f},\
          {-0.0833f,       -0.166f,    -0.0833f}\
        },\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        }\
      };
    #else
      #define curshape (type0_shape)
      #define stencilParaT \
        const REAL filter[3][3][3] = {\
          { {0.50/159,  0.0,  0.50/159},\
            {0.0,   0.0,  0.0},\
            {0.50/159,  0.0,  0.50/159}\
          },\
          { {0.51/159,  0.71/159, 0.91/159},\
            {1.21/159,  1.51/159, 1.21/159},\
            {0.91/159,  0.71/159, 0.51/159}\
          },\
          { {0.52/159,  0.0,  0.52/159},\
            {0.0,   0.0,  0.0},\
            {0.52/159,  0.0,  0.52/159}\
          }\
        };
    #endif
  #endif
  
  #define stencilParaList const REAL filter[halo*2+1][halo*2+1][halo*2+1]
  #define stencilParaInput  filter
  #define isBOX (HALO)
  #define REG_Y_SIZE_MOD (LOCAL_ITEM_PER_THREAD+2*halo)
#endif


// template<class REAL, int RESULT_SIZE, int halo, int SMZ_SIZE=halo+1+halo, int REGZ_SIZE=2*halo+1, int REGY_SIZE=REG_Y_SIZE_MOD, int REGX_SIZE=2*halo+1, int REG_BASE=halo>
//I hope that 
template<class REAL, int RESULT_SIZE, int halo, int REGY_SIZE,  int REGZ_SIZE=2*halo+1, int REGX_SIZE=2*halo+1, int REG_BASE=halo, int SMZ_SIZE=halo+1+halo>
__device__ void __forceinline__ computation(REAL result[RESULT_SIZE],
                                            REAL* sm_ptr[SMZ_SIZE], 
                                            int sm_y_base, int sm_width, int sm_x_ind,
                                            REAL reg_ptr[REGZ_SIZE][RESULT_SIZE],
                                            const REAL west[HALO],const REAL east[HALO],const REAL north[HALO],const REAL south[HALO],const REAL top[HALO], const REAL bottom[HALO], const REAL center,
                                            int SM_BASE=0)
{
    _Pragma("unroll")
    for(int hl=0; hl<halo; hl++)
    {
       _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE; l_y++)
      {
        result[l_y]+=bottom[hl]*reg_ptr[REG_BASE-1-hl][l_y];
      }
    }
    _Pragma("unroll")
    for(int l_y=0; l_y<halo; l_y++)
    {
      // int l_y=0;
      int sm_y_ind=sm_width*(l_y+sm_y_base);
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=west[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind-1-hl];
        result[l_y]+=east[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind+1+hl];
        result[l_y]+=north[hl]*
          sm_ptr[SM_BASE][sm_width*(1+hl)+sm_y_ind + sm_x_ind];
        result[l_y]+=south[hl]*
          sm_ptr[SM_BASE][-sm_width*(1+hl)+sm_y_ind + sm_x_ind];
      }
    }
    _Pragma("unroll")
    for(int l_y=halo; l_y<RESULT_SIZE-halo; l_y++)
    {
      int sm_y_ind=sm_width*(l_y+sm_y_base);
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=west[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind-1-hl];
        result[l_y]+=east[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind+1+hl];
      }
    }
    _Pragma("unroll")
    for(int hl=0; hl<halo; hl++)
    {
      _Pragma("unroll")
      for(int l_y=halo; l_y<RESULT_SIZE-halo; l_y++)
      {
      // int sm_y_ind=sm_width*(l_y+sm_y_base);
      
        result[l_y]+=
            north[hl]*reg_ptr[REG_BASE][l_y+1+hl];
          // sm_ptr[SM_BASE][sm_width*(1+hl)+sm_y_ind + sm_x_ind];
        result[l_y]+=
            south[hl]*reg_ptr[REG_BASE][l_y-1-hl];
          // sm_ptr[SM_BASE][-sm_width*(1+hl)+sm_y_ind + sm_x_ind];
      }
    }
    _Pragma("unroll")
    for(int l_y=RESULT_SIZE-halo; l_y<RESULT_SIZE; l_y++)
    {
      // int l_y=RESULT_SIZE-1;
      int sm_y_ind=sm_width*(l_y+sm_y_base);
      _Pragma("unroll")
      for(int hl=0; hl<halo; hl++)
      {
        result[l_y]+=west[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind-1-hl];
        result[l_y]+=east[hl]*
          sm_ptr[SM_BASE][sm_y_ind + sm_x_ind+1+hl];
        result[l_y]+=north[hl]*
          sm_ptr[SM_BASE][sm_width*(1+hl)+sm_y_ind + sm_x_ind];
        result[l_y]+=south[hl]*
          sm_ptr[SM_BASE][-sm_width*(1+hl)+sm_y_ind + sm_x_ind];
      }
    }

     _Pragma("unroll")
    for(int hl=0; hl<halo; hl++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE; l_y++)
      {
      
        // result[l_y]+=top[hl]*reg_ptr[REG_BASE+1+hl][l_y];
        result[l_y]+=top[hl]* sm_ptr[SM_BASE+hl+1][sm_width*(l_y+sm_y_base) + sm_x_ind];
            // reg_ptr[REG_BASE+1+hl][l_y];
      }
    }
    // }
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE; l_y++)
    {
      result[l_y]+=center*reg_ptr[REG_BASE][l_y];
      // result[l_y]+=center*sm_ptr[SM_BASE][sm_width*(l_y+sm_y_base) + sm_x_ind];
    }

}

template<class REAL, int RESULT_SIZE, int halo, int REGY_SIZE,  int REGZ_SIZE=2*halo+1, int SMZ_SIZE=halo+1+halo>
__device__ void __forceinline__ computation_box(REAL sum[RESULT_SIZE],
                                            REAL* smbuffer_buffer_ptr[SMZ_SIZE], int SMZ_BASE,
                                            int sm_y_base, int sm_width, int sm_x_ind,
                                            REAL r_smbuffer[REGZ_SIZE][REGY_SIZE][2*halo+1],int REGZ_BASE, 
                                            const REAL filter[halo*2+1][halo*2+1][halo*2+1])
{

  #ifdef POISSON
    // _Pragma("unroll")
    // for(int hl_z=-halo; hl_z<1; hl_z++)
    {
      int hl_z=-1;
      {
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][0+halo]*
              r_smbuffer[hl_z+REGZ_BASE][-1+halo+l_y][0+halo];
          }
        }
      }
      {
        int hl_y=0;
        _Pragma("unroll")
        for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              r_smbuffer[hl_z+REGZ_BASE][hl_y+halo+l_y][hl_x+halo];
          }
        }
      }
      {
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][0+halo]*
              r_smbuffer[hl_z+REGZ_BASE][1+halo+l_y][0+halo];
          }
        }
      }
    }
    // _Pragma("unroll")
    // for(int hl_z=-halo; hl_z<1; hl_z++)
    {
      int hl_z=0;
      _Pragma("unroll")
      for(int hl_y=-halo; hl_y<halo+1; hl_y++)
      {
        _Pragma("unroll")
        for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              r_smbuffer[hl_z+REGZ_BASE][hl_y+halo+l_y][hl_x+halo];
          }
        }
      }
    }
    // _Pragma("unroll")
    // for(int hl_z=1; hl_z<halo+1; hl_z++)
    {
      int hl_z=1;
      // _Pragma("unroll")
      // for(int hl_y=-halo; hl_y<halo+1; hl_y++)
      {
        // int hl_y=-1;
        // _Pragma("unroll")
        // for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][0+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+-1+sm_y_base)*sm_width+0+sm_x_ind];
          }
        }
      }
      // _Pragma("unroll")
      // for(int hl_y=0; hl_y<halo+1; hl_y++)
      {
        int hl_y=0;
        _Pragma("unroll")
        for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+hl_y+sm_y_base)*sm_width+hl_x+sm_x_ind];
          }
        }
      }
      // _Pragma("unroll")
      // for(int hl_y=0; hl_y<halo+1; hl_y++)
      {
        // int hl_y=1;
        // _Pragma("unroll")
        // for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][0+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+1+sm_y_base)*sm_width+0+sm_x_ind];
          }
        }
      }
      // _Pragma("unroll")
      // for(int hl_y=-halo; hl_y<halo+1; hl_y++)
      // {
      //   // _Pragma("unroll")
      //   // for(int hl_x=-halo; hl_x<halo+1; hl_x++)
      //   {
      //     _Pragma("unroll")
      //     for(int l_y=0; l_y<RESULT_SIZE; l_y++)
      //     {
      //       sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
      //         smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+hl_y+sm_y_base)*sm_width+hl_x+sm_x_ind];
      //     }
      //   }
      // }
    }

  #else
    _Pragma("unroll")
    for(int hl_z=-halo; hl_z<1; hl_z++)
    {
      _Pragma("unroll")
      for(int hl_y=-halo; hl_y<halo+1; hl_y++)
      {
        _Pragma("unroll")
        for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              r_smbuffer[hl_z+REGZ_BASE][hl_y+halo+l_y][hl_x+halo];
          }
        }
      }
    }
    _Pragma("unroll")
    for(int hl_z=1; hl_z<halo+1; hl_z++)
    {
      _Pragma("unroll")
      for(int hl_y=-halo; hl_y<halo+1; hl_y++)
      {
        _Pragma("unroll")
        for(int hl_x=-halo; hl_x<halo+1; hl_x++)
        {
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][hl_y+halo][hl_x+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+hl_y+sm_y_base)*sm_width+hl_x+sm_x_ind];
          }
        }
      }
    }
  #endif
}


template<class REAL, int halo>
__global__ void kernel3d_restrict(REAL* input, REAL* output,
                                  int height, int width_y, int width_x); 
#define PERKS_DECLARE_INITIONIZATION_NAIVE(_type,halo) \
    __global__ void kernel3d_restrict<_type,halo>(_type*,_type*,int,int,int);

template<class REAL, int halo , int ipt, int tilex, int blockdim=256>
__global__ void kernel3d_baseline(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x); 
#define PERKS_DECLARE_INITIONIZATION_BASELINE(_type,halo,ipt,tilex,blockdim) \
    __global__ void kernel3d_baseline<_type,halo,ipt,tilex,blockdim>(_type*__restrict__,_type*__restrict__,int,int,int);


// template<class REAL, int halo , int ipt, int tilex, int tiley>
// __global__ void kernel3d_baseline_memwarp(REAL* __restrict__ input, REAL*__restrict__ output,
//                                   int height, int width_y, int width_x); 
// #define PERKS_DECLARE_INITIONIZATION_BASELINE_MEMWARP(_type,halo,ipt,tilex,tiley) \
//     __global__ void kernel3d_baseline_memwarp<_type,halo,ipt,tilex,tiley>(_type*__restrict__,_type*__restrict__,int,int,int);


template<class REAL, int halo, int ipt, int tilex, int blockdim=256>
__global__ void kernel3d_persistent(REAL* __restrict__ input, REAL*__restrict__ output,
                                  int height, int width_y, int width_x, 
                                  REAL * l2_cache_i, REAL * l2_cache_o, 
                                  int iteration); 
#define PERKS_DECLARE_INITIONIZATION_PERSISTENT(_type,halo,ipt,tilex,blockdim) \
    __global__ void kernel3d_persistent<_type,halo,ipt,tilex,blockdim>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, int);

// #define TEMPSTEP (5)
// #define TEMPSTEP (5)

#include"temporalconfig.cuh"
#include"iptconfig.cuh"

template<class REAL, int halo,  int blockdim=256,int tilex=ipts<halo,curshape,REAL>::tile_x, int ipt=ipts<halo,curshape,REAL>::val, int tstep=timesteps<halo,curshape,ipt,REAL>::val>
__global__ void kernel3d_temporal(REAL * __restrict__ input, 
                                REAL *  output, 
                                int width_z, int width_y, int width_x,
                                // int base_z, int base_y, int base_x,
                                REAL* l2_cache_i, REAL* l2_cache_o
                                ); 

template<class REAL, int halo,  int blockdim=256,int tilex=ipts<halo,curshape,REAL>::tile_x, int ipt=ipts<halo,curshape,REAL>::val, int tstep=timesteps<halo,curshape,ipt,REAL>::val>
__global__ void kernel3d_temporal_traditional(REAL * __restrict__ input, 
                                REAL *  output, 
                                int width_z, int width_y, int width_x//,
                                // int base_z, int base_y, int base_x,
                                // REAL* l2_cache_i, REAL* l2_cache_o
                                ); 
// #define PERKS_DECLARE_INITIONIZATION_TEMPORAL2(_type,halo,ipt,tilex,blockdim,tstep) \
//     __global__ void kernel3d_temporal<_type,halo,ipt,tilex,blockdim,tstep>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, _type**,_type**, int);

// #define PERKS_DECLARE_INITIONIZATION_TEMPORAL(_type,halo,ipt,tilex,blockdim) \
//     __global__ void kernel3d_temporal<_type,halo,ipt,tilex,blockdim>(_type*__restrict__,_type*__restrict__,int,int,int, _type*, _type*, _type**,_type**, int);


