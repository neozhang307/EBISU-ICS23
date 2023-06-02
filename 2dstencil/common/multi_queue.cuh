

template<class REAL, int QUEUE_SIZE, int RANGE, int reg_base=0>
__device__ void __forceinline__ regEnqueue(REAL* sm_queue, REAL reg_queue[QUEUE_SIZE],
                                      int y_base, int y_range,
                                      int x_base, int x_id,
                                      int sm_width)
{
  _Pragma("unroll")
  for(int l_y=0; l_y<RANGE ; l_y++)
  {
    reg_queue[l_y+reg_base] = sm_queue[((l_y+y_base)&(y_range-1))*sm_width+x_base+x_id];
  }
}

template<class REAL, int DEPTH, int DQUEUE_SIZE, int SQUEUE_SIZE, int DQUEUE_BASE, int SQUEUE_BASE, int RANGE>
__device__ void __forceinline__ regEnqueues(REAL d_queue[DEPTH][DQUEUE_SIZE],REAL s_queue[DEPTH][SQUEUE_SIZE]
                                      )
{
  _Pragma("unroll")
  for(int step=1; step < DEPTH; step++)
  {
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RANGE; l_y++)
      {
        d_queue[step-1][l_y+DQUEUE_BASE]=s_queue[step][l_y+SQUEUE_BASE];
      }
    }
  }
}

template<class REAL, int DEPTH, int QUEUE_SIZE,  int SHUFFLE_DISTANCE, int SHUFFLE_RANGE, int LAST_SHUFFLE_RANGE>
__device__ void __forceinline__ regShuffle(REAL reg_mqueues[DEPTH][QUEUE_SIZE])
{
   _Pragma("unroll")
  for(int step=0; step<DEPTH-1; step++)
  {
      _Pragma("unroll")
      for(int l_y=0; l_y<SHUFFLE_RANGE; l_y++)
      {
        reg_mqueues[step][l_y+0]=reg_mqueues[step][l_y+SHUFFLE_DISTANCE];
      }
  }
  {
      for(int l_y=0; l_y<LAST_SHUFFLE_RANGE; l_y++)
      {
        reg_mqueues[DEPTH-1][l_y+0]=reg_mqueues[DEPTH-1][l_y+SHUFFLE_DISTANCE];
      }
  }
}

template<class REAL, int DEPTH, int SHUFFLE_DISTANCE>
__device__ void __forceinline__ smShuffle(int sm_mqueue_index[DEPTH],int sm_tile_range_y)
{
  _Pragma("unroll") for (int i = 0; i < DEPTH; i++)
  {
    sm_mqueue_index[i] = (sm_mqueue_index[i] + SHUFFLE_DISTANCE) & (sm_tile_range_y - 1);
  }
}

template<class REAL, int RANGE>
__device__ void __forceinline__ smEnqueueAsync (REAL* sm_queue, 
                                                int sm_tile_y, int sm_tile_x, int sm_tile_range_y, int sm_tile_width_x,
                                                REAL* input,
                                                int global_y, int global_x, 
                                                int width_y, int width_x)
{
  _Pragma("unroll") 
  for (int l_y = 0; l_y < RANGE; l_y++)
  {
    int l_global_y;
    l_global_y = (MAX(global_y + l_y, 0));
    l_global_y = (MIN(l_global_y, width_y - 1));
    __pipeline_memcpy_async(sm_queue + ((l_y + sm_tile_y) & (sm_tile_range_y - 1)) * sm_tile_width_x + sm_tile_x,
                            input + (l_global_y)*width_x + global_x, sizeof(REAL));
  }
  __pipeline_commit();
}
template<class REAL, int DEPTH, int QUEUE_SIZE, int RANGE>
__device__ void __forceinline__ smEnqueues(REAL* sm_queue, int sm_mqueue_index[DEPTH], 
                                          int sm_tile_y, int sm_tile_x, int sm_tile_range_y, int sm_tile_width_x,
                                          REAL regQueues[DEPTH][QUEUE_SIZE]
                                          )
{
  _Pragma("unroll") for (int step = 1; step < DEPTH; step++)
  {
    _Pragma("unroll") for (int l_y = 0; l_y < RANGE; l_y++)
    {
      sm_queue[((l_y + sm_mqueue_index[step]+sm_tile_y) & (sm_tile_range_y - 1)) * sm_tile_width_x + sm_tile_x] = regQueues[step][l_y];
    }
  }
}

template<class REAL, int QUEUE_SIZE>
__device__ void __forceinline__ store (REAL* aim, REAL reg_queue[QUEUE_SIZE], 
                                      int width_x, int width_y, 
                                      int tile_y_beg, int tile_y_end, 
                                      int index_x, int index_y,
                                      int range
                                      )
{
  _Pragma("unroll") 
  for (int l_y = 0; l_y < range; l_y++)
  {
    int l_global_y = index_y + l_y;
    int l_global_x = index_x;
    {
      if (l_global_y >= tile_y_end)
        break;
      if (l_global_y < tile_y_beg)
        continue;
      if (l_global_x >= width_x)
        break;
    }
    aim[(l_global_y)*width_x + l_global_x] = reg_queue[l_y];
  }
}


template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE*2+2*halo)>
__device__ void __forceinline__ compute(REAL result[RESULT_SIZE], 
                                            REAL* sm_ptr, int sm_y_base, int sm_y_range,
                                            int sm_x_base,int sm_x_ind,int sm_width, int sm_x_range,
                                            // REAL R_PTR,
                                            REAL r_ptr[INPUTREG_SIZE],
                                            int reg_base, 
                                            const REAL west[2],const REAL east[2], 
                                            const REAL north[2],const REAL south[2],
                                            const REAL center 
                                          )
{
  {
    int indexy[RESULT_SIZE];
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      indexy[l_y]=((sm_y_base+l_y)&(sm_y_range-1))*sm_width;
    }
    _Pragma("unroll")
    for(int hl=0; hl<halo; hl++)
    {
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {
        result[l_y]+=sm_ptr[indexy[l_y]+((sm_x_ind+1+hl)&(sm_x_range-1))+sm_x_base]*east[hl];
      }
    }
    _Pragma("unroll")
    for(int hl=0; hl<halo; hl++){
      _Pragma("unroll")
      for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
      {

        result[l_y]+=sm_ptr[indexy[l_y]+((sm_x_ind-1-(halo-1-hl))&(sm_x_range-1))+sm_x_base]*west[halo-1-hl];
      }
    }
  }
  //south
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y-1-hl]*south[hl];
    }
  }
  //center
  _Pragma("unroll")
  for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
  {
    result[l_y]+=r_ptr[reg_base+l_y]*center;
  }
  //north
  _Pragma("unroll")
  for(int hl=0; hl<halo; hl++)
  {
    _Pragma("unroll")
    for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
    {
      result[l_y]+=r_ptr[reg_base+l_y+1+hl]*north[hl];
    }
  }
}
