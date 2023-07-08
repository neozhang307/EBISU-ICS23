
template<class REAL, int RESULT_SIZE, int halo, int REGY_SIZE,  int REGZ_SIZE=2*halo+1, int SMZ_SIZE=halo+1+halo>
__device__ void __forceinline__ computation_box(REAL sum[RESULT_SIZE],
                                            REAL* smbuffer_buffer_ptr[SMZ_SIZE], int SMZ_BASE,
                                            int sm_y_base, int sm_width, int sm_x_ind,
                                            REAL r_smbuffer[REGZ_SIZE][REGY_SIZE][2*halo+1],int REGZ_BASE, 
                                            const REAL filter[halo*2+1][halo*2+1][halo*2+1])
{

    //botten & middle from register
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
    //upper from shared memory
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
}

