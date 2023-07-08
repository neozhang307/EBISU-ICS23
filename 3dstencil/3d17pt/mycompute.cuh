
template<class REAL, int RESULT_SIZE, int halo, int REGY_SIZE,  int REGZ_SIZE=2*halo+1, int SMZ_SIZE=halo+1+halo>
__device__ void __forceinline__ computation_box(REAL sum[RESULT_SIZE],
                                            REAL* smbuffer_buffer_ptr[SMZ_SIZE], int SMZ_BASE,
                                            int sm_y_base, int sm_width, int sm_x_ind,
                                            REAL r_smbuffer[REGZ_SIZE][REGY_SIZE][2*halo+1],int REGZ_BASE, 
                                            const REAL filter[halo*2+1][halo*2+1][halo*2+1])
{

 
    // bottem register
    {
      int hl_z=-1;

          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][-1+halo]*
              r_smbuffer[hl_z+REGZ_BASE][1+halo+l_y][-1+halo];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][-1+halo]*
              r_smbuffer[hl_z+REGZ_BASE][-1+halo+l_y][-1+halo];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][1+halo]*
              r_smbuffer[hl_z+REGZ_BASE][-1+halo+l_y][1+halo];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][1+halo]*
              r_smbuffer[hl_z+REGZ_BASE][1+halo+l_y][1+halo];
          }

    }
// middle register
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
// upper register
    {
      int hl_z=1;

          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][-1+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+1+sm_y_base)*sm_width-1+sm_x_ind];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][-1+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y-1+sm_y_base)*sm_width-1+sm_x_ind];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][-1+halo][1+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y-1+sm_y_base)*sm_width+1+sm_x_ind];
          }
          _Pragma("unroll")
          for(int l_y=0; l_y<RESULT_SIZE; l_y++)
          {
            sum[l_y]+=filter[hl_z+halo][1+halo][1+halo]*
              smbuffer_buffer_ptr[hl_z+SMZ_BASE][(l_y+1+sm_y_base)*sm_width+1+sm_x_ind];            
          }

    }

}

