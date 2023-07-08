

template <class REAL, int RANGE>
__device__ void __forceinline__ regEnqueue(REAL reg_que[RANGE],
                                           REAL *sm_queue,
                                           int sm_tile_y_ind, int sm_tile_x_ind, int tile_y_with_halo, int tile_x_with_halo)
{
    _Pragma("unroll") for (int l_y = 0; l_y < RANGE; l_y++)
    {
        reg_que[l_y] = sm_queue[(sm_tile_y_ind + l_y) * tile_x_with_halo + sm_tile_x_ind];
    }
}


template <class REAL, int TILE_Y, int TILE_X>
__device__ void __forceinline__ regEnqueue(REAL reg_que[TILE_Y][TILE_X],
                                           REAL *sm_queue,
                                           int sm_tile_y_ind, int sm_tile_x_ind, int tile_y_with_halo, int tile_x_with_halo)
{
    _Pragma("unroll") for (int l_y = 0; l_y < TILE_Y; l_y++)
    {
       _Pragma("unroll") for (int l_x = 0; l_x < TILE_X; l_x++)
        {
            reg_que[l_y][l_x] = sm_queue[(sm_tile_y_ind + l_y) * tile_x_with_halo + sm_tile_x_ind+l_x];
        }
    }
}
/// Load data from gm to register
/// @tparam REAL data type
/// @tparam RANGE range of the shared memory indexs for the multiqueue
/// @param sm_queueindex shared memory multi-queue indexs
/// @param sm_tile_y_ind shared memory tile y index
/// @param sm_tile_x_ind shared memory tile x index
/// @param tile_x_with_halo shared memory tile x size with halo
/// @param tile_y_with_halo shared memory tile y size with halo
/// @param input global memory input
/// @param global_z global memory z index
/// @param global_y global memory y index
/// @param global_x global memory x index
/// @param width_z global memory z size
/// @param width_y global memory y size
/// @param width_x global memory x size
template <class REAL, int MQRANGE, int RANGE>
__device__ void __forceinline__ regEnqueues(REAL reg_mque[MQRANGE][RANGE],
                                            REAL *input,
                                            int global_z, int global_y, int global_x,
                                            int width_z, int width_y, int width_x)
{
    _Pragma("unroll") for (int l_z = 0; l_z < MQRANGE; l_z++)
    {
        _Pragma("unroll") for (int l_y = 0; l_y < RANGE; l_y++)
        {

            int l_global_y = (MIN(l_y + global_y, width_y - 1));
            l_global_y = (MAX(l_global_y, 0));
            int l_global_z = (MIN(global_z, width_z - 1));
            l_global_z = (MAX(l_global_z, 0));
            reg_mque[l_z][l_y] = input[l_global_z * width_x * width_y + (l_global_y)*width_x + global_x]; // smbuffer_buffer_ptr[l_z+sizeofsm-(2*halo+2)][(l_y+ps_y+index_y)*tile_x_with_halo+ps_x+tid_x];//input[(global_y) * width_x + global_x];
        }
    }
}

template <class REAL, int DEPTH, int REG_MQSIZE, int REG_Q_RANGE, int REG_Q_SIZE, int SHUFFLE_DISTANCE, int SHUFFLE_RANGE>
__device__ void __forceinline__ regShuffle(REAL reg_mqueues[REG_MQSIZE][REG_Q_SIZE])
{
    _Pragma("unroll") for (int l_z = 0; l_z < DEPTH; l_z++)
    {
        _Pragma("unroll") for (int l_h = 0; l_h < SHUFFLE_RANGE; l_h++)
        {
            _Pragma("unroll") for (int l_y = 0; l_y < REG_Q_SIZE; l_y++)
            {
                reg_mqueues[l_z * (REG_Q_RANGE) + l_h][l_y] = reg_mqueues[l_z * (REG_Q_RANGE) + SHUFFLE_DISTANCE + l_h][l_y];
            }
        }
    }
}

template <class REAL, int MQSIZE, int REGQUESIZE, int  REG_Q_RANGE, int SHUFFLE_RANGE, int SHUFFLE_DISTANCE, int TILE_Y, int TILE_X>
__device__ void __forceinline__ regShuffle(REAL reg_mqueues[REG_Q_RANGE][TILE_Y][TILE_X])
{
    _Pragma("unroll")
    for(int l_z=0; l_z<MQSIZE ; l_z++)
    {
        _Pragma("unroll")
        for(int l_hz=0; l_hz<SHUFFLE_RANGE; l_hz++)
        {
            _Pragma("unroll")
            for(int l_y=0; l_y<TILE_Y; l_y++)
            {
                _Pragma("unroll")
                for(int l_x=0; l_x<TILE_X ; l_x++)
                {
                    reg_mqueues[l_hz+l_z*(REGQUESIZE)][l_y][l_x]=reg_mqueues[l_hz+SHUFFLE_DISTANCE+l_z*(REGQUESIZE)][l_y][l_x];
                }
        }
        }
    }
}


/// Load data from gm to register
/// @tparam REAL data type
/// @tparam TILERANGE range of the shared memory indexs for the multiqueue
/// @param sm_queueindex shared memory multi-queue indexs
/// @param sm_tile_y_ind shared memory tile y index
/// @param sm_tile_x_ind shared memory tile x index
/// @param tile_x_with_halo shared memory tile x size with halo
/// @param tile_y_with_halo shared memory tile y size with halo
/// @param input global memory input
/// @param global_z global memory z index
/// @param global_y global memory y index
/// @param global_x global memory x index
/// @param width_z global memory z size
/// @param width_y global memory y size
/// @param width_x global memory x size
template <class REAL, int TILERANGE>
__device__ void __forceinline__ store(REAL *output,
                                      int global_z, int global_y, int global_x,
                                      int width_z, int width_y, int width_x,
                                      REAL reg_que[TILERANGE])
{
    if (global_x < width_x)
    {
        _Pragma("unroll") for (int l_y = 0; l_y < TILERANGE; l_y++)
        {
            // if(p_x+tid_x>=width_x)break;
            int l_global_y = global_y + l_y;
            output[global_z * width_x * width_y + (l_global_y)*width_x + global_x] = reg_que[l_y];
        }
    }
}

/// Store data from register to shared memory multiqueue
/// @tparam REAL data type
/// @tparam RANGE range of the shared memory indexs for the multiqueue
/// @param sm_queueindex shared memory multi-queue indexs
/// @param sm_tile_y_ind shared memory tile y index
/// @param sm_tile_x_ind shared memory tile x index
/// @param tile_x_with_halo shared memory tile x size with halo
/// @param tile_y_with_halo shared memory tile y size with halo
/// @param input global memory input
/// @param global_z global memory z index
/// @param global_y global memory y index
/// @param global_x global memory x index
/// @param width_z global memory z size
/// @param width_y global memory y size
/// @param width_x global memory x size
template <class REAL, int RANGE>
__device__ void __forceinline__ smEnqueueAsync(REAL *sm_queueindex[RANGE],//range in qeue
                                               int sm_tile_y_ind, int sm_tile_x_ind, int tile_x_with_halo, int tile_y_with_halo,
                                               REAL *input,
                                               int global_z, int global_y, int global_x,
                                               int width_z, int width_y, int width_x)
{
    _Pragma("unroll") for (int l_z = 0; l_z < RANGE; l_z++)
    {
        for (int lid = threadIdx.x; lid < tile_x_with_halo * tile_y_with_halo; lid += blockDim.x)
        {
            int l_x = lid % tile_x_with_halo - sm_tile_x_ind;
            int l_y = lid / tile_x_with_halo - sm_tile_y_ind;

            // int l_global_z=(MIN(global_z-(halo+1)*(TSTEP-1)+l_z,width_z-1));
            int l_global_z = (MIN(global_z, width_z - 1));
            l_global_z = (MAX(l_global_z, 0));

            int l_global_y = (MIN(global_y + l_y, width_y - 1));
            l_global_y = (MAX(l_global_y, 0));
            int l_global_x = (MIN(global_x + l_x, width_x - 1));
            l_global_x = (MAX(l_global_x, 0));
            __pipeline_memcpy_async(sm_queueindex[l_z] + tile_x_with_halo * (l_y + sm_tile_y_ind) + l_x + sm_tile_x_ind,
                                    input + l_global_z * width_x * width_y + l_global_y * width_x + l_global_x, sizeof(REAL));
        }
    }
    __pipeline_commit();
}

/// Store data from register to shared memory multiqueue
/// @tparam REAL data type
/// @tparam DEPTH depth of the temporal blocking
/// @tparam SM_RANGE range of the shared memory indexs for the multiqueue
/// @tparam REG_RANGE range of the register indexs
/// @tparam QUEUE_SIZE  size of the queue (shared memory multi-queue)
/// @param sm_queueindex shared memory multi-queue indexs
/// @param sm_tile_y_ind shared memory tile y index
/// @param sm_tile_x_ind shared memory tile x index
/// @param tile_x_with_halo shared memory tile x size with halo
/// @param tile_y_with_halo shared memory tile y size with halo
/// @param regQueues register queues
template <class REAL, int MQSIZE, int REGQSIZE, int SM_RANGE, int TILERANGE, int QUEUE_SIZE>
__device__ void __forceinline__ smEnqueues(REAL *sm_queueindex[SM_RANGE],
                                           int sm_tile_y_ind, int sm_tile_x_ind, int tile_x_with_halo, int tile_y_with_halo,
                                           REAL regQueues[MQSIZE*REGQSIZE][TILERANGE])
{
    _Pragma("unroll") for (int step = 1; step < MQSIZE; step++)
    {
        _Pragma("unroll") for (int l_y = 0; l_y < TILERANGE; l_y++)
        {
            sm_queueindex[SM_RANGE - (step) * (QUEUE_SIZE)-1][(sm_tile_y_ind + l_y) * tile_x_with_halo + sm_tile_x_ind] = regQueues[MQSIZE*REGQSIZE - step][l_y];
        }
    }
}

template <class REAL, int TILE_Y>
__device__ void __forceinline__ smEnqueue(REAL *sm_que,
                                           int sm_tile_y_ind, int sm_tile_x_ind, int tile_x_with_halo, int tile_y_with_halo,
                                           REAL reg_que[TILE_Y])
{
   _Pragma("unroll")
    for(int l_y=0; l_y<TILE_Y; l_y++)
    {
        sm_que[(sm_tile_y_ind+l_y)*tile_x_with_halo+sm_tile_x_ind]=reg_que[l_y];
    }
}

/// Restore the index status to its original
/// @tparam REAL data type
/// @tparam SM_RANGE range of the shared memory indexs for the multiqueue
/// @tparam SHUFFLE_DISTANCE shuffle distance of the indexs
/// @param sm_queueindex shared memory multi-queue indexs
template <class REAL, int SM_RANGE, int SHUFFLE_DISTANCE>
__device__ void __forceinline__ smShuffle(REAL *sm_queueindex[SM_RANGE])
{
    REAL *tmp = sm_queueindex[0];
    _Pragma("unroll") for (int hl = 1; hl < SM_RANGE; hl++)
    {
        sm_queueindex[hl - 1] = sm_queueindex[hl];
    }
    sm_queueindex[SM_RANGE - 1] = tmp;
}

template <class REAL, int TILERANGE, int halo, int REGY_SIZE, int REGZ_SIZE = 2 * halo + 1, int REGX_SIZE = 2 * halo + 1, int REG_BASE = halo, int SMZ_SIZE = 2 * halo + 1>
__device__ void __forceinline__ computation(REAL result[TILERANGE],
                                            REAL *smques[SMZ_SIZE],
                                            int sm_y_base, int sm_width, int sm_x_ind,
                                            REAL regques[REGZ_SIZE][TILERANGE],
                                            const REAL west[HALO], const REAL east[HALO], const REAL north[HALO], const REAL south[HALO], const REAL top[HALO], const REAL bottom[HALO], const REAL center)
{
    _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
    {
        _Pragma("unroll") for (int l_y = 0; l_y < TILERANGE; l_y++)
        {
            result[l_y] += bottom[hl] * regques[REG_BASE - 1 - hl][l_y];
        }
    }
    _Pragma("unroll") for (int l_y = 0; l_y < halo; l_y++)
    {
        int sm_y_ind = sm_width * (l_y + sm_y_base);
        _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
        {
            result[l_y] += west[hl] *
                           smques[0][sm_y_ind + sm_x_ind - 1 - hl];
            result[l_y] += east[hl] *
                           smques[0][sm_y_ind + sm_x_ind + 1 + hl];
            result[l_y] += north[hl] *
                           smques[0][sm_width * (1 + hl) + sm_y_ind + sm_x_ind];
            result[l_y] += south[hl] *
                           smques[0][-sm_width * (1 + hl) + sm_y_ind + sm_x_ind];
        }
    }
    _Pragma("unroll") for (int l_y = halo; l_y < TILERANGE - halo; l_y++)
    {
        int sm_y_ind = sm_width * (l_y + sm_y_base);
        _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
        {
            result[l_y] += west[hl] *
                           smques[0][sm_y_ind + sm_x_ind - 1 - hl];
            result[l_y] += east[hl] *
                           smques[0][sm_y_ind + sm_x_ind + 1 + hl];
        }
    }
    _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
    {
        _Pragma("unroll") for (int l_y = halo; l_y < TILERANGE - halo; l_y++)
        {

            result[l_y] +=
                north[hl] * regques[REG_BASE][l_y + 1 + hl];
            result[l_y] +=
                south[hl] * regques[REG_BASE][l_y - 1 - hl];
        }
    }
    _Pragma("unroll") for (int l_y = TILERANGE - halo; l_y < TILERANGE; l_y++)
    {
        int sm_y_ind = sm_width * (l_y + sm_y_base);
        _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
        {
            result[l_y] += west[hl] *
                           smques[0][sm_y_ind + sm_x_ind - 1 - hl];
            result[l_y] += east[hl] *
                           smques[0][sm_y_ind + sm_x_ind + 1 + hl];
            result[l_y] += north[hl] *
                           smques[0][sm_width * (1 + hl) + sm_y_ind + sm_x_ind];
            result[l_y] += south[hl] *
                           smques[0][-sm_width * (1 + hl) + sm_y_ind + sm_x_ind];
        }
    }

    _Pragma("unroll") for (int hl = 0; hl < halo; hl++)
    {
        _Pragma("unroll") for (int l_y = 0; l_y < TILERANGE; l_y++)
        {
            result[l_y] += top[hl] * smques[0 + hl + 1][sm_width * (l_y + sm_y_base) + sm_x_ind];
        }
    }
    _Pragma("unroll") for (int l_y = 0; l_y < TILERANGE; l_y++)
    {
        result[l_y] += center * regques[REG_BASE][l_y];
    }
}
