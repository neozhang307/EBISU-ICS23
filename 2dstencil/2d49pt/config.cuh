static constexpr int const HALO = 3;     // rad or halo
static constexpr int const FPC = 98;     // flops per cell
static constexpr int const isBox = HALO; // 0: star, HALO: box
static constexpr int const curshape = 2; // 1: star, 2: box

// todo haven't figure out a better way to pass parameter
#define stencilParaT                                                 \
    const REAL filter[7][7] = {                                      \
        {1.0 / 118,1.0 / 118, 2.0 / 118, 3.0 / 118, 4.0 / 118, 5.0 / 118, 5.0 / 118},     \
        {1.0 / 118,1.0 / 118, 2.0 / 118, 3.0 / 118, 4.0 / 118, 5.0 / 118, 5.0 / 118},     \
        {1.0 / 118,7.0 / 118, 7.0 / 118, 5.0 / 118, 7.0 / 118, 6.0 / 118, 5.0 / 118},     \
        {1.0 / 118,8.0 / 118, 12.0 / 118, 15.0 / 118, 12.0 / 118, 12.0 / 118, 5.0 / 118}, \
        {1.0 / 118,9.0 / 118, 9.0 / 118, 5.0 / 118, 7.0 / 118, 15.0 / 118, 5.0 / 118},    \
        {1.0 / 118,9.0 / 118, 9.0 / 118, 5.0 / 118, 7.0 / 118, 15.0 / 118, 5.0 / 118},    \
        {1.0 / 118,10.0 / 118, 11.0 / 118, 12.0 / 118, 13.0 / 118, 14.0 / 118, 5.0 / 118}};
        