
static constexpr int const HALO = 1;     // rad or halo
static constexpr int const FPC = 18;     // flops per cell
static constexpr int const isBox = HALO; // 0: star, HALO: box
static constexpr int const curshape = 2; // 1: star, 2: box

// todo haven't figure out a better way to pass parameter
#define stencilParaT                          \
    const REAL filter[3][3] = {               \
        {7.0 / 118, 5.0 / 118, 9.0 / 118},    \
        {12.0 / 118, 15.0 / 118, 12.0 / 118}, \
        {9.0 / 118, 5.0 / 118, 7.0 / 118}};
