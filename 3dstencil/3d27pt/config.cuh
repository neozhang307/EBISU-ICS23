

static constexpr int const HALO = 1;     // rad or halo
static constexpr int const FPC = 54;     // flops per cell
static constexpr int const isBox = HALO; // 0: star, HALO: box
static constexpr int const curshape = 2; // 1: star, 2: box

// todo haven't figure out a better way to pass parameter

#define stencilParaT                          \
  const REAL filter[3][3][3] = {              \
      {{0.5 / 159, 0.7 / 159, 0.90 / 159},    \
       {1.2 / 159, 1.5 / 159, 1.2 / 159},     \
       {0.9 / 159, 0.7 / 159, 0.50 / 159}},   \
      {{0.51 / 159, 0.71 / 159, 0.91 / 159},  \
       {1.21 / 159, 1.51 / 159, 1.21 / 159},  \
       {0.91 / 159, 0.71 / 159, 0.51 / 159}}, \
      {{0.52 / 159, 0.72 / 159, 0.920 / 159}, \
       {1.22 / 159, 1.52 / 159, 1.22 / 159},  \
       {0.92 / 159, 0.72 / 159, 0.520 / 159}}};


