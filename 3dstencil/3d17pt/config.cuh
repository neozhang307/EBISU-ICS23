



static constexpr int const HALO = 1;     // rad or halo
static constexpr int const FPC = 34;     // flops per cell
static constexpr int const isBox = HALO; // 0: star, HALO: box
static constexpr int const curshape = 2; // 1: star, 2: box 3: type0  4: poisson

// todo haven't figure out a better way to pass parameter

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



