




static constexpr int const HALO = 1;     // rad or halo
static constexpr int const FPC = 38;     // flops per cell
static constexpr int const isBox = HALO; // 0: star, HALO: box
static constexpr int const curshape = 4; // 1: star, 2: box 

// todo haven't figure out a better way to pass parameter

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

      