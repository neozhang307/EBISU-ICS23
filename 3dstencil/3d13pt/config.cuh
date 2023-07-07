// #define HALO (2)




static constexpr int const HALO = 2;        // rad or halo
static constexpr int const FPC = 26;        // flops per cell
static constexpr int const isBox = 0;       // 0: star, HALO: box
static constexpr int const curshape = 1;    // 1: star, 2: box

// todo haven't figure out a better way to pass parameter

#define stencilParaT \
        const REAL center=-0.996f;\
        const REAL west[2]={0.083f,0.083f};\
        const REAL east[2]={0.083f,0.083f};\
        const REAL north[2]={0.083f,0.083f};\
        const REAL south[2]={0.083f,0.083f};\
        const REAL bottom[2]={0.083f,0.083f};\
        const REAL top[2]={0.083f,0.083f};

