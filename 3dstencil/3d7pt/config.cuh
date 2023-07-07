


static constexpr int const HALO = 1;        // rad or halo
static constexpr int const FPC = 14;        // flops per cell
static constexpr int const isBox = 0;       // 0: star, HALO: box
static constexpr int const curshape = 1;    // 1: star, 2: box

// todo haven't figure out a better way to pass parameter

#define stencilParaT \
        const REAL center=-1.67f;\
        const REAL west[1]={0.162f};\
        const REAL east[1]={0.161f};\
        const REAL north[1]={0.163f};\
        const REAL south[1]={0.164f};\
        const REAL bottom[1]={0.166f};\
        const REAL top[1]={0.165f};

