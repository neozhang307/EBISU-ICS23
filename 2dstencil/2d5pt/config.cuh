// #define HALO (1)
static constexpr int const &HALO = 1; // rad or halo
static constexpr int const &FPC = 10; // flops per cell
static constexpr int isBox = 0;       // 0: star, 1: box
#define FPC (10)

#ifndef Halo
#define Halo HALO
#endif
