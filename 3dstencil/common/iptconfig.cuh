#pragma once
#ifndef star_shape
  #define star_shape (1)
#endif
#ifndef box_shape
  #define box_shape (2)
#endif
#ifndef type0_shape
  #define type0_shape (3)
#endif
#ifndef poisson_shape
  #define poisson_shape (4)
#endif


template<int halo, int shape, class REAL>
struct ipts
{
  static int const val = 4;
  static int const tile_x = 32;
};


template<int halo, int shape> 
struct ipts<halo,shape,double>
{
  static int const val = 4;
  static int const tile_x = 32;
};

template<int halo, int shape> 
struct ipts<halo,shape,float>
{
  static int const val = 8;
  static int const tile_x = 64;
};

// template<> 
// struct ipts<1,star_shape,double>
// {
//   static int const val = 2;
//   static int const tile_x = 64;
// };
