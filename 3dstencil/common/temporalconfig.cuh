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

#include "iptconfig.cuh"


template<int halo, int shape, int ipt, class REAL>
struct timesteps
{
  static int const val = 1;
};


template<int halo, int shape, int ipt>
struct timesteps<halo,shape,ipt,double>
{
  static int const val = 1;
};

template<>
struct timesteps< 1,  star_shape,  4,  double>
{
  static int const val = 8;
};

template<>
struct timesteps< 1,  star_shape,  8,  float>
{
  static int const val = 8;
};



template<>
struct timesteps< 2,  star_shape,  4,  double>
{
  static int const val = 5;
};

template<>
struct timesteps< 2,  star_shape,  8,  float>
{
  static int const val = 5;
};

template<>
struct timesteps<1, box_shape,  4,  double>
{
  static int const val = 5;
  // static int const val = 4;
  // static int const val = 1;
  // static int const val = 2;
};

template<>
struct timesteps< 1,  box_shape,  8,  float>
{
  static int const val = 6;
};

template<>
struct timesteps<1, type0_shape,  4,  double>
{
  static int const val = 6;
  // static int const val = 2;
};

template<>
struct timesteps< 1,  type0_shape,  8,  float>
{
  static int const val = 6;
};

template<>
struct timesteps<1, poisson_shape,  4,  double>
{
  static int const val = 6;
};

template<>
struct timesteps< 1,  poisson_shape,  8,  float>
{
  static int const val = 6;
};



