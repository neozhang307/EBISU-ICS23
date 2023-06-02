#pragma once

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
  static int const val =12;
};


template<>
struct timesteps< 1,  star_shape,  8,  float>
{
  static int const val = 9;
};

template<>
struct timesteps< 2,  star_shape,  4,  double>
{
  static int const val = 8;
};


template<>
struct timesteps< 2,  star_shape,  8,  float>
{
  static int const val = 8;
};


template<>
struct timesteps< 1,  box_shape,  4,  double>
{
  static int const val = 6;
  // static int const val = 2;
};

template<>
struct timesteps< 1,  box_shape,  8,  float>
{
  static int const val = 8;//7;
  // static int const val = 1;//7;
};

template<>
struct timesteps< 2,  box_shape,  4,  double>
{
  static int const val = 4;
};

template<>
struct timesteps< 2,  box_shape,  8,  float>
{
  static int const val = 6;//7;
};

