#pragma once


static constexpr int star_shape = 1;      // 1: star, 2: box
static constexpr int box_shape = 2;       // 1: star, 2: box


template<int halo, int shape, class REAL>
struct ipts
{
  static int const val = 4;
};

template<int halo, int shape> 
struct ipts<halo,shape,double>
{
  static int const val = 4;
};


template<int halo, int shape> 
struct ipts<halo,shape,float>
{
  static int const val = 8;
};

