#include "../common/common.hpp"
#include "../common/jacobi_reference.hpp"


//#ifndef REAL
//#define REAL float
//#endif


template<class REAL>
static void print_matrix 
(int width_y, int width_x, REAL* l_output)
{
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 1 ; i < width_y-1 ; i++ )
    for( int j = 1 ; j < width_x-1 ; j++ )
      printf ("Output[%d][%d] = %.6f\n", i, j, output[i][j]);
}

template<class REAL>
static void jacobi_step
(const REAL* l_input, int width_y, int width_x, REAL* l_output, int step)
{
  const REAL (*input)[width_x] = (const REAL (*)[width_x])l_input;
  REAL (*output)[width_x] = (REAL (*)[width_x])l_output;
  for( int i = 1 ; i < width_y-1 ; i++ )
    for( int j = 1 ; j < width_x-1 ; j++ )
      output[i][j] = (
        7*input[i-1][j-1] + 5*input[i-1][j] + 9*input[i-1][j+1] +
        12*input[i][j-1] + 15*input[i][j] + 12*input[i][j+1] + 
        9*input[i+1][j-1] + 5*input[i+1][j] + 7*input[i+1][j+1]) 
    / 118;
  //if (step == 1) print_matrix(width_y, width_x, l_output);
}

template<class REAL>
void jacobi_gold
(REAL* l_input, int width_y, int width_x, REAL* l_output)
{
  REAL* temp = getZero2DArray<REAL>(width_y, width_x);
  //REAL start_time = rtclock ();
  jacobi_step(l_input, width_y, width_x, temp, 0);
  //REAL end_time = rtclock ();
  //printf ("Time taken by reference : %lf secs\n", end_time - start_time);
  //print_matrix (width_y, width_x, temp);
  jacobi_step(temp, width_y, width_x, l_output, 1);
  memset(temp, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(l_output, width_y, width_x, temp, 2);
  memset(l_output, 0, sizeof(REAL) * width_y * width_x);
  jacobi_step(temp, width_y, width_x, l_output, 3);
}

template void jacobi_gold<float>(float *, int, int, float *);
template void jacobi_gold<double>(double *, int, int, double *);