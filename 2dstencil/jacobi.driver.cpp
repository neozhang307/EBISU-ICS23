#include <cassert>
#include <cstdio>
#include "./common/common.hpp"
#include "./common/cub_utils.cuh"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "./common/temporalconfig.cuh"
#include "config.cuh"

const double TOLERANCE = 1e-5;
// this is for debugging the CPU reference
#ifdef REFCHECK
#define CPUCODE
#endif

template <class REAL>
double runTest(int width_x, int width_y, int iteration, int bdimx, int blkpsm, bool check, bool async, bool usewarmup, bool verbose)
{
  double error = 0;
  REAL(*input)
  [width_x] = (REAL(*)[width_x])
      getRandom2DArray<REAL>(width_y, width_x);
  REAL(*output)
  [width_x] = (REAL(*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
  REAL(*output_gold)
  [width_x] = (REAL(*)[width_x])
      getZero2DArray<REAL>(width_y, width_x);
#ifdef REFCHECK
  jacobi_gold((REAL *)input, width_y, width_x, (REAL *)output);
  jacobi_gold_iterative((REAL *)input, width_y, width_x, (REAL *)output_gold, iteration);
#else
  int err = jacobi_iterative((REAL *)input, width_y, width_x, (REAL *)output, bdimx, blkpsm, iteration, async, usewarmup, verbose);
  if (err == 1)
  {
    if (check)
      printf("unsupport setting, no free space for cache with shared memory\n");
    check = 0;
  }
  if (check != 0)
  {
    jacobi_gold_iterative((REAL *)input, width_y, width_x, (REAL *)output_gold, iteration);
  }
#endif
  if (check != 0)
  {
    int halo = HALO * iteration;
    error =
        checkError2D<REAL>(width_x, (REAL *)output, (REAL *)output_gold, halo, width_y - halo, halo, width_x - halo);
    printf("[Test] RMS Error : %e\n", error);
  }
  delete[] input;
  delete[] output;
  delete[] output_gold;
  return error;
}

int main(int argc, char *argv[])
{
  int width_x;
  int width_y;
  int iteration = 3;
  int warmupiteration = -1;
  width_x = width_y = 2048; // 4096;
  bool fp32 = true;         // float
  bool check = false;       // compare result with CPU
  int bdimx = 256;          // block dim deprecated
  int blkpsm = 0;           // block per stream multiprocessor

  bool async = false;       // if use async copy deprecated
  bool usewarmup = false;   // if use warmup
  bool isExpriment = false; // the experiment setting in ICS23 paper
  bool verbose = false;     // verbose output
  if (argc >= 3)
  {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);
    width_x = width_x == 0 ? 2048 : width_x;
    width_y = width_y == 0 ? 2048 : width_y;
  }

  CommandLineArgs args(argc, argv);

  args.GetCmdLineArgument("bdim", bdimx);
  args.GetCmdLineArgument("blkpsm", blkpsm);
  args.GetCmdLineArgument("iter", iteration);

  fp32 = args.CheckCmdLineFlag("fp32");
  check = args.CheckCmdLineFlag("check");
  usewarmup = args.CheckCmdLineFlag("warmup");
  isExpriment = args.CheckCmdLineFlag("experiment");
  verbose = args.CheckCmdLineFlag("verbose");

  if (bdimx == 0)
    bdimx = 256;
  if (iteration == 0)
    iteration = 3;
#ifndef REFCHECK
  if (isExpriment)
  {
    bdimx = 256;
    usewarmup = true;
    if (fp32)
    {
      getExperimentSetting<float>(&iteration, &width_y, &width_x, bdimx);
    }
    else
    {
      getExperimentSetting<double>(&iteration, &width_y, &width_x, bdimx);
    }
  }
#endif
#ifdef REFCHECK
  iteration = 4;
#endif
  double error = 0;
  if (fp32)
  {
    error = runTest<float>(width_x, width_y, iteration, bdimx, blkpsm, check, async, usewarmup, verbose);
  }
  else
  {
    error = runTest<double>(width_x, width_y, iteration, bdimx, blkpsm, check, async, usewarmup, verbose);
  }
  if (error >= TOLERANCE)
  {
    return -1;
  }
  return 0;
}
