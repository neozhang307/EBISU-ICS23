#include "./common/common.hpp"
#include "./common/jacobi_reference.hpp"
#include "./common/jacobi_cuda.cuh"
#include "./config.cuh"
#include <cassert>
#include <cstdio>

#define TOLERANCE 1e-5
#include "./common/cub_utils.cuh"

template <class REAL>
double runTest(int height, int width_y, int width_x,
               int global_bdimx,
               int blkpsm,
               int iteration,
               bool usewarmup,
               bool check,
               bool verbose)
{
  REAL(*input)
  [width_y][width_x] = (REAL(*)[width_y][width_x])
      getRandom3DArray<REAL>(height, width_y, width_x);
  REAL(*output)
  [width_y][width_x] = (REAL(*)[width_y][width_x])
      getZero3DArray<REAL>(height, width_y, width_x);
  REAL(*output_gold)
  [width_y][width_x] = (REAL(*)[width_y][width_x])
      getZero3DArray<REAL>(height, width_y, width_x);

  iteration = iteration == 0 ? 3 : iteration;
#ifdef REFCHECK
  iteration = 4;
  // j3d_iterative((REAL*)input, height, width_y, width_x, (REAL*)output,iteration);
  j3d_gold((REAL *)input, height, width_y, width_x, (REAL *)output);
  j3d_gold_iterative((REAL *)input, height, width_y, width_x, (REAL *)output_gold, iteration);
#else
  int err = j3d_iterative((REAL *)input, height, width_y, width_x, (REAL *)output, global_bdimx, blkpsm, iteration, usewarmup, verbose);
  if (err == -1)
  {
    if (check)
      printf("the kernel is inlaunchable\n");
    check = 0;
  }
  if (err == -2)
  {
    if (check)
      printf("the height is too smal\n");
    check = 0;
  }
  if (check != 0)
  {
    j3d_gold_iterative((REAL *)input, height, width_y, width_x, (REAL *)output_gold, iteration);
  }

#endif
  if (check != 0)
  {
    int domain_hallo = HALO * iteration;
    double error =
        checkError3D<REAL>(width_y, width_x, (REAL *)output, (REAL *)output_gold, domain_hallo, height - domain_hallo, domain_hallo,
                           width_y - domain_hallo, domain_hallo, width_x - domain_hallo);
    printf("[Test] RMS Error : %e\n", error);
    return error;
  }
  delete[] input;
  delete[] output;
  delete[] output_gold;
}

int main(int argc, char **argv)
{
  // int sm_count;
  // cudaDeviceGetAttribute ( &sm_count, cudaDevAttrMultiProcessorCount,0 );
  int height, width_y, width_x;
  height = width_y = width_x = 256;

  bool fp32 = true; // float
  bool check = false;
  int bdimx = 256; //////////////might be a issue?
  int blkpsm = 0;

  bool usewarmup = false;
  bool usesmall = false;
  bool isExpriment = false;
  bool verbose = false;

  int iteration = 3;

  CommandLineArgs args(argc, argv);
  fp32 = args.CheckCmdLineFlag("fp32");
  check = args.CheckCmdLineFlag("check");
  usewarmup = args.CheckCmdLineFlag("warmup");
  usesmall = args.CheckCmdLineFlag("small");
  isExpriment = args.CheckCmdLineFlag("experiment");
  verbose = args.CheckCmdLineFlag("verbose");

  args.GetCmdLineArgument("bdim", bdimx); /////////////////////might be a issue
  args.GetCmdLineArgument("iter", iteration);
  args.GetCmdLineArgument("blkpsm", blkpsm);

  if (bdimx == 0)
    bdimx = 256; ////////////////////////might be a issue
  if (iteration == 0)
    iteration = 3;
#ifndef REFCHECK
  if (isExpriment)
  {
    bdimx = 256;
    usewarmup = true;
    if (fp32)
    {
      getExperimentSetting<float>(&iteration, &height, &width_y, &width_x, bdimx);
    }
    else
    {
      getExperimentSetting<double>(&iteration, &height, &width_y, &width_x, bdimx);
    }
  }
#endif
  if (argc >= 3)
  {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);

    height = height <= 0 ? 256 : height;
    width_x = width_x <= 0 ? 256 : width_x;
    width_y = width_y <= 0 ? 256 : width_y;
  }

  if (fp32)
  {

    double error = runTest<float>(height, width_y, width_x, bdimx, blkpsm, iteration, usewarmup, check, verbose);

    if (check != 0)
    {

      if (error > TOLERANCE)
        return -1;
    }
  }
  else
  {
    double error = runTest<double>(height, width_y, width_x, bdimx, blkpsm, iteration, usewarmup, check, verbose);
    if (check != 0)
    {

      if (error > TOLERANCE)
        return -1;
    }
  }
}
