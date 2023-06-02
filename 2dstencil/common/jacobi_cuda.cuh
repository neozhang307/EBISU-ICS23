#ifndef PERKS_CUDA_HEADER
#define PERKS_CUDA_HEADER

template <class REAL>
int jacobi_iterative(REAL *, int, int, REAL *, int, int, int, bool, bool, bool);

template <class REAL>
void getExperimentSetting(int *, int *, int *, int);

#endif
