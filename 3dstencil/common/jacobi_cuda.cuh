#ifndef PERKS_CUDA_HEADER
#define PERKS_CUDA_HEADER
//template<class REAL>

//this is where the aimed implementation located
template<class REAL>
int j3d_iterative(REAL*, int, int, int, REAL*, int, int, int, bool, bool );


// template<class REAL>int getMinWidthY(int , int, int, bool isDoubleTile=false);
template<class REAL>void getExperimentSetting(int* , int* , int*, int*, int);

#endif
