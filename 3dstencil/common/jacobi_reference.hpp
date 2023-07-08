

#ifndef PERKS_REFERENCE_HEADER
#define PERKS_REFERENCE_HEADER
//template<class REAL>
//void jacobi(REAL*, int, int, REAL*);

// single step reference
template<class REAL>
void j3d_gold(REAL*, int, int, int, REAL*);
// iterative reference
template<class REAL>
void j3d_gold_iterative(REAL*, int, int, int, REAL*, int );


