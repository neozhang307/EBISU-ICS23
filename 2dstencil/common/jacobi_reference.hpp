

#ifndef PERKS_REFERENCE_HEADER
#define PERKS_REFERENCE_HEADER
// template<class REAL>
// void jacobi(REAL*, int, int, REAL*);

// single step reference
template <class REAL>
void jacobi_gold(REAL *, int, int, REAL *);
// iterative reference
template <class REAL>
void jacobi_gold_iterative(REAL *, int, int, REAL *, int);

#endif