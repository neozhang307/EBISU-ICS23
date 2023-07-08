# EBISU-ICS23
This is a repo to keep the experimental implementation of EBISU used in ICS23

The detailed explanations are in of stencils are in the 2dstencil and 3dstencil README.md. 


Explanation of how to run stencil solver is in:

stencil/2dstencil/README.md

stencil/3dstencil/README.md

You can compile code with config.sh and then build.sh in the ./conjugateGradient, ./stencil/2dstencil and ./stencil/3dstencil folders.
The executable will be in ./build/init folders. 

## Acknowledge
stencil used some code from the following repository:
- Unit test & data generation: https://github.com/pssrawat/IEEE2017
- Shared memory optimizations: https://github.com/naoyam/benchmarks

## Cite EBISU:
Lingqi Zhang, Mohamed Wahib, Peng Chen, Jintao Meng, Xiao Wang, Toshio Endo, and Satoshi Matsuoka. 2023. Revisiting Temporal Blocking Stencil Optimizations. In Proceedings of the 37th International Conference on Supercomputing (ICS '23). Association for Computing Machinery, New York, NY, USA, 251–263. https://doi.org/10.1145/3577193.3593716
