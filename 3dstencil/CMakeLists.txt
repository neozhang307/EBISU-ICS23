cmake_minimum_required(VERSION 3.10)
project(EBISU_3DSTENCIL VERSION 1.0)
cmake_policy(SET CMP0077 NEW)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)


#start
add_subdirectory(3d7pt)
add_subdirectory(3d13pt)

#box
add_subdirectory(3d17pt)
add_subdirectory(3d27pt)
add_subdirectory(poisson)
