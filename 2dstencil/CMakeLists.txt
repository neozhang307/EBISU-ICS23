cmake_minimum_required(VERSION 3.10)

project(EBISU_2DSTENCIL VERSION 1.0)

cmake_policy(SET CMP0077 NEW)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#star
add_subdirectory(2d5pt)
add_subdirectory(2d13pt)
#add_subdirectory(2ds9pt)

#box
add_subdirectory(2d9pt)
#add_subdirectory(2d25pt)
add_subdirectory(2d49pt)

