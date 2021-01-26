#!/bin/bash
source env.fusiont5.sh

cmake -DTHRUST_INCLUDE_DIR=/home/tqd/Code/thrust/thrust \
    -DNETCDF_CXX_INCLUDE_DIR=/home/tqd/Code/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/netcdf-cxx4-4.3.0-rokguzynggbzgzwzkx27v477ldmiz4ux/include \
    -DNETCDF_CXX_LIBRARY=/home/tqd/Code/spack/opt/spack/linux-ubuntu18.04-x86_64/gcc-7.4.0/netcdf-cxx4-4.3.0-rokguzynggbzgzwzkx27v477ldmiz4ux/lib/libnetcdf_c++4.so \
    -DBoost_DIR=/home/tqd/Code/boost/boostBuild \
    -DBoost_INCLUDE_DIR=/home/tqd/Code/boost/boostBuild/include \
    -DLIBCONFIGPP_LIBRARY=$LIBCONFIGLIB \
    -DMPI_C_LIBRARIES=/cm/shared/apps/mpich/ge/gcc/64/3.2.1/lib \
    ..
