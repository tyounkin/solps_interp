#!/bin/bash
source env.fusiont5.sh

cmake -DNETCDF_CXX_INCLUDE_DIR=/home/cloud/myRepos/netcdfcxxbuild/include \
    -DNETCDF_CXX_LIBRARY=/home/cloud/myRepos/netcdfcxxbuild/lib/libnetcdf_c++4.so \
    -DLIBCONFIGPP_LIBRARY=$LIBCONFIGLIB \
    ..
