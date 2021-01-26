cmake_minimum_required(VERSION 3.1)
project(solps_interp)

set(CUDA 1)
set(APPLE 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake/)
set(CMAKE_CXX_STANDARD 17) # Doesn't set NVCC C++ standard
set(CXX_STANDARD_REQUIRED TRUE)

find_package(NetCDF COMPONENTS CXX REQUIRED)

include_directories(${NETCDF_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

    
# setup for OpenMP backend
if (${OPENMP})
    if(APPLE AND OPENMP)
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(OpenMP_C "${CMAKE_C_COMPILER}")
            set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument -I/opt/local/include/libomp")
            set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
            set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
          set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument -I/opt/local/include/libomp")
          set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
          set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
          set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
          set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
          set(OpenMP_omp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif()
    endif()
    
    find_package(Thrust REQUIRED)
    include_directories(${THRUST_INCLUDE_DIRS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++1z -O3")

    add_executable(app src/solps_interp.cpp)
    add_definitions("-x c++ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP")
endif ()

if (${CUDA})
     Message("Enabling CUDA backend")
     find_package(CUDA REQUIRED)
     set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
     set(CUDA_SEPARABLE_COMPILATION TRUE)
     set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -DCUDA -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA --std=c++14 -O3 --expt-relaxed-constexpr --expt-extended-lambda) #-O3 --expt-extended-lambda --expt-relaxed-constexpr -g -G -    -cudart shared
     set_source_files_properties(src/solps_interp.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
     cuda_add_executable(app src/solps_interp.cpp)
endif ()

target_link_libraries(app ${NETCDF_CXX_LIBRARIES})
target_link_libraries(app ${NETCDF_C_LIBRARIES})
