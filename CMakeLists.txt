cmake_minimum_required(VERSION 3.14)
include(ExternalProject)
include(FetchContent)

# set project name and languages
project(Fiesta CXX C Fortran)

# project wide settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default cache variables
set (Fiesta_CUDA         OFF CACHE BOOL "Enable Cuda build"       )
set (Fiesta_HIP          OFF CACHE BOOL "Enable HIP build"        )
set (Fiesta_OPENMP       OFF CACHE BOOL "Enable OpenMP build"     )
set (Fiesta_SERIAL       OFF CACHE BOOL "Enable Serial build"     )
set (Fiesta_BUILD_ALL    OFF CACHE BOOL "Enable super-build"      )
set (Fiesta_BUILD_KOKKOS OFF CACHE BOOL "Build kokkos"            )
set (Fiesta_BUILD_HDF5   OFF CACHE BOOL "Build hdf5"              )
set (Fiesta_BUILD_LUA    OFF CACHE BOOL "Build lua"               )
set (Fiesta_BUILD_FMT    OFF CACHE BOOL "Build fmt"               )
set (Fiesta_BUILD_TESTS  OFF CACHE BOOL "Build all tests"         )
set (Fiesta_ENABLE_DEBUG OFF CACHE BOOL "Enable Fiesta Debugging" )
set (Fiesta_NO_MPI       ON  CACHE BOOL "Build Fiesta without MPI")

# get git branch and hash to set version, build type and date variables
string(TIMESTAMP BUILD_TIME "%Y-%m-%dT%H:%MZ" UTC)
set(FIESTA_OPTS "NO_MPI")

# compile flags
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -pedantic -Wall -Wextra -fdiagnostics-color=always")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")

# set parameters for debug builds
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Fiesta_ENABLE_DEBUG ON CACHE BOOL "Enable Fiesta Debugging" FORCE)
    set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "" FORCE)
endif()

#setup device list
if (Fiesta_CUDA)
    set(DEVICE CUDA)
    set(FIESTA_OPTS "${FIESTA_OPTS}+CUDA")
elseif(Fiesta_HIP)
    set(DEVICE HIP)
    set(FIESTA_OPTS "${FIESTA_OPTS}+HIP")
elseif(Fiesta_OPENMP)
    set(DEVICE OPENMP)
    set(FIESTA_OPTS "${FIESTA_OPTS}+OPENMP")
else()
    set(DEVICE SERIAL)
    set(FIESTA_OPTS "${FIESTA_OPTS}+SERIAL")
    set(Fiesta_SERIAL ON CACHE BOOL "Enable Serial build" FORCE)
endif()

if(Fiesta_BUILD_ALL)
    message(STATUS "FIESTA: Super-build enabled.")
    set (Fiesta_BUILD_KOKKOS ON CACHE BOOL "Build kokkos" FORCE)
    set (Fiesta_BUILD_HDF5   ON CACHE BOOL "Build hdf5"   FORCE)
    set (Fiesta_BUILD_LUA    ON CACHE BOOL "Build lua"    FORCE)
    set (Fiesta_BUILD_FMT    ON CACHE BOOL "Build fmt"    FORCE)
endif()

#find third party components
include(tpls/kokkos.cmake)
include(tpls/lua.cmake)
include(tpls/fmt.cmake)
include(tpls/hdf5.cmake)

#import FiestaCore and fiesta targetrs
include_directories(src)
add_subdirectory(src)

# import tests
if (Fiesta_BUILD_TESTS)
    include(tests/tests.cmake)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (Fiesta_ENABLE_DEBUG ON CACHE BOOL "Enable Fiesta Debugging" FORCE)
endif()

# print build strings
message("Fiesta Build Options: ${FIESTA_OPTS}")
message("Fiesta Build Time:    ${BUILD_TIME}" )
