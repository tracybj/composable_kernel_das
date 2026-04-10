#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

if [ -e /opt/dtk/bin/aicc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/bin/aicc
elif [ -e /opt/dtk/llvm/bin/hipcc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/hipcc
else
	CK_CMAKE_PREFIX_PATH=/opt/rocm
	CK_CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
fi

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=${CK_CMAKE_PREFIX_PATH}                                                      \
-D CMAKE_CXX_COMPILER=${CK_CMAKE_CXX_COMPILER}                                                    \
-D CMAKE_CXX_FLAGS="-O3"                                                                          \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=OFF                                                                                  \
-D GPU_TARGETS="gfx908;gfx90a;gfx940"                                                             \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=ON                                                                 \
-D USE_BITINT_EXTENSION_INT4=OFF                                                                  \
${MY_PROJECT_SOURCE}

