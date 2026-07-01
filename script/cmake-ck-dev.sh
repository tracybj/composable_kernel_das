#!/bin/bash
rm -f CMakeCache.txt
rm -f *.cmake
rm -rf CMakeFiles

MY_PROJECT_SOURCE=$1

CK_EXTRA_CMAKE_ARGS=""

if [ -e /opt/dtk/bin/aicc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/bin/aicc
	# aicc 使用 aillvm clang18；但 DTK 的 hip-config.cmake 默认从 /opt/dtk/llvm
	# 搜索 HIP_CLANG_INCLUDE_PATH，可能误选 hipcc 的 clang17 resource dir，
	# 进而链接 clang17 的 libclang_rt.builtins，导致 host _Float16 转换错误。
	# 因此 aicc 分支显式使用 aicc 自己的 resource dir；hipcc 分支保持 HIP 默认策略。
	CK_AICC_RESOURCE_DIR=$(${CK_CMAKE_CXX_COMPILER} -print-resource-dir 2>/dev/null || true)
	if [ -n "${CK_AICC_RESOURCE_DIR}" ] && [ -d "${CK_AICC_RESOURCE_DIR}/include" ]; then
		CK_EXTRA_CMAKE_ARGS="${CK_EXTRA_CMAKE_ARGS} -D HIP_CLANG_INCLUDE_PATH=${CK_AICC_RESOURCE_DIR}/include"
	fi
elif [ -e /opt/dtk/llvm/bin/hipcc ]; then
	CK_CMAKE_PREFIX_PATH=/opt/dtk
	CK_CMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/hipcc
else
	CK_CMAKE_PREFIX_PATH=/opt/rocm
	CK_CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc
fi

CK_SAVE_TEMPS=${CK_SAVE_TEMPS:-OFF}
CK_VERBOSE_MAKEFILE=${CK_VERBOSE_MAKEFILE:-ON}
CK_EXTRA_CXX_FLAGS=""
if [ "${CK_SAVE_TEMPS}" = "ON" ]; then
	CK_EXTRA_CXX_FLAGS="${CK_EXTRA_CXX_FLAGS} -save-temps=$PWD"
fi

cmake                                                                                             \
-D CMAKE_PREFIX_PATH=${CK_CMAKE_PREFIX_PATH}                                                      \
-D CMAKE_CXX_COMPILER=${CK_CMAKE_CXX_COMPILER}                                                    \
-D CMAKE_CXX_FLAGS="-std=c++17 -O3 -ftemplate-backtrace-limit=0  -fPIE  -Wno-gnu-line-marker      \
${CK_EXTRA_CXX_FLAGS}"                                                                            \
-D CMAKE_BUILD_TYPE=Release                                                                       \
-D BUILD_DEV=ON                                                                                   \
-D GPU_TARGETS="gfx938"                                                                           \
-D BUILD_EXAMPLE=ON                                                                               \
-D USE_BITINT_EXTENSION=ON                                                                        \
-D CMAKE_VERBOSE_MAKEFILE:BOOL=${CK_VERBOSE_MAKEFILE}                                             \
${CK_EXTRA_CMAKE_ARGS}                                                                            \
${MY_PROJECT_SOURCE}
