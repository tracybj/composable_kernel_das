SRC_DIR=$(realpath $(dirname $0))
BUILD_DIR=$SRC_DIR/build
BUILD_TYPE=${BUILD_TYPE:=Release}
ARTIFACTS_DIR=${ARTIFACTS_DIR:=$SRC_DIR/../my_rocm}
CPACK_INSTALL_PREFIX=${CPACK_INSTALL_PREFIX:=/opt/rocm-5.7.0}
BUILD_CPUS=${BUILD_CPUS:=$(nproc)}
PACK_TYPE=${PACK_TYPE^^}

# FIXME: a workaround for adding 512/768 vgpr support option,
# since cmake will generate wrong option with target_compile_options()/add_compile_options()
# remove it after figure out how to do it in CMakeLists.txt

# resolve rocm-6.3.3 not set HIP_LIB_PATH issue with hipcc
export HIP_LIB_PATH=$ARTIFACTS_DIR/lib
export HIPCC_COMPILE_FLAGS_APPEND="-mllvm -stream-unfolded-args-in-metadata"
export LD_LIBRARY_PATH=$ARTIFACTS_DIR/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH

[ -d $BUILD_DIR ] || mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake $SRC_DIR \
      -DCMAKE_PREFIX_PATH=${ARTIFACTS_DIR} \
      -DCMAKE_INSTALL_PREFIX=${ARTIFACTS_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DCPACK_PACKAGING_INSTALL_PREFIX=${CPACK_INSTALL_PREFIX} \
      -DCMAKE_CXX_COMPILER=${ARTIFACTS_DIR}/bin/hipcc \
      -DBUILD_EXAMPLE=OFF \
      -DBUILD_TEST=OFF \
      -DCPACK_GENERATOR=${PACK_TYPE} \
      -DUSE_BITINT_EXTENSION=ON \
      || exit 1

make -j 16 || exit 1
make install || exit 1

if [ -n "$CPACK_INSTALL_PREFIX" ] && [ -n "$PACK_TYPE" ]; then
    make package -j 6 || exit 1
fi
