#!/bin/bash

set -eou pipefail

apply_patch1() {
  patch -p1 -b <<'EOF'
diff --git a/CMakeLists.txt b/CMakeLists.txt
index e8e4d18..c2c312a 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -793,8 +793,8 @@ if (UNIX OR APPLE)
           -DCPU_ONLY=${CAFFE_CPU_ONLY}
           -DCMAKE_BUILD_TYPE=Release
           -DBUILD_docs=OFF
-          -DBUILD_python=OFF
-          -DBUILD_python_layer=OFF
+         -DBUILD_python=ON
+         -DBUILD_python_layer=ON
           -DUSE_LEVELDB=OFF
           -DUSE_LMDB=OFF
           -DUSE_OPENCV=OFF)
@@ -812,8 +812,8 @@ if (UNIX OR APPLE)
           -DCPU_ONLY=${CAFFE_CPU_ONLY}
           -DCMAKE_BUILD_TYPE=Release
           -DBUILD_docs=OFF
-          -DBUILD_python=OFF
-          -DBUILD_python_layer=OFF
+         -DBUILD_python=ON
+         -DBUILD_python_layer=ON
           -DUSE_LEVELDB=OFF
           -DUSE_LMDB=OFF
           -DUSE_OPENCV=OFF)
EOF
}

apply_patch2() {
  patch -p1 -b <<'EOF'
diff --git a/CMakeLists.txt b/CMakeLists.txt
index f48e942..e8e4d18 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -787,7 +787,7 @@ if (UNIX OR APPLE)
           -DMKLDNN_INSTALL_DIR:PATH=<INSTALL_DIR>
           -DUSE_MKL2017_AS_DEFAULT_ENGINE=${CAFFE_CPU_ONLY}
           -DUSE_CUDNN=${USE_CUDNN}
-          -DCUDA_ARCH_NAME=${CUDA_ARCH}
+          -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME}
           -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN}
           -DCUDA_ARCH_PTX=${CUDA_ARCH_PTX}
           -DCPU_ONLY=${CAFFE_CPU_ONLY}
@@ -806,7 +806,7 @@ if (UNIX OR APPLE)
           CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
           -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
           -DUSE_CUDNN=${USE_CUDNN}
-          -DCUDA_ARCH_NAME=${CUDA_ARCH}
+          -DCUDA_ARCH_NAME=${CUDA_ARCH_NAME}
           -DCUDA_ARCH_BIN=${CUDA_ARCH_BIN}
           -DCUDA_ARCH_PTX=${CUDA_ARCH_PTX}
           -DCPU_ONLY=${CAFFE_CPU_ONLY}
diff --git a/cmake/Cuda.cmake b/cmake/Cuda.cmake
index c6315dc..6fe4c57 100644
--- a/cmake/Cuda.cmake
+++ b/cmake/Cuda.cmake
@@ -307,6 +307,11 @@ op_select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
 list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
 message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")

+if(NOT CUDA_ARCH_NAME)
+  # Create flag for Caffe, which is just the name without the parentheses
+  string(REGEX MATCH "([a-zA-Z]*)" CUDA_ARCH_NAME ${CUDA_ARCH})
+endif()
+
 # Boost 1.55 workaround, see https://svn.boost.org/trac/boost/ticket/9392 or
 # https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
 if (Boost_VERSION EQUAL 105500)
EOF
}

copy_models () {
  local model

  model="/data/models/pose_iter_116000.caffemodel"
  if [[ -f "${model}" ]]; then
    cp "${model}" models/face
  fi

  model="/data/models/pose_iter_102000.caffemodel"
  if [[ -f "${model}" ]]; then
    cp "${model}" models/hand
  fi

  model="/data/models/pose_iter_584000.caffemodel"
  if [[ -f "${model}" ]]; then
    cp "${model}" models/pose/body_25
  fi
}

main() {
  local mode="${1:-cpu}"

  apt update
  apt install python3.11-dev libprotobuf-dev libgoogle-glog-dev protobuf-compiler

  git config --global user.email "user@example.org"
  git config --global user.name "User"

  git clone --depth 1 https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

  cd openpose

  git submodule update --recursive --remote --init
  git add 3rdparty/pybind11
  git commit -m "Update 3rdparty"

  apply_patch1
  apply_patch2

  copy_models

  mkdir build
  cd build

  case "${mode}" in
    cpu)
      cmake .. -DGPU_MODE:STRINGS=CPU_ONLY -DUSE_CUDNN=OFF -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE="/usr/bin/python3.11" -DPYTHON_LIBRARY='/usr/lib/x86_64-linux-gnu/libpython3.11.so'
      ;;
    gpu)
      cmake .. -DUSE_CUDNN=OFF -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE="/usr/bin/python3.11" -DPYTHON_LIBRARY="/usr/lib/x86_64-linux-gnu/libpython3.11.so"
      ;;
    *)
      printf "Unknown mode '%s' (must be 'cpu' or 'gpu')\n" "${mode}" >&2
      exit 1
      ;;
  esac

  make -j`nproc`

  ln -s pyopenpose.cpython-311-x86_64-linux-gnu.so python/openpose/pyopenpose.so
}

main "$@"
