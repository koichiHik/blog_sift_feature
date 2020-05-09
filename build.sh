#!/bin/bash
ADDITIONAL_CMAKE_MODULE_DIR=./cmake
ROOT_3RD=~/workspace/3rd_party

# Configure Main Repository.
if [ ! -e ./build ]; then
  mkdir build
fi
cd build

cmake \
  -D ADDITIONAL_CMAKE_MODULE_DIR=${ADDITIONAL_CMAKE_MODULE_DIR} \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -D ROOT_3RD=${ROOT_3RD} \
  ../

make

cd ../