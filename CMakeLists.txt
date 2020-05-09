
cmake_minimum_required(VERSION 3.5)
project(vlfeat_sample)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/${ADDITIONAL_CMAKE_MODULE_DIR}")

set(OpenCV_DIR "${ROOT_3RD}/opencv/install/share")
message("OpenCV_DIR" ${OpenCV_DIR})
FIND_PACKAGE(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH 3.4.7 EXACT)
FIND_PACKAGE(Glog)
FIND_PACKAGE(Gflags)
FIND_PACKAGE(Eigen3)

set(TARGET "sift_test")
set(SRC src/sift_test.cc src/sift_copy.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})
