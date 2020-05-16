
cmake_minimum_required(VERSION 3.8)
project(vlfeat_sample)

# CXX Version.
set_property(GLOBAL
    PROPERTY 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
)

# Additional modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/${ADDITIONAL_CMAKE_MODULE_DIR}")

# Load external libraries.
FIND_PACKAGE(OpenCV)
FIND_PACKAGE(Glog)
FIND_PACKAGE(Gflags)
FIND_PACKAGE(Eigen3)

# Source files.
set(TARGET "sift_test")
set(SRC src/sift_test.cc src/sift_detector.cc src/sift_descriptor.cc  src/sift_from_opencv.cc)

# Generate Exe Files
add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${GFLAGS_LIBRARIES})
target_link_libraries(${TARGET} ${GLOG_LIBRARIES})
target_link_libraries(${TARGET} ${OpenCV_LIBS})

# Macro Definition
target_compile_definitions(${TARGET}
    PRIVATE PRJ_FOLDER_PATH="${CMAKE_SOURCE_DIR}")
