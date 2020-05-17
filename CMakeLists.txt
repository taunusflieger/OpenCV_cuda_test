cmake_minimum_required(VERSION 3.1)
project( OpenCVTest )
find_package( OpenCV REQUIRED )

message(STATUS "OpenCV environment used")
message(STATUS "opencv include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv libs       : ${OpenCV_LIBS}")
message(STATUS "opencv compute cap: ${OpenCV_COMPUTE_CAPABILITIES}")
message(STATUS "opencv install path: ${OpenCV_INSTALL_PATH}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( resizegpu resize_gpu.cpp )
target_link_libraries( resizegpu ${OpenCV_LIBS} )

