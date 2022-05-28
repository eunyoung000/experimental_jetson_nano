cmake_minimum_required(VERSION 3.0.0)
project(qrcode VERSION 0.1.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# OpenCV library

# configure the necessary common CMake environment variables
# needed to include and link the OpenCV program into this
# demo project, namely OpenCV_INCLUDE_DIRS and OpenCV_LIBS
set(OpenCV_INCLUDE_DIRS /usr/include/opencv4)
set(OpenCV_LIBRARIES /usr/lib/opencv4)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
find_package(glog 0.4.0 REQUIRED)

set(Zbar_INCLUDE_DIRS /usr/include/zbar)
set(Zbar_LIBRARIES /usr/lib/aarch64-linux-gnu/libzbar.a)

include_directories(${OpenCV_INCLUDE_DIRS} ${Zbar_INCLUDE_DIRS} ${gflags_INCLUDE_DIR} ${glog_INCLUDE_DIRS})

# Focuser class
set(FOCUSER_SOURCE focuser.cpp)
set(FOCUSER_HEADERS focuser.h)
set(FOCUSER_LIB "focuser_lib")

add_library(${FOCUSER_LIB} ${FOCUSER_SOURCE} ${FOCUSER_HEADERS})

# Box fitting class
set(BOX_FITTING_SOURCE box_fitting.cpp)
set(BOX_FITTING_HEADERS box_fitting.h)
set(BOX_FITTING_LIB "box_fitting_lib")

add_library(${BOX_FITTING_LIB} ${BOX_FITTING_SOURCE} ${BOX_FITTING_HEADERS})

# QRcode recognizer class
set(QRCODE_RECOGNIZER_SOURCE qrcode_recognizer.cpp multiframe_based_qrcode_recognizer.cpp)
set(QRCODE_RECOGNIZER_HEADER qrcode_recognizer.h multiframe_based_qrcode_recognizer.h)
set(QRCODE_RECOGNIZER_LIB "qrcode_recognizer_lib")

add_library(${QRCODE_RECOGNIZER_LIB} ${QRCODE_RECOGNIZER_SOURCE} ${QRCODE_RECOGNIZER_HEADER})

set(QRCODE_SOURCE main.cpp)

add_executable(qrcode ${QRCODE_SOURCE})
target_link_libraries(qrcode ${BOX_FITTING_LIB} ${FOCUSER_LIB}  ${QRCODE_RECOGNIZER_LIB} ${OpenCV_LIBRARIES} ${Zbar_LIBRARIES} gflags glog::glog)

set(CAPTURE_SOURCE
   ${FOCUSER_SOURCE}
   video_capture_main.cpp
)

add_executable(video_capture ${CAPTURE_SOURCE} ${HEADERS})

set(IMAGE_CAPTURE_SOURCE
   ${FOCUSER_SOURCE}
   image_capture_main.cpp
)

add_executable(image_capture ${IMAGE_CAPTURE_SOURCE} ${HEADERS})



target_link_libraries(video_capture ${OpenCV_LIBRARIES})
target_link_libraries(image_capture ${OpenCV_LIBRARIES})
