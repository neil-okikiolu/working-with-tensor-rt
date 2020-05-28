cmake_minimum_required(VERSION 2.8.3)
project(jetsoncam)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_transport
    roscpp
    sensor_msgs
    vision_msgs
    std_msgs
)

find_package(jetson-utils REQUIRED)
find_package(jetson-inference REQUIRED)

find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jetsoncam
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
    LIBRARIES ros_deep_learning_nodelets
    CATKIN_DEPENDS nodelet roscpp image_transport sensor_msgs
)


###########
## Build ##
###########

# enable c++11 (TensorRT requirement)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# for gstreamer i.e. <gst/gst.h>
include_directories(
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/include/libxml2
    /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include
    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/
    /usr/lib/aarch64-linux-gnu/glib-2.0/include/
    /usr/lib/x86_64-linux-gnu/glib-2.0/include/
)


add_executable(imagetaker_detectnet
    src/node_imagetaker_detectnet.cpp
    src/image_converter.cpp
)
target_link_libraries(imagetaker_detectnet ${catkin_LIBRARIES} jetson-inference)


add_executable(imagetaker_segnet
    src/node_imagetaker_segnet.cpp
    src/image_converter.cpp
    src/image_ops.cpp
)
target_link_libraries(imagetaker_segnet ${catkin_LIBRARIES} jetson-inference)


add_executable(imagetaker_imagenet
    src/node_imagetaker_imagenet.cpp
    src/image_converter.cpp
    src/image_ops.cpp
)
target_link_libraries(imagetaker_imagenet ${catkin_LIBRARIES} jetson-inference)


add_executable(imagetaker_superresnet
    src/node_imagetaker_superresnet.cpp
    src/image_converter.cpp
    src/image_ops.cpp
)
target_link_libraries(imagetaker_superresnet
    ${catkin_LIBRARIES}
    jetson-inference
)

# used to compile CUDA files .cu, .cuh
CUDA_COMPILE(posenet_cuda_o  src/poseNet.cu)

# add library files that we would NOT change
# in 'src/parse/*'
file(GLOB parseSrc
    "src/parse/*.cpp"
)



# add the compiled CUDA *.o files here
add_executable(imagetaker_posenet
    src/node_imagetaker_posenet.cpp
    src/image_converter.cpp
    src/image_ops.cpp
    src/poseNet.cpp
    ${posenet_cuda_o}
    src/plugins.cpp
    src/ParseObjects.cpp
    ${parseSrc}
)
target_link_libraries(imagetaker_posenet
    ${catkin_LIBRARIES}
    jetson-inference
    jetson-utils
)

#add_library(trt_pose SHARED
#    trt_pose/parse/find_peaks.cpp
#    trt_pose/parse/refine_peaks.cpp
#    trt_pose/parse/paf_score_graph.cpp
#    trt_pose/parse/munkres.cpp
#    trt_pose/parse/connect_parts.cpp
#)