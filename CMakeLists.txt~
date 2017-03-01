cmake_minimum_required(VERSION 2.8.3)
project(hotpot_slam)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "-std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  dynamic_reconfigure
  std_msgs
  image_transport
  dynamic_reconfigure
  sensor_msgs
  tf
  visualization_msgs
  cv_bridge
  pcl_conversions
  pcl_ros
  visualization_msgs
)

#find_package(OpenCV REQUIRED)
set(OpenCV_DIR "/home/rokid/projects/opencv3/build/")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(PCL 1.7 REQUIRED )

##Needed for ros packages
catkin_package(CATKIN_DEPENDS roscpp rospy dynamic_reconfigure std_msgs image_transport dynamic_reconfigure sensor_msgs tf visualization_msgs cv_bridge pcl_conversions pcl_ros visualization_msgs)


include_directories(include
${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/EXTERNAL/csparse/
)

link_directories(${PCL_LIBRARY_DIRS} 
#${OpenCV_LIBS} 
${EIGEN3_LIBS} 
#${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so 
#${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/ 
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/)

add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########

###########my output
add_executable(hotpot_slam_node src/main.cc)
target_link_libraries(hotpot_slam_node ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so )

# add_executable(capture_img src/capture_img.cc)
# target_link_libraries(capture_img ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so )

# add_executable(orb_match_node src/orb_match.cc)
# target_link_libraries(orb_match_node ${OpenCV_LIBS} ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so )
###########my output end



















