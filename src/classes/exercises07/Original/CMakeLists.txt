CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( view )
FIND_PACKAGE( OpenCV REQUIRED )
FIND_PACKAGE( PCL 1.2 REQUIRED )
FIND_PACKAGE(VTK REQUIRED)

ADD_EXECUTABLE( reconstruct reconstruct.cpp )
ADD_EXECUTABLE( viewCloud viewcloud.cpp )
ADD_EXECUTABLE( viewOpenni pcl_io.cpp )
#ADD_EXECUTABLE( viewFreenect viewFreenect.cpp SimpleFreenect.cpp)

include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

TARGET_LINK_LIBRARIES( reconstruct ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES( viewCloud ${PCL_LIBRARIES} ${VTK_LIBRARIES})
TARGET_LINK_LIBRARIES( viewOpenni ${OpenCV_LIBS} ${PCL_LIBRARIES} )
#TARGET_LINK_LIBRARIES( viewFreenect ${OpenCV_LIBS} ${PCL_LIBRARIES} freenect)

