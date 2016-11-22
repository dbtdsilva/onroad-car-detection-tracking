CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( calib )
FIND_PACKAGE( OpenCV REQUIRED )
#find_package(PCL 1.2 REQUIRED)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

ADD_EXECUTABLE( lukaskanade_tracking lukaskanade_tracking.cpp )
TARGET_LINK_LIBRARIES( lukaskanade_tracking ${OpenCV_LIBS} )

ADD_EXECUTABLE( foregroundDetection foregroundDetection.cpp )
TARGET_LINK_LIBRARIES( foregroundDetection ${OpenCV_LIBS} )

ADD_EXECUTABLE( tracker tracker.cpp )
TARGET_LINK_LIBRARIES( tracker ${OpenCV_LIBS} )


ADD_EXECUTABLE( farneback farneback.cpp )
TARGET_LINK_LIBRARIES( farneback ${OpenCV_LIBS} )
