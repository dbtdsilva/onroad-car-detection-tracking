CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( calib )
FIND_PACKAGE( OpenCV REQUIRED )
#find_package(PCL 1.2 REQUIRED)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

#copy data file to build directory
file(GLOB JPG "${CMAKE_CURRENT_SOURCE_DIR}/images/*.jpg")
file(GLOB BMP "${CMAKE_CURRENT_SOURCE_DIR}/images/*.bmp")
file(COPY ${BMP} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${JPG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

ADD_EXECUTABLE( calibrate calibrate.cpp)
TARGET_LINK_LIBRARIES( calibrate ${OpenCV_LIBS} )
ADD_EXECUTABLE( undistort undistort.cpp)
TARGET_LINK_LIBRARIES( undistort ${OpenCV_LIBS} )
ADD_EXECUTABLE( epipolarLines epipolarLines.cpp)
TARGET_LINK_LIBRARIES( epipolarLines ${OpenCV_LIBS} )
ADD_EXECUTABLE( rectify rectification.cpp)
TARGET_LINK_LIBRARIES( rectify ${OpenCV_LIBS} )