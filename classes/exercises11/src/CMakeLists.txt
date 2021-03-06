CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( features )
FIND_PACKAGE( OpenCV REQUIRED )

SET(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

file(GLOB JPG "${CMAKE_CURRENT_SOURCE_DIR}/../images/*.jpg")
file(GLOB BMP "${CMAKE_CURRENT_SOURCE_DIR}/../images/*.bmp")
file(GLOB PNG "${CMAKE_CURRENT_SOURCE_DIR}/../images/*.png")

file(COPY ${BMP} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${JPG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${PNG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

ADD_EXECUTABLE( hough hough.cpp hough_representation.cpp hough_representation.h )
TARGET_LINK_LIBRARIES( hough ${OpenCV_LIBS} -ltbb)

ADD_EXECUTABLE( surfExample surfExample.cpp)
TARGET_LINK_LIBRARIES( surfExample ${OpenCV_LIBS} )

ADD_EXECUTABLE( siftExample siftExample.cpp)
TARGET_LINK_LIBRARIES( siftExample ${OpenCV_LIBS} )
