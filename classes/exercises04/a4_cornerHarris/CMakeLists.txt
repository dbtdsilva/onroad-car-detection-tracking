cmake_minimum_required(VERSION 3.5.1)
project( main )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

#copy data file to build directory
file(GLOB JPG "${CMAKE_CURRENT_SOURCE_DIR}/../../standard_test_images/*.jpg")
file(GLOB BMP "${CMAKE_CURRENT_SOURCE_DIR}/../../standard_test_images/*.bmp")
file(GLOB TIF "${CMAKE_CURRENT_SOURCE_DIR}/../../standard_test_images/*.tif")
file(GLOB TIFF "${CMAKE_CURRENT_SOURCE_DIR}/../../standard_test_images/*.tiff")
file(GLOB PNG "${CMAKE_CURRENT_SOURCE_DIR}/../../standard_test_images/*.png")
file(COPY ${PNG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${TIF} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${TIFF} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${JPG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${BMP} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable( main main.cpp )
target_link_libraries( main ${OpenCV_LIBS} )

