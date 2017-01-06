CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(onroad-vehicle)
FIND_PACKAGE(OpenCV REQUIRED)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

ADD_EXECUTABLE(obtain_samples obtain_samples.cpp)
TARGET_LINK_LIBRARIES(obtain_samples ${OpenCV_LIBS})

ADD_EXECUTABLE(raw_cascade raw_cascade.cpp)
TARGET_LINK_LIBRARIES(raw_cascade ${OpenCV_LIBS})

ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})

ADD_EXECUTABLE(tracker tracker.cpp)
TARGET_LINK_LIBRARIES(tracker ${OpenCV_LIBS})
