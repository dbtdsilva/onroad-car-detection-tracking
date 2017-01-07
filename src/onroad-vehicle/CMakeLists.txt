CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(onroad-vehicle)
FIND_PACKAGE(OpenCV REQUIRED)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

ADD_EXECUTABLE(obtain_samples obtain_samples.cpp)
TARGET_LINK_LIBRARIES(obtain_samples ${OpenCV_LIBS})

ADD_EXECUTABLE(raw_cascade raw_cascade.cpp)
TARGET_LINK_LIBRARIES(raw_cascade ${OpenCV_LIBS})

ADD_EXECUTABLE(false_positives false_positives.cpp)
TARGET_LINK_LIBRARIES(false_positives ${OpenCV_LIBS})

ADD_EXECUTABLE(detection detection.cpp DetectorMatchingFeatures.cpp)
TARGET_LINK_LIBRARIES(detection ${OpenCV_LIBS})

ADD_EXECUTABLE(main_new_tracker main_new_tracker.cpp
        DetectorHaarCascade.cpp DetectorHaarCascade.h
        FilterFalsePositives.cpp FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(main_new_tracker ${OpenCV_LIBS})

ADD_EXECUTABLE(main main.cpp DetectorHaarCascade.cpp DetectorHaarCascade.h FilterFalsePositives.cpp FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})

ADD_EXECUTABLE(tracker tracker.cpp)
TARGET_LINK_LIBRARIES(tracker ${OpenCV_LIBS})
