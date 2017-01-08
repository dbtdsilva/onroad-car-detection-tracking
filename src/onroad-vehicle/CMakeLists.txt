CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(onroad-vehicle)
FIND_PACKAGE(OpenCV REQUIRED)

INCLUDE(FindPkgConfig)
INCLUDE(ExternalProject)

SET(opentld_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/opentld)
EXTERNALPROJECT_ADD(
        opentld
        PREFIX ${opentld_PREFIX}
        URL "https://github.com/gnebehay/OpenTLD/archive/master.zip"
        URL_MD5 "3b829793a570a1cfd91dfce2c753c65a"
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${opentld_PREFIX} -DCMAKE_BUILD_TYPE=${opentld_PREFIX}
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        BUILD_IN_SOURCE 1
        LOG_UPDATE ON
        LOG_BUILD ON
        LOG_INSTALL ON
)

include_directories(${opentld_PREFIX}/src/opentld/src/libopentld/)
set(OPENTLD_LIBRARIES
        ${opentld_PREFIX}/src/opentld/lib/libopentld.a
        ${opentld_PREFIX}/src/opentld/lib/libcvblobs.a)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)

ADD_EXECUTABLE(obtain_samples obtain_samples.cpp)
TARGET_LINK_LIBRARIES(obtain_samples ${OpenCV_LIBS})

ADD_EXECUTABLE(raw_cascade raw_cascade.cpp)
TARGET_LINK_LIBRARIES(raw_cascade ${OpenCV_LIBS})

ADD_EXECUTABLE(detection detection.cpp DetectorMatchingFeatures.cpp)
TARGET_LINK_LIBRARIES(detection ${OpenCV_LIBS})

ADD_EXECUTABLE(main_new_tracker main_new_tracker.cpp
        DetectorHaarCascade.cpp DetectorHaarCascade.h
        FilterFalsePositives.cpp FilterFalsePositives.h TrackerOpenTLD.cpp TrackerOpenTLD.h MultiTrackerOpenTLD.cpp MultiTrackerOpenTLD.h Helper.cpp Helper.h)
TARGET_LINK_LIBRARIES(main_new_tracker ${OpenCV_LIBS} ${OPENTLD_LIBRARIES})

ADD_EXECUTABLE(main main.cpp DetectorHaarCascade.cpp DetectorHaarCascade.h FilterFalsePositives.cpp FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(main ${OpenCV_LIBS})

ADD_EXECUTABLE(tracker tracker.cpp)
TARGET_LINK_LIBRARIES(tracker ${OpenCV_LIBS})

ADD_EXECUTABLE(camshift camshift.cpp)
TARGET_LINK_LIBRARIES(camshift ${OpenCV_LIBS})
