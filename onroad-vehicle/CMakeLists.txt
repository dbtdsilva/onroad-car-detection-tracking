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

ADD_SUBDIRECTORY(src)
