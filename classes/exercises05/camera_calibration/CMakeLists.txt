CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( calib )
FIND_PACKAGE( OpenCV REQUIRED )
#find_package(PCL 1.2 REQUIRED)

ADD_EXECUTABLE( calib chessboard.cpp )
TARGET_LINK_LIBRARIES( calib ${OpenCV_LIBS} )
ADD_EXECUTABLE( load_cube load_cube.cpp )
TARGET_LINK_LIBRARIES( load_cube ${OpenCV_LIBS} )

