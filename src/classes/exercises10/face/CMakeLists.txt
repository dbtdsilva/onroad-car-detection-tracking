CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( face_detection )
FIND_PACKAGE( OpenCV REQUIRED )

SET(dir ${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/build")

file(GLOB XML "${CMAKE_CURRENT_SOURCE_DIR}/*.xml")
file(GLOB PNG "${CMAKE_CURRENT_SOURCE_DIR}/*.png")
file(GLOB CSV "${CMAKE_CURRENT_SOURCE_DIR}/*.csv")

file(COPY ${XML} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${PNG} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(COPY ${CSV} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

ADD_EXECUTABLE( face_detection face_crop_from_video.cpp)
TARGET_LINK_LIBRARIES( face_detection ${OpenCV_LIBS} )

ADD_EXECUTABLE( face_crop_from_image face_crop_from_image.cpp )
TARGET_LINK_LIBRARIES( face_crop_from_image ${OpenCV_LIBS} )

ADD_EXECUTABLE( face_recognizer face_recognizer.cpp)
TARGET_LINK_LIBRARIES( face_recognizer ${OpenCV_LIBS} )

ADD_EXECUTABLE( face_crop_from_video face_crop_from_video.cpp)
TARGET_LINK_LIBRARIES( face_crop_from_video ${OpenCV_LIBS} )
