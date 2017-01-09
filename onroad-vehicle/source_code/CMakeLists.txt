ADD_EXECUTABLE(obtain_samples obtain_samples.cpp)
TARGET_LINK_LIBRARIES(obtain_samples ${OpenCV_LIBS})

ADD_EXECUTABLE(raw_cascade raw_cascade.cpp)
TARGET_LINK_LIBRARIES(raw_cascade ${OpenCV_LIBS})

ADD_EXECUTABLE(detection_road detection_road.cpp 
        detectors/DetectorMatchingFeatures.cpp detectors/DetectorMatchingFeatures.h)
TARGET_LINK_LIBRARIES(detection_road ${OpenCV_LIBS})

ADD_EXECUTABLE(tracker_alternative_tld tracker_alternative_tld.cpp
        detectors/DetectorHaarCascade.cpp detectors/DetectorHaarCascade.h
        trackers/TrackerOpenTLD.cpp trackers/TrackerOpenTLD.h 
        trackers/MultiTrackerOpenTLD.cpp trackers/MultiTrackerOpenTLD.h 
        trackers/MultiTrackerOpenCV.cpp trackers/MultiTrackerOpenCV.h
        extra/Helper.cpp extra/Helper.h
        extra/FilterFalsePositives.cpp extra/FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(tracker_alternative_tld ${OpenCV_LIBS} ${OPENTLD_LIBRARIES})

ADD_EXECUTABLE(tracker_alternative tracker_alternative.cpp
        detectors/DetectorHaarCascade.cpp detectors/DetectorHaarCascade.h
        trackers/TrackerOpenTLD.cpp trackers/TrackerOpenTLD.h 
        trackers/MultiTrackerOpenTLD.cpp trackers/MultiTrackerOpenTLD.h 
        trackers/MultiTrackerOpenCV.cpp trackers/MultiTrackerOpenCV.h
        extra/Helper.cpp extra/Helper.h
        extra/FilterFalsePositives.cpp extra/FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(tracker_alternative ${OpenCV_LIBS} ${OPENTLD_LIBRARIES})

ADD_EXECUTABLE(detection_haar detection_haar.cpp 
        detectors/DetectorHaarCascade.cpp detectors/DetectorHaarCascade.h 
        extra/FilterFalsePositives.cpp extra/FilterFalsePositives.h)
TARGET_LINK_LIBRARIES(detection_haar ${OpenCV_LIBS})

ADD_EXECUTABLE(tracker tracker.cpp)
TARGET_LINK_LIBRARIES(tracker ${OpenCV_LIBS})

ADD_EXECUTABLE(camshift camshift.cpp)
TARGET_LINK_LIBRARIES(camshift ${OpenCV_LIBS})
