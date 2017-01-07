#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "DetectorHaarCascade.h"
#include "FilterFalsePositives.h"
#include <iostream>

using namespace std;
using namespace cv;


/*
 * False positives removal:
 * https://pythonspot.com/car-tracking-with-cascades/
 */

bool isNewRoi(const Rect& rect, vector<Rect> &rects){
    for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it) {
        if(abs((*it).x - rect.x) < 40 && abs((*it).y - rect.y) < 40)
            return false;
    }
    return true;
}

int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
    }

    Mat frame;

    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()) {
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int keyboard;
    vector<Rect> cars;
    vector<Rect> rects;
    int frameCount = 0, scale = 2, height, width, minY;
    Mat frame_gray;

    DetectorHaarCascade detector(argv[2]);
    FilterFalsePositives filter;
    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        // Scale down the frame to remove the very small boxes
        height = frame.size().height;
        width = frame.size().width;
        Mat tmp_frame_gray;
        resize(frame, tmp_frame_gray, Size(static_cast<int>(width / 2),
                                           static_cast<int>(height / 2)));
        minY = static_cast<int>(tmp_frame_gray.size().height * 0.3);

        cars = detector.detect(tmp_frame_gray);

        vector<Rect> newRegions;
        for (vector<Rect>::iterator it = cars.begin(); it != cars.end(); ++it) {
            Mat roi_image = tmp_frame_gray(*it);
            if ((*it).y > minY && filter.filter(roi_image, FilterType::MEAN_SQUARE)) {
                newRegions.push_back(Rect((*it).x * scale, (*it).y * scale, (*it).width * scale, (*it).height * scale));
            }
        }

        for (vector<Rect>::iterator it = newRegions.begin(); it != newRegions.end(); ++it)
            if(isNewRoi(*it, rects))
                rects.push_back(*it);

        for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it)
            rectangle(frame, *it, Scalar(0, 255, 0), 2);

        frameCount++;
        if(frameCount >= 5){
            frameCount = 0;
            rects.clear();
        }
        imshow("Camera", frame);
        keyboard = waitKey( 100 );

    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}