#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat frame;

    int lastState;
    while (true) {
        cap >> frame;

        char received = (char)waitKey(10);
        if (received == '+')
            lastState = (lastState + 1) % 4;
        else if (received == '-')
            lastState = (lastState - 1) % 4;
        else if (received == 'q' || received == 'Q')
            break;

        switch (lastState) {
            case 1:
                cvtColor(frame, frame, CV_BGR2GRAY);
                break;
            case 2:
                cvtColor(frame, frame, CV_BGR2GRAY);
                adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);
                break;
            case 3:
                cvtColor(frame, frame, CV_BGR2GRAY);
                adaptiveThreshold(frame, frame, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
        }
        namedWindow("Camera", CV_WINDOW_AUTOSIZE);
        imshow("Camera", frame);
    }
    return 0;
}