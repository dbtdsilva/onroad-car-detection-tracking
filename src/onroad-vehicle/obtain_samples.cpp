#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <libnet.h>

using namespace std;
using namespace cv;

int imageCounter = 0;
void detectAndSave(Mat& frame, CascadeClassifier& cascade);

int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
    }

    if (mkdir("./images", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1 && errno != EEXIST) {
        cout << "Failed to create folder" << endl;
        exit(EXIT_FAILURE);
    }

    Mat frame;
    CascadeClassifier car_cascade;
    if(!car_cascade.load(argv[2])) {
        cout << "Failed to load " << argv[2] << endl;
        exit(EXIT_FAILURE);
    }

    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int keyboard;

    vector<Rect> cars;
    Mat frame_gray;
    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        if(!frame.empty())
            detectAndSave(frame, car_cascade);

        keyboard = waitKey(10);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}

void detectAndSave(Mat& frame, CascadeClassifier& cascade)
{
    std::vector<Rect> cars;
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    cascade.detectMultiScale(frame_gray, cars, 1.1, 2, 0, Size(10, 10));
    for (auto& car : cars)
    {
        Mat croppedImage;
        Mat(frame_gray, car).copyTo(croppedImage);
        Mat croppedDest;
        Size size(50, 50);
        resize(croppedImage,croppedDest,size);
        imshow("Cropped window", croppedDest );
        char c = (char)waitKey(10);
        if(c == 's'){
            imageCounter++;
            cout << "Saving image " << format("%d.png", imageCounter) << endl;

            imwrite(format("images/%d.png", imageCounter), croppedDest);
        }

        rectangle(frame, car, Scalar(0, 255, 0), 2);
    }

    imshow("Frame", frame);
}