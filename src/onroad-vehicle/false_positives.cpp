#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
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
    Mat frame_gray, frame_hsv, final;
    Mat canny;
    Mat canny_dst;
    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        if (frame.empty())
            continue;

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        cvtColor(frame, frame_hsv, CV_BGR2HSV);

        GaussianBlur(frame_gray, frame_gray, Size(3, 3), 0, 0);
        Canny(frame_gray, canny, 0, 100, 3);

        Mat cannyInv;
        threshold(canny, cannyInv, 128, 255, THRESH_BINARY_INV);

        vector<Vec4i> lines2;
        std::vector<Vec2f> lines;

        imshow("asd", canny);
        //HoughLinesP(canny, lines2, 1, (M_PI / 180.0), 80, 100, 20);
        //HoughLines(canny, lines,1, M_PI/180, 200, 100, 10);
        /*std::vector<Vec2f>::const_iterator it= lines.begin();
        while (it!=lines.end()) {

            float rho= (*it)[0];   // first element is distance rho
            float theta= (*it)[1]; // second element is angle theta

            if ( (theta > 0.09 && theta < 1.48) || (theta < 3.14 && theta > 1.66) ) { // filter to remove vertical and horizontal lines

                // point of intersection of the line with first row
                Point pt1(rho / cos(theta),0);
                // point of intersection of the line with last row
                Point pt2((rho - frame_gray.rows*sin(theta))/cos(theta), frame_gray.rows);
                // draw a line: Color = Scalar(R, G, B), thickness
                line(frame_gray, pt1, pt2, Scalar(255, 0, 255), 1);

            }

            //std::cout << "line: (" << rho << "," << theta << ")\n";
            ++it;
        }*/

        //HoughLines(canny, lines, 1, 1.28571428571429 * (M_PI / 180.0), 80);

        /*for (size_t i = 0; i < lines2.size(); i++) {
            Vec4i l = lines2[i];
            line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 160), 3, CV_AA);
        }
        imshow("Canny", frame_gray);*/
        imshow("HSV", frame_hsv);
        Scalar scalar = sum(frame_hsv);
        cout << frame_hsv.size() << endl;
        int averageHue = frame_hsv.at<Vec3b>(300, 260)[0];//scalar[0] / (frame_hsv.cols * frame_hsv.rows);
        int averageSat = frame_hsv.at<Vec3b>(300, 260)[1];//scalar[1] / (frame_hsv.cols * frame_hsv.rows);
        int averageVal = frame_hsv.at<Vec3b>(300, 260)[2];//scalar[2] / (frame_hsv.cols * frame_hsv.rows);
        inRange(frame_hsv, cv::Scalar(averageHue - 200, averageSat - 15, averageVal - 20),
                cv::Scalar(averageHue + 200, averageSat + 15, averageVal + 20), final);
        //inRange(frame_hsv, cv::Scalar(scalar[0] - 180, scalar[1] - 15, scalar[2] - 20),
        //        cv::Scalar(scalar[0] + 180, scalar[1] + 15, scalar[2] + 20), final);

        imshow("Final", final);


        //int averageHue = sumHue / (rectangle_hsv_channels[0].rows*rectangle_hsv_channels[0].cols);
        //int averageSat = sumSat / (rectangle_hsv_channels[1].rows*rectangle_hsv_channels[1].cols);
        //int averageVal = sumVal / (rectangle_hsv_channels[2].rows*rectangle_hsv_channels[2].cols);

        //equalizeHist(frame_gray, frame_gray);

        car_cascade.detectMultiScale(frame_gray, cars, 1.2, 2, 0);

        Mat frame_new = Mat::zeros(frame.rows, frame.cols, final.type());
        for (const auto& car : cars) {

            rectangle(frame_new, car, Scalar(255), CV_FILLED);
        }

        Mat fram;
        cout << final.size() << frame_new.size() << endl;
        bitwise_and(final, frame_new, fram);

        for (const auto& car : cars) {
            Mat crop = fram(car);
            if (sum(crop).val[0] >= (crop.rows * crop.cols * 255 * 0.2) &&
                    sum(crop).val[0] < (crop.rows * crop.cols * 255 * 0.5)) {
                rectangle(frame, car, Scalar(0, 255, 0), 2);
            }
        }

        imshow("Camera new", frame_new);
        imshow("Fram", fram);
        imshow("Camera", frame);

        keyboard = waitKey( 30 );
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}