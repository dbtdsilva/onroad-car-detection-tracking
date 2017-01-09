#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat src, original, src_gray;

    /// Read the image
    src = imread("damas.jpg", 1);

    if (!src.data) {
        return -1;
    }

    original = src.clone();
    /// Convert it to gray
    cvtColor(src, src_gray, CV_BGR2GRAY);

    /// Reduce the noise so we avoid false circle detection
    // GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

    vector<Vec3f> circles;
    vector<Vec4i> lines;

    /// Apply the Hough Transform to find the circles
    cout << "Calculating Hough Circles" << endl;
    HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.cols / 8, 220,
                 35, 18, 30);

    cout << "Calculating Hough Lines" << endl;
    Canny( src_gray, src_gray, 50, 200, 3 );
    HoughLinesP(src_gray, lines, 1, CV_PI/ 180, 100, 40, 200 );
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3,
             CV_AA);
    }
    /// Draw the circles detected
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        // circle outline
        circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
    }
    /// Show your results
    imshow("Original", src_gray);
    imshow("Hough Transform Demo", src);

    waitKey(0);
    return 0;
}