#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int thresh = 100;
Mat src, src_gray, dst;

/// Function header
void thresh_callback(int, void* );

int main(int argc, char** argv) {
    /// Load source image and convert it to gray
    src = imread("../damas.jpg", 1 );

    /// Convert image to gray and blur it
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) );

    /// Create Window
    namedWindow("Source", CV_WINDOW_AUTOSIZE );
    imshow("Source", src );

    createTrackbar("Canny thresh:", "Source", &thresh, 255, thresh_callback );
    thresh_callback( 0, 0 );

    waitKey(0);
    return 0;
}

void thresh_callback(int, void* )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
        drawContours(drawing, contours, i, Scalar(255,255,255), 2, 8, hierarchy, 0, Point() );

    dst = src.clone();
    for( int i = 0; i< contours.size(); i++ ) {
        approxPolyDP(Mat(contours[i]), contours[i], 5, true);
        if (contours[i].size() == 4) {
            Rect r = boundingRect(Mat(contours[i]));
            rectangle( dst, r.tl(), r.br(), Scalar(0, 0, 255), 2, 8, 0 );
        }
        if(contours[i].size()>5 && contours[i].size()<12){
            Point2f center;
            float radius;
            minEnclosingCircle( (Mat)contours[i], center, radius );
            circle( dst, center, (int)radius, Scalar(0, 255, 0), 2, 8, 0 );
        }
    }

    /// Show in a window
    imshow("Source", dst );
}
