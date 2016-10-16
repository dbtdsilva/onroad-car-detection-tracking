#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int thresh = 100;
RNG rng(12345);
Mat src_gray;

/// Function header
void thresh_callback(int, void* );
bool verify_contours(const vector<Point>&);

int main(int argc, char** argv) {

    Mat src;
    /// Load source image and convert it to gray
    src = imread("damas.jpg", 1 );

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
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    Mat drawing_final = Mat::zeros( canny_output.size(), CV_8UC3 );
    int stats = 0;
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        if (verify_contours(contours[i])) {
            stats++;
            drawContours(drawing_final, contours, i, Scalar(0,0,255), 2, 8, hierarchy, 0, Point());
        }
    }
    cout << "Found " << stats << " matches out of " << contours.size() << " contours" << endl;
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    namedWindow( "Contours Final", CV_WINDOW_AUTOSIZE );
    imshow( "Contours Final", drawing_final);
}

bool verify_contours(const vector<Point> &list) {
    if (list.size() == 1) return false;
    // Verifying lines first
    // http://math.stackexchange.com/questions/701862/how-to-find-if-the-points-fall-in-a-straight-line-or-not
    // (x2 − x1) * (y3 − y1) − (y2 − y1) * (x3 − x1) = 0
    bool is_line = true;
    unsigned long&& last = list.size() - 1;
    for (int i = 1; i < list.size(); i++) {
        double equation = (list[i].x - list[0].x) * (list[last].y - list[0].y) -
                (list[i].y - list[0].y) * (list[last].x - list[0].x);
        if (equation != 0)
            is_line = false;
    }
    if (is_line)
        return true;

    bool is_circle = true;
    for (int i = 0; i < list.size(); i++) {
        is_circle = false;
    }
    return false;
}