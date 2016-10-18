#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

#define _USE_MATH_DEFINES

int thresh = 100;
RNG rng(12345);
Mat src_gray;

/// Function header
void thresh_callback(int, void* );
bool verify_contours(const vector<Point>& list);
double calculate_slope(const Point& p1, const Point& p2);
long double calculate_distance(const Point& p1, const Point& p2);

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
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    cout << "H: " << hierarchy.size() << ", C: " << contours.size() << endl;
    for (int i = 0; i < hierarchy.size(); i++) {

    }
    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    Mat drawing_final = Mat::zeros( canny_output.size(), CV_8UC3 );
    int stats = 0;
    for( int i = 0; i< contours.size(); i++ ) {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        //if (verify_contours(contours[i])) {
        //    stats++;
        //    drawContours(drawing_final, contours, i, Scalar(0,0,255), 2, 8, hierarchy, 0, Point());
        //}
    }

    /*for (int j = 0; j < contours.size(); j++) {
        bool detecting_line = false;
        double last_slope = 0;
        double first_idx = 0;
        long double distance_travelled = 0;
        if (j != 145) continue;
        for (int i = 1; i < contours[j].size() - 1; i++) {
            try {
                double slope = abs(calculate_slope(contours[j][i-1], contours[j][i]) - calculate_slope(contours[j][i], contours[j][i+1]));
                cout << slope << endl;
                if (slope <= 45) {
                    line(drawing_final, contours[j][i-1], contours[j][i+1], Scalar(0, 0, 255), 2, 8);
                }
                if (slope < 50 && (!detecting_line || last_slope == slope)) {
                    if (!detecting_line) {
                        last_slope = slope;
                        distance_travelled = 0;
                        detecting_line = true;
                        first_idx = i-1;
                        cout << "Got" << endl;
                    }
                    distance_travelled += calculate_distance(contours[j][i-1], contours[j][i-1]);
                } else {
                    if (detecting_line && distance_travelled > 10)
                        line(drawing_final, contours[j][first_idx], contours[j][i+1], Scalar(0, 0, 255));
                    detecting_line = false;
                }
            } catch(Exception&) {
                //line(drawing_final, contours[j][i-1], contours[j][i+1], Scalar(0, 0, 255), 2, 8);
            }
        }
    }*/

    for (int i = 0; i < contours.size(); i++) {
        for (int j = 1; j < contours[i].size() - 1; j++) {
            double slope = abs(calculate_slope(contours[i][j-1], contours[i][j]) -
                               calculate_slope(contours[i][j], contours[i][j+1]));
            long double distance = calculate_distance(contours[i][j-1], contours[i][j+1]);
            if (slope < 25 && distance >= 5) {
                drawing_final.at<Vec3b>(Point(contours[i][j].x, contours[i][j].y)) = Vec3b(0, 0, 255);
                line(drawing_final, contours[i][j-1], contours[i][j], Scalar(0, 255, 0), 1, 8);
                line(drawing_final, contours[i][j], contours[i][j+1], Scalar(0, 255, 0), 1, 8);
            }
        }
    }
    cout << "Found " << stats << " matches out of " << contours.size() << " contours" << endl;
    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
    namedWindow( "Contours Final", CV_WINDOW_AUTOSIZE );
    imshow( "Contours Final", drawing_final);
    namedWindow( "canny_output Final", CV_WINDOW_AUTOSIZE );
    imshow( "canny_output Final", canny_output);
}

long double calculate_distance(const Point& p1, const Point& p2)
{
    double x = p1.x - p2.x; //calculating number to square in next step
    double y = p1.y - p2.y;
    return sqrt(pow(x, 2) + pow(y, 2));
}

bool verify_contours(const vector<Point> &list) {
    if (list.size() == 1) return false;
    // Verifying lines first
    // http://math.stackexchange.com/questions/701862/how-to-find-if-the-points-fall-in-a-straight-line-or-not
    // (x2 − x1) * (y3 − y1) − (y2 − y1) * (x3 − x1) = 0
    bool is_line = true;
    double slope1, slope2;
    unsigned long&& last = list.size() - 1;
    for (int i = 1; i < list.size(); i++) {
        try {
            slope1 = calculate_slope(list[0], list[i]);
            slope2 = calculate_slope(list[i], list[last]);
        } catch (Exception&) {
            continue;
        }

        if (abs(slope1 - slope2) > 1)
            is_line = false;
        else {
            cout << "S1: " << slope1 << ", S2: " << slope2 << endl;
        }
    }
    if (is_line)
        return true;

    bool is_circle = true;
    for (int i = 0; i < list.size(); i++) {
        is_circle = false;
    }
    return false;
}

double calculate_slope(const Point& p1, const Point& p2) {
    int diff_x = p2.x - p1.x;
    int diff_y = p2.y - p1.y;
    if (diff_x == 0) return 0;
    return atan(diff_y / diff_x) * 180 / M_PI;
}