#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src, src_gray, dst;
int thresh = 35;
int max_thresh = 255;

void cornerHarris_callback( int, void* );

int main( int, char** argv ) {
    src = imread( argv[1], IMREAD_COLOR );
    cvtColor( src, src_gray, COLOR_BGR2GRAY );

    namedWindow( "cornerHarris", WINDOW_AUTOSIZE );
    createTrackbar( "Threshold: ", "cornerHarris", &thresh, max_thresh, cornerHarris_callback );
    imshow( "cornerHarris", src );

    cornerHarris_callback(0,0);

    waitKey(0);
    return 0;
}

void cornerHarris_callback( int, void* ) {
    Mat norm;

    cornerHarris( src_gray, dst, 2, 3, 0.008, BORDER_DEFAULT );
    normalize( dst, norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );

    dst = src.clone();

    /// Drawing a circle around corners
    for( int j = 0; j < norm.rows ; j++ )
        for( int i = 0; i < norm.cols; i++ )
            if( (int) norm.at<float>(j,i) > thresh )
                circle( dst, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );

    imshow( "cornerHarris", dst );
}
