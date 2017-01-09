
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html

int main(int argc, char **argv) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat src, dst;
    Mat b_hist, g_hist, r_hist, gray_hist;
    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    // Draw the histograms for B, G and R
    int hist_w = 640; int hist_h = 480;
    int bin_w = cvRound( (double) hist_w/histSize );

    while (true) {
        cap >> src;

        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
        Mat histImage2( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
        Mat histImage3( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
        Mat histImage4( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        split( src, bgr_planes );
        cvtColor(src, dst, CV_BGR2GRAY);

        calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
        calcHist( &dst, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );
        
        /// Draw for each channel
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
                  Scalar( 255, 255, 255), 2, 8, 0  );
            line( histImage2, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                  Scalar( 255, 0, 0), 2, 8, 0  );
            line( histImage3, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                  Scalar( 0, 255, 0), 2, 8, 0  );
            line( histImage4, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                  Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                  Scalar( 0, 0, 255), 2, 8, 0  );
        }

        /// Display
        hconcat(histImage, histImage2, histImage);
        hconcat(histImage3, histImage4, histImage3);
        vconcat(histImage, histImage3, histImage);

        imshow("Histograms", histImage );
        imshow("Camera", src);

        if((char)waitKey(30)=='q')
            return 0;
    }
}