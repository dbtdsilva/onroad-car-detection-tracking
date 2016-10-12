
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

//http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

int main(int argc, char **argv) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat src, src_gray, sobel, scharr;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    int kernel = 1;


    while (true) {
        cap >> src;

        GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

        /// Convert it to gray
        cvtColor( src, src_gray, CV_BGR2GRAY );

        /// Generate grad_x and grad_y
        Mat grad_x_sobel, grad_y_sobel,grad_x_scharr, grad_y_scharr;
        Mat abs_grad_x_sobel, abs_grad_y_sobel, abs_grad_x_scharr, abs_grad_y_scharr;

        /// Gradient X
        Scharr( src_gray, grad_x_scharr, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x_scharr, abs_grad_x_scharr );

        Sobel( src_gray, grad_x_sobel, ddepth, 1, 0, 1+kernel*2, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x_sobel, abs_grad_x_sobel );

        /// Gradient Y
        Scharr( src_gray, grad_y_scharr, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y_scharr, abs_grad_y_scharr );

        Sobel( src_gray, grad_y_sobel, ddepth, 0, 1, 1+kernel*2, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y_sobel, abs_grad_y_sobel );

        /// Total Gradient (approximate)
        addWeighted( abs_grad_x_sobel, 0.5, abs_grad_y_sobel, 0.5, 0, sobel );
        addWeighted( abs_grad_x_scharr, 0.5, abs_grad_y_scharr, 0.5, 0, scharr );

        putText(sobel, "Sobel, kernel " + to_string(1+kernel*2), cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        putText(scharr, "Scharr", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        hconcat(sobel, scharr, sobel);

        imshow("image", sobel);
        createTrackbar( "scale", "image", &scale, 100 );
        createTrackbar( "Sobel Kernel Size", "image", &kernel, 3 );
        if((char)waitKey(30)=='q')
            return 0;
    }
}