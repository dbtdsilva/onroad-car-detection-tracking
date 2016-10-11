
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// http://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html#smoothing

int main(int argc, char **argv) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat src;
    int kernel_size = 0, i = 4;
    while (true) {
        cap >> src;
        char received = (char)waitKey(30);
        if (received == '+')
            i++;
        else if (received == '-')
            i--;
        else if (received == 'q' || received == 'Q')
            break;
        kernel_size = 3 + 2 * ( i % 25 );

        Mat blur_dst, gaussian_blur_dst,median_blur_dst,bilateral_filter_dst;

        blur( src, blur_dst, Size( kernel_size, kernel_size ), Point(-1,-1) );
        GaussianBlur( src, gaussian_blur_dst, Size( kernel_size, kernel_size ), 0, 0 );
        medianBlur ( src, median_blur_dst, kernel_size );
        bilateralFilter ( src, bilateral_filter_dst, kernel_size, kernel_size*2, kernel_size/2 );

        putText(blur_dst, "Blur", cvPoint(10,40), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);
        putText(gaussian_blur_dst, "Gaussian Blur", cvPoint(10,20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);
        putText(median_blur_dst, "Median Blur", cvPoint(10,20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);
        putText(bilateral_filter_dst, "Bilateral Filter", cvPoint(10,20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);

        hconcat(blur_dst,gaussian_blur_dst,blur_dst);
        hconcat(median_blur_dst,bilateral_filter_dst,median_blur_dst);
        vconcat(blur_dst, median_blur_dst, blur_dst);

        putText(blur_dst, "Kernel Size: " + to_string(kernel_size),
                cvPoint(10, 20),
                FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,0,0), 1, CV_AA);
        imshow("Filters", blur_dst);
    }
}