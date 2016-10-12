
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

    Mat src, sobel, scharr;

    while (true) {
        cap >> src;



        Mat dst = src.clone();



        hconcat(src,dst,src);
        imshow("image", src);

        if((char)waitKey(30)=='q')
            return 0;
    }
}