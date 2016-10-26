#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Mat dst_left, dst_right;
Mat intrinsic[2];
Mat distortionCoefficients[2];
Mat rotationMatrix;
Mat translationVector;
Mat essentialMatrix;
Mat fundamentalMatrix;

void mouseHandler(int event, int x, int y, int flags, void* param)
{
    switch(event){
        case CV_EVENT_LBUTTONDOWN:

            Point from(0,y);
            Point to(dst_left.size().width, y);

            line(dst_left, from, to, Scalar(0, 255, 0));
            line(dst_right, from, to, Scalar(0, 255, 0));
            Mat dst;
            hconcat(dst_left, dst_right, dst);
            imshow("Rectified stereo images", dst);

            break;
    }
}


int main(int argc, char **argv)
{
    FileStorage fs("../CamParams.xml", FileStorage::READ);

    char filenameL[200];
    char filenameR[200];
    int n_boards = 13; //Number of images

    fs["intrinsic0"] >> intrinsic[0];
    fs["intrinsic1"] >> intrinsic[1];
    fs["distortionCoefficients0"] >> distortionCoefficients[0];
    fs["distortionCoefficients1"] >> distortionCoefficients[1];
    fs["rotationMatrix"] >> rotationMatrix;
    fs["translationVector"] >> translationVector;
    fs["essentialMatrix"] >> essentialMatrix;
    fs["fundamentalMatrix"] >> fundamentalMatrix;

    Mat src_left, src_right;
    Mat R1, R2, P1, P2, Q;
    Mat map1x, map1y, map2x, map2y;
    Rect validPixROI[2];

    for (int i = 0; i < n_boards; i++) {
        sprintf(filenameL, "left%02d.jpg", i + 1);
        printf("Reading %s \n", filenameL);
        src_left = imread(filenameL, CV_LOAD_IMAGE_COLOR);
        if (!src_left.data) {
            printf("\nCould not load image file: %s\n", filenameL);
            getchar();
            return 0;
        }

        sprintf(filenameR, "right%02d.jpg", i + 1);
        printf("Reading %s \n", filenameR);
        src_right = imread(filenameR, CV_LOAD_IMAGE_COLOR);
        if (!src_right.data) {
            printf("\nCould not load image file: %s\n", filenameR);
            getchar();
            return 0;
        }

        stereoRectify(intrinsic[0], distortionCoefficients[0], intrinsic[1], distortionCoefficients[1],
                      src_left.size(), rotationMatrix, translationVector,
                      R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, src_left.size(), &validPixROI[0], &validPixROI[1]);

        initUndistortRectifyMap(intrinsic[0], distortionCoefficients[0], R1, P1, src_left.size(), CV_16SC2, map1x, map1y);
        initUndistortRectifyMap(intrinsic[1], distortionCoefficients[1], R2, P2, src_left.size(), CV_16SC2, map2x, map2y);

        remap(src_left, dst_left, map1x, map1y, INTER_LINEAR);
        remap(src_right, dst_right, map2x, map2y, INTER_LINEAR);

        for(int y=25; y<src_left.size().height; y+=25) {
            Point from(0, y);
            Point to(dst_left.size().width, y);

            line(dst_left, from, to, Scalar(255, 0, 0));
            line(dst_right, from, to, Scalar(255, 0, 0));
        }

        Mat dst;
        hconcat(dst_left, dst_right, dst);
        imshow("Rectified stereo images", dst);
        setMouseCallback( "Rectified stereo images", mouseHandler);
        waitKey(0);
    }

    return 0;
}
