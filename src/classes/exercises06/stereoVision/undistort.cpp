#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    FileStorage fs("../CamParams.xml", FileStorage::READ);

    Mat intrinsic[2];
    Mat distortionCoefficients[2];
    Mat rotationMatrix;
    Mat translationVector;
    Mat essentialMatrix;
    Mat fundamentalMatrix;

    char filename[200];
    int n_boards = 13; //Number of images

    fs["intrinsic0"] >> intrinsic[0];
    fs["intrinsic1"] >> intrinsic[1];
    fs["distortionCoefficients0"] >> distortionCoefficients[0];
    fs["distortionCoefficients1"] >> distortionCoefficients[1];
    fs["rotationMatrix"] >> rotationMatrix;
    fs["translationVector"] >> translationVector;
    fs["essentialMatrix"] >> essentialMatrix;
    fs["fundamentalMatrix"] >> fundamentalMatrix;

    Mat src, dst;

    for (int i = 0; i < n_boards; i++) {
        sprintf(filename, "left%02d.jpg", i + 1);
        printf("Reading %s \n", filename);
        src = imread(filename, CV_LOAD_IMAGE_COLOR);
        if (!src.data) {
            printf("\nCould not load image file: %s\n", filename);
            getchar();
            return 0;
        }
        undistort( src,  dst, intrinsic[0], distortionCoefficients[0]);
        putText(src, "Original", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        putText(dst, "Undistorted", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        hconcat(src, dst, src);
        imshow("Undistort", src);
        waitKey(0);
    }
    for (int i = 0; i < n_boards; i++) {
        sprintf(filename, "right%02d.jpg", i + 1);
        printf("Reading %s \n", filename);
        src = imread(filename, CV_LOAD_IMAGE_COLOR);
        if (!src.data) {
            printf("\nCould not load image file: %s\n", filename);
            getchar();
            return 0;
        }

        undistort( src,  dst, intrinsic[1], distortionCoefficients[1]);
        putText(src, "Original", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        putText(dst, "Undistorted", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        hconcat(src, dst, src);
        imshow("Undistort", src);
        waitKey(0);
    }
    return 0;
}
