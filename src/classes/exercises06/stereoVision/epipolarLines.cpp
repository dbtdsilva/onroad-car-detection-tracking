#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

Mat undistorted_left, undistorted_right;
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
            bool imageRight = x > undistorted_left.size().width;

            if(imageRight)
                x %= undistorted_left.size().width;

            vector<Vec3f> lines;
            Point from(0,0);
            Point to(undistorted_left.size().width, 0);
            vector<Point2f > points;
            points.push_back(Point2f(x,y));

            if(imageRight){
                computeCorrespondEpilines(Mat(points), 2, fundamentalMatrix, lines);
                from.y = (- lines[0][0] * from.x - lines[0][2]) / lines[0][1];
                to.y = (- lines[0][0] * to.x - lines[0][2]) / lines[0][1];

                line(undistorted_left, from, to, Scalar(0, 255, 0));
            }
            else{
                computeCorrespondEpilines(Mat(points), 1, fundamentalMatrix, lines);
                from.y = (- lines[0][0] * from.x - lines[0][2]) / lines[0][1];
                to.y = (- lines[0][0] * to.x - lines[0][2]) / lines[0][1];

                line(undistorted_right, from, to, Scalar(0, 255, 0));
            }
            Mat dst;
            hconcat(undistorted_left, undistorted_right, dst);
            imshow("Undistorted stereo images", dst);

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
    Mat dst;

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

        undistort( src_left,  undistorted_left, intrinsic[0], distortionCoefficients[0]);
        putText(undistorted_left, filenameL, cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        undistort( src_right, undistorted_right, intrinsic[1], distortionCoefficients[1]);
        putText(undistorted_right, filenameR, cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        hconcat(undistorted_left, undistorted_right, dst);
        imshow("Undistorted stereo images", dst);
        setMouseCallback( "Undistorted stereo images", mouseHandler);
        waitKey(0);
    }

    return 0;
}
