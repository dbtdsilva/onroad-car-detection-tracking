#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

void mouseHandler(int event, int x, int y, int flags, void *param) {
    switch (event) {
        case CV_EVENT_LBUTTONDOWN:

            Point from(0, y);
            Point to(dst_left.size().width, y);

            line(dst_left, from, to, Scalar(0, 255, 0));
            line(dst_right, from, to, Scalar(0, 255, 0));
            Mat dst;
            hconcat(dst_left, dst_right, dst);
            imshow("Rectified stereo images", dst);

            break;
    }
}


int main(int argc, char **argv) {
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

    cv::Mat imgDisparity8U, imgDisparity16S;

    // 1- Variable definition
    // the preset has to do with the system configuration (basic, fisheye, etc.)
    // ndisparities is the size of disparity range,
    // in which the optimal disparity at each pixel is searched for.
    // SADWindowSize is the size of averaging window used to match pixel blocks
    // (larger values mean better robustness to noise, but yield blurry disparity maps)
    int ndisparities = 16 * 5;
    int SADWindowSize = 21;
    cv::Ptr<cv::StereoBM> sbm;
    Mat gray_image_left, gray_image_right;

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

        initUndistortRectifyMap(intrinsic[0], distortionCoefficients[0], R1, P1, src_left.size(), CV_16SC2, map1x,
                                map1y);
        initUndistortRectifyMap(intrinsic[1], distortionCoefficients[1], R2, P2, src_left.size(), CV_16SC2, map2x,
                                map2y);

        remap(src_left, dst_left, map1x, map1y, INTER_LINEAR);
        remap(src_right, dst_right, map2x, map2y, INTER_LINEAR);

        Mat dst;
        hconcat(dst_left, dst_right, dst);
        imshow("Rectified stereo images", dst);
        //setMouseCallback("Rectified stereo images", mouseHandler);
        //waitKey(0);

        // -- 2. Call the constructor for StereoBM
        sbm = cv::StereoBM::create( ndisparities, SADWindowSize );

        // -- 3. Calculate the disparity image
        cvtColor(dst_left, gray_image_left, CV_BGR2GRAY);
        cvtColor(dst_right, gray_image_right, CV_BGR2GRAY);
        sbm->compute(gray_image_left, gray_image_right, imgDisparity16S);
        // -- Check its extreme values
        double minVal; double maxVal;
        cv::minMaxLoc(imgDisparity16S, &minVal, &maxVal);
        printf("Min disp: %f Max value: %f \n", minVal, maxVal);
        // -- 4. Display it as a CV_8UC1 image
        // Display disparity as a CV_8UC1 image
        // the disparity will be 16-bit signed (fixed-point) or
        // 32-bit floating-point image of the same size as left.
        imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
        namedWindow("disparity", cv::WINDOW_NORMAL);
        imshow("disparity", imgDisparity8U);

        if (i == 5) {
            Mat image3d;
            reprojectImageTo3D(imgDisparity16S, image3d, Q);

            FileStorage fw("../Image3D_Reconstructed.xml", FileStorage::WRITE);
            fw << "Image" << dst;
            fw << "Image3D" << image3d;
            fw.release();
        }
        waitKey(0);
    }

    return 0;
}
