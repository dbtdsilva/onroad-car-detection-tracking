#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int FindAndDisplayChessboard(Mat image, int board_w, int board_h, vector<Point2f> *corners) {
    CvSize board_sz = cvSize(board_w, board_h);
    Mat grey_image;
    cvtColor(image, grey_image, CV_BGR2GRAY);

    // find chessboard corners
    bool found = findChessboardCorners(grey_image, board_sz, *corners, 0);
    cornerSubPix(grey_image, *corners, Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

    // Draw results
    drawChessboardCorners(image, board_sz, Mat(*corners), found);
    return (int) corners->size();
}

int main(int argc, char **argv) {
    // ChessBoard Properties
    int n_boards = 13; //Number of images
    int board_w = 9;
    int board_h = 6;
    int board_sz = board_w * board_h;

    char filenameL[200];
    char filenameR[200];
    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points_left;
    vector<vector<Point2f> > image_points_right;
    vector<vector<Point2f> > corners(2);
    int corner_count_left, corner_count_right;
    Mat imageL, imageR;

    vector<Point3f> obj;
    for (int j = 0; j < board_sz; j++)
        obj.push_back(Point3f(float(j / board_w), float(j % board_w), 0.0));

    for (int i = 0; i < n_boards; i++) {
        sprintf(filenameL, "left%02d.jpg", i + 1);
        printf("Reading %s \n", filenameL);
        imageL = imread(filenameL, CV_LOAD_IMAGE_COLOR);
        if (!imageL.data) {
            printf("\nCould not load image file: %s\n", filenameL);
            getchar();
            return 0;
        }

        sprintf(filenameR, "right%02d.jpg", i + 1);
        printf("Reading %s \n", filenameR);
        imageR = imread(filenameR, CV_LOAD_IMAGE_COLOR);
        if (!imageR.data) {
            printf("\nCould not load image file: %s\n", filenameR);
            getchar();
            return 0;
        }

        // find and display corners
        corner_count_left = FindAndDisplayChessboard(imageL, board_w, board_h, &corners[0]);
        corner_count_right = FindAndDisplayChessboard(imageR, board_w, board_h, &corners[1]);

        if ((corner_count_left == board_w * board_h) && (corner_count_right == board_w * board_h)) {
            object_points.push_back(obj);
            image_points_left.push_back(corners[0]);
            image_points_right.push_back(corners[1]);

        }

        putText(imageL, filenameL , cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
        putText(imageR, filenameR, cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        hconcat(imageL, imageR, imageL);
        imshow("Calibration", imageL);
        waitKey(0);
    }

    Mat intrinsic[2];
    intrinsic[0] = Mat::eye(3, 3, CV_64F);
    intrinsic[1] = Mat::eye(3, 3, CV_64F);

    Mat distortionCoefficients[2];
    Mat rotationMatrix;
    Mat translationVector;
    Mat essentialMatrix;
    Mat fundamentalMatrix;

#if CV_VERSION_MAJOR < 3
    double rms = stereoCalibrate(object_points, image_points_left, image_points_right,
                                 intrinsic[0], distortionCoefficients[0],
                                 intrinsic[1], distortionCoefficients[1],
                                 imageR.size(), rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix,
                                 TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                                 CV_CALIB_FIX_ASPECT_RATIO +
                                 CV_CALIB_ZERO_TANGENT_DIST +
                                 CV_CALIB_SAME_FOCAL_LENGTH +
                                 CV_CALIB_RATIONAL_MODEL +
                                 CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
#else
    double rms = stereoCalibrate(object_points, image_points_left, image_points_right,
                                 intrinsic[0], distortionCoefficients[0],
                                 intrinsic[1], distortionCoefficients[1],
                                 imageR.size(), rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix,
                                 CV_CALIB_FIX_ASPECT_RATIO +
                                 CV_CALIB_ZERO_TANGENT_DIST +
                                 CV_CALIB_SAME_FOCAL_LENGTH +
                                 CV_CALIB_RATIONAL_MODEL +
                                 CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5,
                                 TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
#endif
    cout << "re-projection error = " << rms << endl;

    FileStorage fs("../CamParams.xml", FileStorage::WRITE);
    fs << "intrinsic0" << intrinsic[0];
    fs << "intrinsic1" << intrinsic[1];
    fs << "distortionCoefficients0" << distortionCoefficients[0];
    fs << "distortionCoefficients1" << distortionCoefficients[1];
    fs << "rotationMatrix" << rotationMatrix;
    fs << "translationVector" << translationVector;
    fs << "essentialMatrix" << essentialMatrix;
    fs << "fundamentalMatrix" << fundamentalMatrix;
    fs.release();

    return 0;
}
