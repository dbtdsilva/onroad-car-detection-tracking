#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    char *imageName = argv[1];

    Mat image = imread(imageName, 1);

    if (argc != 2) {
        cout << "Expected one argument!" << endl;
        return -1;
    }
    if (!image.data) {
        cout << "Failed to read image " << imageName << endl;
        return -1;
    }
    Mat new_image = Mat::zeros(image.size(), image.type());
    double alpha, beta;
    cout << " Basic Linear Transforms " << endl;
    cout << "* Enter the saturation value [1.0-3.0]: ";
    cin >> alpha;
    cout<<"* Enter the brightness value [0-100]: ";
    cin >> beta;
    /*for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for(int c = 0; c < image.channels(); c++) {
                new_image.at<Vec3b>(y,x)[c] = saturate_cast<uchar>(
                        alpha * (image.at<Vec3b>(y,x)[c]) + beta
                );
            }
        }
    }*/

    for (MatIterator_<Vec3b> it = image.begin<Vec3b>(); it != image.end<Vec3b>(); ++it) {
        for(int c = 0; c < image.channels(); c++) {
            new_image.at<Vec3b>(it.pos())[c] = saturate_cast<uchar>(
                    alpha * (*it)[c] + beta);
        }
    }

    namedWindow("Original image", CV_WINDOW_AUTOSIZE);
    imshow("Original image", image);

    namedWindow("Modified image", CV_WINDOW_AUTOSIZE);
    imshow("Modified image", new_image);

    waitKey(0);

    return 0;
}