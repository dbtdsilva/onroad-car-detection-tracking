#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    char *imageName1 = argv[1];
    char *imageName2 = argv[2];

    Mat image1 = imread(imageName1, 1);
    Mat image2 = imread(imageName2, 1);

    if (argc != 3) {
        cout << "Expected two arguments!" << endl;
        return -1;
    }
    if (!image1.data) {
        cout << "Failed to read image " << imageName1 << endl;
        return -1;
    }
    if (!image2.data) {
        cout << "Failed to read image " << imageName2 << endl;
        return -1;
    }

    double alpha = 0.5; double beta; double input;
    cout << "Simple Linear Blender" << endl;
    cout << "* Enter alpha [0-1]: ";
    cin >> input;

    alpha = input >= 0.0 && input <= 1.0 ? input : alpha;
    Mat dst;

    beta = (1.0 - alpha);
    addWeighted(image1, alpha, image2, beta, 0.0, dst);

    namedWindow("image1", CV_WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", CV_WINDOW_AUTOSIZE);
    imshow("image2", image2);

    namedWindow("Linear blend image", CV_WINDOW_AUTOSIZE);
    imshow("Linear blend image", dst);

    waitKey(0);

    return 0;
}