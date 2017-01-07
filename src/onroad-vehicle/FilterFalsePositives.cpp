#include "FilterFalsePositives.h"

using namespace std;
using namespace cv;

FilterFalsePositives::FilterFalsePositives() {

}

bool FilterFalsePositives::filter(Mat frame, FilterType type) {
    switch (type) {
        case FilterType::MEAN_SQUARE:
            return filterMeanSquare(frame);
    }
}

double mse(const Mat& frame1, const Mat& frame2) {
    Mat s1;
    Mat f1 = frame1.clone();
    Mat f2 = frame2.clone();
    f1.convertTo(f1, CV_32F);
    f2.convertTo(f2, CV_32F);

    absdiff(frame1, frame2, s1); // |I1 - I2|
    s1 = s1.mul(s1);             // |I1 - I2|^2

    Scalar s = sum(s1);
    double mse  = s[0] / (double)(frame1.size().height * frame1.size().width);
    return mse;
}

double diffUpDown(const Mat& in) {
    Mat frame = in.clone();
    int height = frame.size().height;
    int width = frame.size().width;
    int half = height / 2;

    Rect topCrop(0, 0, width, half);
    Rect bottomCrop(0, half, width, half);
    Mat top = frame(topCrop);
    Mat bottom = frame(bottomCrop);

    flip(top, top, 1);
    resize(bottom, bottom, Size(32, 64));
    resize(top, top, Size(32,64));
    return mse(top, bottom);
}

double diffLeftRight(const Mat& in) {
    Mat frame = in.clone();
    int height = frame.size().height;
    int width = frame.size().width;
    int half = width / 2;

    Rect leftCrop(0, 0, half, height);
    Rect rightCrop(half, 0, half, height);
    Mat left = frame(leftCrop);
    Mat right = frame(rightCrop);

    flip(right, right, 1);
    resize(left, left, Size(32, 64));
    resize(right, right, Size(32,64));

    return mse(left, right);
}

bool FilterFalsePositives::filterMeanSquare(Mat frame) {
    double diffX = diffLeftRight(frame);
    double diffY = diffUpDown(frame);

    return (diffX > 40 && diffX < 175 && diffY > 200);
}
