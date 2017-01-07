#include "FilterFalsePositives.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

FilterFalsePositives::FilterFalsePositives() {

}

std::vector<cv::Rect> FilterFalsePositives::filter(Mat frame, std::vector<cv::Rect> obj, FilterType type) {
    switch (type) {
        case FilterType::MEAN_SQUARE:
            return filterMeanSquare(frame, obj);
        case FilterType::HSV_ROAD:
            return filterHSVRoad(frame, obj);
    }
}

double FilterFalsePositives::mse(const Mat& frame1, const Mat& frame2) {
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

double FilterFalsePositives::diffUpDown(const Mat& in) {
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

double FilterFalsePositives::diffLeftRight(const Mat& in) {
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

std::vector<cv::Rect> FilterFalsePositives::filterMeanSquare(Mat frame, std::vector<cv::Rect> obj) {
    std::vector<cv::Rect> cars;
    for (auto& car : obj) {
        Mat obj_frame = frame(car);
        double diffX = diffLeftRight(obj_frame);
        double diffY = diffUpDown(obj_frame);

        if (diffX > 80 && diffX < 155 && diffY > 200) {
            cars.push_back(car);
        }
    }
    return cars;
}

std::vector<cv::Rect> FilterFalsePositives::filterHSVRoad(Mat frame, std::vector<cv::Rect> obj) {
    // Equalize channels and bring back the color
    Mat frame_ycrcb, frame_gray, frame_hsv, canny;
    cvtColor(frame, frame_ycrcb,CV_BGR2YCrCb);
    vector<Mat> channels;
    split(frame_ycrcb, channels);
    equalizeHist(channels[0], channels[0]);
    Mat result;
    merge(channels, frame_ycrcb);
    cvtColor(frame_ycrcb, frame, CV_YCrCb2BGR);

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    cvtColor(frame, frame_hsv, CV_BGR2HSV);

    GaussianBlur(frame_gray, frame_gray, Size(3, 3), 0, 0);
    Canny(frame_gray, canny, 100, 200, 3);

    Mat cannyInv;
    threshold(canny, cannyInv, 128, 255, THRESH_BINARY_INV);

    Mat hsv_filtered;

    int averageHue = frame_hsv.at<Vec3b>(390, 220)[0];//scalar[0] / (frame_hsv.cols * frame_hsv.rows);
    int averageSat = frame_hsv.at<Vec3b>(390, 220)[1];//scalar[1] / (frame_hsv.cols * frame_hsv.rows);
    int averageVal = frame_hsv.at<Vec3b>(390, 220)[2];//scalar[2] / (frame_hsv.cols * frame_hsv.rows);
    inRange(frame_hsv, cv::Scalar(averageHue - 200, averageSat - 40, averageVal - 20),
            cv::Scalar(averageHue + 200, averageSat + 40, averageVal + 20), hsv_filtered);

    // Create a black image with white blocks where the cars are located
    Mat frame_cars_white_blk = Mat::zeros(hsv_filtered.rows, hsv_filtered.cols, hsv_filtered.type());
    std::vector<cv::Rect> cars;
    for (const auto& car : obj) {
        rectangle(frame_cars_white_blk, car, Scalar(255), CV_FILLED);
    }

    imshow("a", frame_cars_white_blk);

    Mat bitwise;
    // Bitwise AND between the black image and HSV filtered image
    bitwise_and(hsv_filtered, frame_cars_white_blk, bitwise);
    imshow("b", bitwise);
    for (auto& car : obj) {
        Mat crop = bitwise(car);
        if (sum(crop).val[0] >= (crop.rows * crop.cols * 255 * 0.2) &&
                sum(crop).val[0] < (crop.rows * crop.cols * 255 * 0.9)) {
            cars.push_back(car);
        }
    }
    return cars;
}
