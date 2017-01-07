#include "DetectorMatchingFeatures.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

DetectorMatchingFeatures::DetectorMatchingFeatures() : DetectorMatchingFeatures(3) {

}

DetectorMatchingFeatures::DetectorMatchingFeatures(int history) : DetectorMatchingFeatures(history, 100) {

}

DetectorMatchingFeatures::DetectorMatchingFeatures(int history, int num_features) :
        history_(history), frames_received_(0), num_features_(num_features) {

}

vector<Rect> DetectorMatchingFeatures::detect(Mat img_next_frame) {
    vector<Rect> detections;
    if (img_next_frame.empty())
        return detections;

    frames_received_ += 1;
    buffer_.push_back(img_next_frame.clone());
    if (frames_received_ <= history_)
        return detections;

    Mat img_prev_frame = buffer_.front().clone();
    buffer_.erase(buffer_.begin());

    if(!img_next_frame.data || !img_prev_frame.data)
        return detections;

    cv::cvtColor(img_next_frame, img_next_frame, CV_BGR2GRAY);
    cv::cvtColor(img_prev_frame, img_prev_frame, CV_BGR2GRAY);

    // Step 1: Detect the key points using SIFT Detector
    Ptr<SURF> detector = SURF::create(num_features_);
    std::vector<KeyPoint> keypoints_next_frame, keypoints_prev_frame;
    detector->detect(img_next_frame, keypoints_next_frame);
    detector->detect(img_prev_frame, keypoints_prev_frame);

    Mat img_keypoints_next_frame, img_keypoints_prev_frame;
    drawKeypoints(img_next_frame, keypoints_next_frame, img_keypoints_next_frame,
                  Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawKeypoints(img_prev_frame, keypoints_prev_frame, img_keypoints_prev_frame,
                  Scalar::all(-1), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // imshow("keypoints next_frame", img_keypoints_next_frame);
    // imshow("keypoints prev_frame", img_keypoints_prev_frame);

    // Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_next_frame, descriptors_prev_frame;
    detector->compute(img_next_frame, keypoints_next_frame, descriptors_next_frame);
    detector->compute(img_prev_frame, keypoints_prev_frame, descriptors_prev_frame);

    // Step 3: Matching descriptor vectors using FLANN matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    std::vector<DMatch> matches;
    matcher->match(descriptors_next_frame, descriptors_prev_frame, matches);


    double min_dist = 100;
    for(int i = 0; i < descriptors_next_frame.rows; i++) {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
    }

    std::vector<DMatch> good_matches;
    for(int i = 0; i < descriptors_next_frame.rows; i++) {
        if(matches[i].distance <= max(4 * min_dist, 0.02)) {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    drawMatches(img_next_frame, keypoints_next_frame, img_prev_frame, keypoints_prev_frame,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("good matches", img_matches);
    // Localize the next_frame
    std::vector<Point2f> next_frame;
    std::vector<Point2f> prev_frame;

    for(int i = 0; i < good_matches.size(); i++) {
        next_frame.push_back(keypoints_next_frame[ good_matches[i].queryIdx ].pt);
        prev_frame.push_back(keypoints_prev_frame[ good_matches[i].trainIdx ].pt);
    }

    Mat H = findHomography(next_frame, prev_frame);
    try {
        /*

        std::vector<Point2f> next_frame_corners(4);
        next_frame_corners[0] = cvPoint(0,0); next_frame_corners[1] = cvPoint(img_next_frame.cols, 0);
        next_frame_corners[2] = cvPoint(img_next_frame.cols, img_next_frame.rows); next_frame_corners[3] = cvPoint(0, img_next_frame.rows);
        std::vector<Point2f> prev_frame_corners(4);

        perspectiveTransform(next_frame_corners, prev_frame_corners, H);

        //-- Draw lines between the corners (the mapped next_frame in the prev_frame - image_2)
        line(img_matches, prev_frame_corners[0] + Point2f(img_next_frame.cols, 0), prev_frame_corners[1] +
        Point2f(img_next_frame.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, prev_frame_corners[1] + Point2f(img_next_frame.cols, 0), prev_frame_corners[2] +
        Point2f(img_next_frame.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, prev_frame_corners[2] + Point2f(img_next_frame.cols, 0), prev_frame_corners[3] +
        Point2f(img_next_frame.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, prev_frame_corners[3] + Point2f(img_next_frame.cols, 0), prev_frame_corners[0] +
        Point2f(img_next_frame.cols, 0), Scalar(0, 255, 0), 4);
        imshow("detection", img_matches);

         */
        Mat warped;
        warpPerspective(img_next_frame, warped, H, img_prev_frame.size());

        Mat diffImage;
        //subtract(img_prev_frame, warped, diffImage);
        cv::absdiff(img_prev_frame, warped, diffImage);

        Mat image(img_prev_frame.rows, img_prev_frame.cols, img_prev_frame.type(), Scalar(0,0,0));
        imshow("Warped", diffImage - img_prev_frame);
        imshow("Warped Diff", diffImage);
        //imshow("Diff2", img_next_frame - warped);
    } catch (Exception& e) {

    }
    return detections;
}