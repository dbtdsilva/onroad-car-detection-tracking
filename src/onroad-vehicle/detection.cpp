#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();

#include <list>
#define FRAMES_TO_SKIP 3

int main(int argc, char** argv)
{
    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int frames_captured = 0;
    vector<Mat> buffer;
    vector<int> val;
    int keyboard;
    do {
        Mat img_object;
        if (!capture.read(img_object)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting." << endl;
            exit(EXIT_FAILURE);
        }

        if (img_object.empty())
            continue;

        buffer.push_back(img_object.clone());
        val.push_back(frames_captured);
        if (frames_captured++ <= FRAMES_TO_SKIP)
            continue;

        Mat img_scene = buffer.front().clone();
        buffer.erase(buffer.begin());

        cout << val.front() << ", " << frames_captured << endl;
        val.erase(val.begin());

        if(!img_object.data || !img_scene.data) {
            std::cout<< "Error reading images " << std::endl;
            return -1;
        }
        cv::cvtColor(img_object, img_object, CV_BGR2GRAY);
        cv::cvtColor(img_scene, img_scene, CV_BGR2GRAY);

        //-- Step 1: Detect the keypoints using SIFT Detector
        int nFeatures = 100;
        Ptr<SURF> detector = SURF::create(nFeatures);
        std::vector<KeyPoint> keypoints_object, keypoints_scene;
        detector->detect(img_object, keypoints_object);
        detector->detect(img_scene, keypoints_scene);


        Mat img_keypoints_object, img_keypoints_scene;

        drawKeypoints(img_object, keypoints_object, img_keypoints_object, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        drawKeypoints(img_scene, keypoints_scene, img_keypoints_scene, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

        imshow("keypoints object", img_keypoints_object);
        imshow("keypoints scene", img_keypoints_scene);

        //-- Step 2: Calculate descriptors (feature vectors)
        Mat descriptors_object, descriptors_scene;
        detector->compute(img_object, keypoints_object, descriptors_object);
        detector->compute(img_scene, keypoints_scene, descriptors_scene);

        //-- Step 3: Matching descriptor vectors using FLANN matcher
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
        std::vector< DMatch > matches;
        matcher->match(descriptors_object, descriptors_scene, matches);


        double min_dist = 100;
        for(int i = 0; i < descriptors_object.rows; i++)
        { double dist = matches[i].distance;
            if(dist < min_dist) min_dist = dist;
        }

        std::vector< DMatch > good_matches;
        for(int i = 0; i < descriptors_object.rows; i++) {
            if(matches[i].distance <= max(4 * min_dist, 0.02)) {
                good_matches.push_back(matches[i]);
            }
        }

        Mat img_matches;
        drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                     good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("good matches", img_matches);
        //-- Localize the object
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for(int i = 0; i < good_matches.size(); i++) {
            obj.push_back(keypoints_object[ good_matches[i].queryIdx ].pt);
            scene.push_back(keypoints_scene[ good_matches[i].trainIdx ].pt);
        }

        Mat H = findHomography(obj, scene);
        try {
            /* Used to debug

            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint(img_object.cols, 0);
            obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
            std::vector<Point2f> scene_corners(4);


            perspectiveTransform(obj_corners, scene_corners, H);

            //-- Draw lines between the corners (the mapped object in the scene - image_2)
            line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] +
             Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] +
             Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] +
             Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] +
             Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
            imwrite("detection.jpg", img_matches);
            imshow("detection", img_matches); */
            Mat warped;
            warpPerspective(img_object, warped, H, img_scene.size());

            Mat diffImage;
            //subtract(img_scene, warped, diffImage);
            cv::absdiff(img_scene, warped, diffImage);

            Mat image(img_scene.rows, img_scene.cols, img_scene.type(), Scalar(0,0,0));
            imshow("Warped", diffImage - img_scene);
            imshow("Warped2", diffImage);
            imshow("Diff", diffImage);
            //imshow("Diff2", img_object - warped);
        } catch (Exception& e) { } // Might not be able to transform
        keyboard = waitKey(30);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27);
    return 0;
}