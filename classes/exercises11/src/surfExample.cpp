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

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
    if( argc != 4 )
    { readme(); return -1; }

    Mat img_object = imread( argv[1], IMREAD_GRAYSCALE );
    Mat img_scene = imread( argv[2], IMREAD_GRAYSCALE );

    if( !img_object.data || !img_scene.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //-- Step 1: Detect the keypoints using SURF Detector
    int hessianThreshold = atoi(argv[3]);
    Ptr<SURF> detector = SURF::create( hessianThreshold );
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    detector->detect( img_object, keypoints_object );
    detector->detect( img_scene, keypoints_scene );


    Mat img_keypoints_object, img_keypoints_scene;

    drawKeypoints( img_object, keypoints_object, img_keypoints_object, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( img_scene, keypoints_scene, img_keypoints_scene, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    imshow("keypoints object", img_keypoints_object);
    imshow("keypoints scene", img_keypoints_scene);
    imwrite("kp_object.jpg", img_keypoints_object);
    imwrite("kp_scene.jpg", img_keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_object, descriptors_scene;
    detector->compute( img_object, keypoints_object, descriptors_object );
    detector->compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    std::vector< DMatch > matches;
    matcher->match( descriptors_object, descriptors_scene, matches );


    double min_dist = 100;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
    }
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance <= max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }
    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imwrite( "good_matches.jpg", img_matches );
    imshow("good matches", img_matches);
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    imwrite( "detection.jpg", img_matches );
    imshow("detection", img_matches);
    waitKey(0);

    return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./surfExample <img1> <img2> <hessianThreshold>" << std::endl; }