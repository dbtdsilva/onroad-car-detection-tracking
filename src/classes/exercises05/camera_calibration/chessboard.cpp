/***********************************************************************************
Name:           chessboard.cpp
Revision:
Author:         Paulo Dias
Comments:       ChessBoard Tracking


images
Revision:
Libraries:
***********************************************************************************/
#include <iostream>
#include <vector>

// OpenCV Includes
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
// Function FindAndDisplayChessboard
// find corners in a cheesboard with board_w x board_h dimensions
// Display the corners in image and return number of detected corners
int FindAndDisplayChessboard(Mat image, int board_w, int board_h, std::vector<Point2f> *corners)
{
  int board_size = board_w * board_h;
  CvSize board_sz = cvSize(board_w, board_h);

  Mat grey_image;

  cvtColor(image, grey_image, CV_BGR2GRAY);

  // find chessboard corners
  bool found = findChessboardCorners(grey_image, board_sz, *corners, 0);

  // Draw results
  if (true)
  {
    drawChessboardCorners(image, board_sz, Mat(*corners), found);
    //imshow("Calibration", image);
    printf("\n Number of corners: %lu", corners->size());
    //waitKey(0);
  }
  return corners->size();
}

int main(int argc, char **argv)
{
  
  // ChessBoard Properties
  int n_boards = 13; //Number of images
  int board_w = 9;
  int board_h = 6;

  int board_sz = board_w * board_h;

  // Chessboard coordinates and image pixels
  std::vector<std::vector<Point3f> > object_points;
  std::vector<std::vector<Point2f> > image_points;

  // Corners detected in each image
  std::vector<Point2f> corners;

  int corner_count;

  Mat image;
  int i;

  int sucesses = 0;


  // chessboard coordinates
  std::vector<Point3f> obj;
  for (int j = 0; j < board_sz; j++)
    obj.push_back(Point3f(float(j / board_w), float(j % board_w), 0.0));

  cv::VideoCapture cap(0); // open the video camera no. 0
  // Access Video
  if (!cap.isOpened()) // if not success, exit program
  {
    std::cout << "Cannot open the video file" << std::endl;
    getchar();
    return -1;
  }

  while(cv::waitKey(30) < 0)
  {
    cap >> image; // get a new frame from camera
    imshow("cam", image);

    corner_count = FindAndDisplayChessboard(image, board_w, board_h, &corners);
    imshow("Camera", image);
    if (corner_count == board_w * board_h)
    {
      image_points.push_back(corners);
      object_points.push_back(obj);
      sucesses++;

      waitKey(0);
    }

  }

  /*for (i = 0; i < n_boards; i++)
  {
    // read image
    sprintf(filename, "..//images//left%02d.jpg", i + 1);
    printf("\nReading %s", filename);
    image = imread(filename, CV_LOAD_IMAGE_COLOR);


    if (!image.data)
    {
      printf("\nCould not load image file: %s\n", filename);
      getchar();
      return 0;
    }

    // find and display corners
    corner_count = FindAndDisplayChessboard(image, board_w, board_h, &corners);

    if (corner_count == board_w * board_h)
    {
      image_points.push_back(corners);
      object_points.push_back(obj);
      sucesses++;
    }
  }*/

  Mat intrinsic = Mat(3, 3, CV_32FC1);
  Mat distCoeffs;
  std::vector<Mat> rvecs;
  std::vector<Mat> tvecs;

  calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs, 0);

  std::cout << std::endl << "Intrinsics = " << std::endl << " " << intrinsic << std::endl << std::endl;
  std::cout << std::endl << "Distortion = " << std::endl << " " << distCoeffs << std::endl << std::endl;
  std::cout << std::endl << "Translations = " << std::endl ;
  for (i = 0; i < n_boards; i++)
    std::cout << std::endl << tvecs.at(i);
  std::cout << std::endl << "Rotations= " << std::endl;
  for (i = 0; i < n_boards; i++)
    std::cout << std::endl << rvecs.at(i);

  FileStorage fs("../CamParams.xml", FileStorage::WRITE);
  fs << "cameraMatrix" << intrinsic << "distCoeffs" << distCoeffs;
  fs.release();


  // Create a cube
  /*std::vector<Point3f> new_object_points;
  new_object_points.push_back(Point3f(0.0, 0.0, 0.0)); // 0
  new_object_points.push_back(Point3f(0.0, 0.0, 1.0)); // 1
  new_object_points.push_back(Point3f(1.0, 0.0, 0.0)); // 2
  new_object_points.push_back(Point3f(1.0, 0.0, 1.0)); // 3
  new_object_points.push_back(Point3f(0.0, 1.0, 0.0)); // 4
  new_object_points.push_back(Point3f(0.0, 1.0, 1.0)); // 5
  new_object_points.push_back(Point3f(1.0, 1.0, 0.0)); // 6
  new_object_points.push_back(Point3f(1.0, 1.0, 1.0)); // 7

  for (i = 0; i < n_boards; i++)
  {
    // read image
    sprintf(filename, "..//images//left%02d.jpg", i + 1);
    printf("\nReading %s", filename);
    image = imread(filename, CV_LOAD_IMAGE_COLOR);


    if (!image.data)
    {
      printf("\nCould not load image file: %s\n", filename);
      getchar();
      return 0;
    }

    std::vector<cv::Point2f> projected_points;
    projectPoints(new_object_points, rvecs.at(i), tvecs.at(i), intrinsic, distCoeffs, projected_points);

    line(image, projected_points[0], projected_points[1], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[0], projected_points[2], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[0], projected_points[4], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[1], projected_points[3], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[1], projected_points[5], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[2], projected_points[3], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[2], projected_points[6], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[3], projected_points[7], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[4], projected_points[5], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[4], projected_points[6], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[5], projected_points[7], Scalar(0, 0, 255), 2, 8, 0);
    line(image, projected_points[6], projected_points[7], Scalar(0, 0, 255), 2, 8, 0);
    imshow("Calibration", image);
    waitKey(0);
  }*/
  
  return 0;
}
