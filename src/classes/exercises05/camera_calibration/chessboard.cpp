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

  while(sucesses <= 15)
  {
    cap >> image; // get a new frame from camera
    imshow("cam", image);

    if (waitKey(30) >= 0)
      break;

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

  Mat intrinsic = Mat(3, 3, CV_32FC1);
  Mat distCoeffs;
  std::vector<Mat> rvecs;
  std::vector<Mat> tvecs;

  calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs, 0);

  std::cout << std::endl << "Intrinsics = " << std::endl << " " << intrinsic << std::endl << std::endl;
  std::cout << std::endl << "Distortion = " << std::endl << " " << distCoeffs << std::endl << std::endl;
  std::cout << std::endl << "Translations = " << std::endl ;
  for (i = 0; i < sucesses; i++)
    std::cout << std::endl << tvecs.at(i);
  std::cout << std::endl << "Rotations= " << std::endl;
  for (i = 0; i < sucesses; i++)
    std::cout << std::endl << rvecs.at(i);

  FileStorage fs("../CamParams.xml", FileStorage::WRITE);
  fs << "cameraMatrix" << intrinsic << "distCoeffs" << distCoeffs;
  fs.release();
  
  return 0;
}
