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
#include <sstream>

using namespace cv;
using namespace std;

// Function FindAndDisplayChessboard
// find corners in a cheesboard with board_w x board_h dimensions
// Display the corners in image and return number of detected corners
int FindAndDisplayChessboard(Mat image, int board_w, int board_h, std::vector<Point2f> *corners)
{
  int board_size = board_w * board_h;
  CvSize board_sz = cvSize(board_w, board_h);

  Mat gray_image;

  cvtColor(image, gray_image, CV_BGR2GRAY);

  // find chessboard corners
  bool found = findChessboardCorners(gray_image, board_sz, *corners, 0);
  // Draw results
  if (found)
  {
    cornerSubPix(gray_image, *corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
    drawChessboardCorners(image, board_sz, Mat(*corners), found);
    //imshow("Calibration", image);
    //printf("\n Number of corners: %lu", corners->size());
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

  char recv, last_key;
  bool not_saved = false;

  int total_calibrations = 20;
  stringstream ss;
  while(sucesses < total_calibrations)
  {
    cap >> image; // get a new frame from camera
    corner_count = FindAndDisplayChessboard(image, board_w, board_h, &corners);

    ss.str("");
    ss << sucesses << " out of " << total_calibrations << " calibrations";
    putText(image, "Press any key to calibrate! Q to exit", cvPoint(10, 20),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
    putText(image, ss.str(), cvPoint(10, 40),FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);
    imshow("Camera", image);

    last_key = recv;
    recv = waitKey(30);
    if (recv == 'q' || recv == 'Q')
      break;
    if (recv != last_key)
      not_saved = true;
    
    if (corner_count == board_w * board_h && not_saved && recv != -1)
    {
      image_points.push_back(corners);
      object_points.push_back(obj);
      sucesses++;

      not_saved = false;  
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
