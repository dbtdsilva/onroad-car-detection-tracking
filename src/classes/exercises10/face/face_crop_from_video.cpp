#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

void detectAndSave( Mat frame);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";
String crop_window_name = "Capture - Face Crop";
int imageCounter = 0;
int main( int argc, char **argv )
{
    VideoCapture capture;
    Mat frame;

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    //-- 2. Read the video stream
    capture.open( -1 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    cout << "Press 's' to store a frame." << endl;

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndSave( frame );
        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }
    return 0;
}
void detectAndSave( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Mat croppedImage;
        cv::Mat(frame_gray, faces[i]).copyTo(croppedImage);
        Mat croppedDest;
        Size size(200,200);
        resize(croppedImage,croppedDest,size);
        imshow( crop_window_name, croppedDest );
        char c = (char)waitKey(10);
        if(c=='s'){
            imageCounter++;
            cout << "saving image " << format("%d.png", imageCounter) << endl;

            imwrite( format("%d.png", imageCounter) , croppedDest );
        }

        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
    }

    imshow( window_name, frame );
}