#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int main(int argc, char **argv) {
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

    Mat frame_gray;
    if(argc < 3)
        cout << "usage: face_crop_from_image <source_file> <dest_file>" << endl;

    frame_gray = imread(argv[1]);
    string dest = argv[2];

    vector<Rect> faces;
    //Mat frame_gray;
    //cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ ) {
        Mat croppedImage;
        cv::Mat(frame_gray, faces[i]).copyTo(croppedImage);

        Size size(200, 200);//the dst image size,e.g.100x100
        Mat dst;//dst image
        Mat src;//src image
        resize(croppedImage, croppedImage, size);

        imwrite( dest, croppedImage );
    }


    return 0;
}