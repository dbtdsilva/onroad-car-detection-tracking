#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <string>

using namespace std;
using namespace cv;


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

bool isNewRoi(const Rect& rect, vector<Rect> &rects){
    for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it) {
        if(abs((*it).x - rect.x) < 40 && abs((*it).y - rect.y) < 40)
            return false;
    }
    return true;
}

double overlapPercentage(const Rect& A, const Rect2d& B){
    int x_overlap = max(0, min(A.x+A.width,(int)B.x + (int)B.width) - max(A.x,(int)B.x));
    int y_overlap = max(0, min(A.y+A.height,(int)B.y + (int)B.height) - max(A.y,(int)B.y));
    int overlapArea = x_overlap * y_overlap;
    return (double)overlapArea/(A.width*A.height);
}


int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
    }

    Mat frame;
    CascadeClassifier car_cascade;
    if(!car_cascade.load(argv[2])) {
        cout << "Failed to load " << argv[2] << endl;
        exit(EXIT_FAILURE);
    }

    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int keyboard;
    vector<Rect> cars;
    vector<Rect> rects;

    vector<Ptr<Tracker> > trackers;
    vector<Rect2d> rois;

    int actualFrameCount = 0;

    int frameCount = 0, scale = 2, height, width, minY;
    double diffX, diffY;
    Mat frame_gray;

    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        actualFrameCount++;



        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // scale down the frame to remove the very small boxes
        height = frame_gray.size().height;
        width = frame_gray.size().width;
        Mat tmp_frame_gray;
        resize(frame_gray, tmp_frame_gray, Size(width/scale, height/scale));

        car_cascade.detectMultiScale(tmp_frame_gray, cars, 1.2, 2);

        //to remove the very big boxes
        minY = tmp_frame_gray.size().height*0.3;

        vector<Rect> newRegions;
        for (vector<Rect>::iterator it = cars.begin(); it != cars.end(); ++it) {
            Mat roi_image = tmp_frame_gray(*it);
            if ((*it).y > minY) {
                diffX = diffLeftRight(roi_image);
                diffY = diffUpDown(roi_image);

                //cout << diffX << "  " << diffY << endl;

                if (diffX > 40 && diffX < 175 && diffY > 200) {
                    newRegions.push_back(Rect((*it).x * scale, (*it).y * scale, (*it).width * scale, (*it).height * scale));
                }
            }
            waitKey(10);
        }

        for (vector<Rect>::iterator it = newRegions.begin(); it != newRegions.end(); ++it)
            if(isNewRoi(*it, rects))
                rects.push_back(*it);

        for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it) {

            for(unsigned long i =0; i<trackers.size(); i++){
                if(overlapPercentage(*it, rois[i])>0.3){
                    rois.erase(rois.begin()+i);
                    trackers.erase(trackers.begin()+i);
                    i--;
                }
            }
            trackers.push_back(Tracker::create("KCF"));
            rois.push_back(Rect2d(*it));
            trackers[trackers.size()-1]->init(frame, rois[rois.size()-1]);

        }

        for(unsigned long i =0; i<trackers.size(); i++){
            trackers[i]->update(frame, rois[i]);
            if(rois[i].y + rois[i].height > frame_gray.size().height || rois[i].x + rois[i].width > frame_gray.size().width){
                rois.erase(rois.begin()+i);
                trackers.erase(trackers.begin()+i);
                i--;
            }
            else
                rectangle(frame, rois[i], Scalar(255, 0, 0), 2);
        }

        frameCount++;
        if(frameCount >= 5){
            frameCount = 0;
            rects.clear();
        }

        putText(frame, "Frame "+to_string(actualFrameCount), cvPoint(8,20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);

        imshow("Camera", frame);
        keyboard = waitKey( 100 );

    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}