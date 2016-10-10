
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//http://academic.aua.am/Skhachat/Public/Papers%20on%20Face%20Detection/Survey%20on%20Skin%20Color%20Techniques.pdf
//Formula (10)
bool skin(int R, int G, int B) {
    return (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
}

int main(int argc, char **argv) {
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat src;
    Vec3b black = Vec3b::all(0);

    while (true) {
        cap >> src;

        Mat dst = src.clone();
        for(int i = 0; i < dst.rows; i++) {
            for (int j = 0; j < dst.cols; j++) {
                Vec3b pixel = dst.ptr<Vec3b>(i)[j];
                uchar B = pixel.val[0];
                uchar G = pixel.val[1];
                uchar R = pixel.val[2];
                if(!skin(R,G,B))
                    dst.ptr<Vec3b>(i)[j] = black;

            }
        }

        hconcat(src,dst,src);
        imshow("Skin Color", src);

        if((char)waitKey(30)=='q')
            return 0;
    }
}