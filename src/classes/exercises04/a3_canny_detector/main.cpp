#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

// http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
int main( int argc, char** argv )
{
    Mat src, src_gray;
    Mat dst, detected_edges;

    int lowThreshold = 20;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 0;
    char* window_name = "Edge Map";

    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    while ((char) waitKey(30) != 'q') {
        cap >> src;

        /// Create a matrix of the same type and size as src (for dst)
        dst.create( src.size(), src.type() );
        /// Convert the image to grayscale
        cvtColor( src, src_gray, CV_BGR2GRAY );
        /// Reduce noise with a kernel 3x3
        blur( src_gray, detected_edges, Size(3,3) );
        /// Canny detector
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, 3 + 2 * kernel_size);
        /// Using Canny's output as a mask, we display our result
        dst = Scalar::all(0);

        src.copyTo( dst, detected_edges);

        putText(dst, "Canny, kernel " + to_string(3+kernel_size*2), cvPoint(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(255,255,255), 1, CV_AA);

        imshow( window_name, dst );

        createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, nullptr );
        createTrackbar( "Kernel:", window_name, &kernel_size, 2, nullptr );
    }

    return 0;
}