#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

static void read_data_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator);
static void read_labels_csv(const string& filename, vector<string>& labels, char separator);
void detectAndLabel( Mat frame, Ptr<FaceRecognizer> model, vector<string>& names );

//Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";
int imageCounter = 0;

int main( int argc, char **argv )
{
    //Get the path to the data CSV
    string fn_csv = argv[1];

    //Get the path to the labels CSV
    string labels_csv = argv[2];

    VideoCapture capture;
    Mat frame;

    // These vectors hold the images and corresponding labels and names for the labels:
    vector<Mat> images;
    vector<int> labels;
    vector<string> names;

    try {
        read_data_csv(fn_csv, images, labels, ';');
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    try {
        read_labels_csv(labels_csv, names, ';');
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << labels_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);

    //-- 1. Load the face cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };

    //-- 2. Read the video stream
    capture.open( -1 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    cout << "Press ESC to exit." << endl;

    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndLabel( frame, model, names );
        int c = waitKey(10);
        if( (char)c == 27 ) { break; } // escape
    }
    return 0;
}
void detectAndLabel( Mat frame, Ptr<FaceRecognizer> model, vector<string>& names)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        // Same transform as face_crop_from_video
        Mat croppedImage;
        cv::Mat(frame_gray, faces[i]).copyTo(croppedImage);
        Mat croppedDest;
        Size size(200,200);
        resize(croppedImage,croppedDest,size);

        // Get the model prediction
        int prediction = model->predict(croppedDest);

        // Draw a green rectangle around the detected face
        rectangle(frame, faces[i], CV_RGB(0, 255,0), 1);

        // Create the text we will annotate the box with
        string box_text = format("Prediction = %s", names.at(prediction).c_str());
        // Calculate the position for annotated text (make sure we don't put illegal values in there)
        int pos_x = std::max(faces[i].tl().x - 10, 0);
        int pos_y = std::max(faces[i].tl().y - 10, 0);
        // And now put it into the image
        putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    }
    imshow( window_name, frame );
}

static void read_data_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat tmp = imread(path, 2);
            images.push_back(tmp);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

static void read_labels_csv(const string& filename, vector<string>& labels, char separator) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, label, identifier;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, label, separator);
        getline(liness, identifier);
        if(!label.empty() && !identifier.empty()) {
            int i = atoi(label.c_str());
            labels.insert(labels.begin()+i, identifier.c_str());
        }
    }
}