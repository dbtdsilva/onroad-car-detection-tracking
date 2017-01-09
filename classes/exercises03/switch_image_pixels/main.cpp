
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    string filename = "lena_color_512.tif";
    if (argc > 1)
        filename = string(argv[1]);
    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);
    int height = image.rows;
    int width = image.cols;

    Mat new_image(height, width, CV_8UC3);
    vector<Vec3b> pixels;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            pixels.push_back(image.at<Vec3b>(row, col));
        }
    }
    std::random_shuffle(pixels.begin(), pixels.end());
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            new_image.at<Vec3b>(row, col) = pixels.at(row * width + col);
        }
    }

    imshow("Image", image);
    imshow("New image", new_image);

    stringstream ss;
    ss << "shuffled_" << filename;
    imwrite(ss.str(), new_image);
    while ((char)waitKey(30) !='q');
}
