
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;

// http://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
// http://stackoverflow.com/questions/19189014/how-do-i-find-files-with-a-specific-extension-in-a-directory-that-is-provided-by

//#define DEBUG
#ifdef DEBUG
    #include <cv.h>
#endif

vector<Mat> get_histograms_image(const Mat& image);
double compare_two_images(const string& image1, const string& image2);
bool has_suffix(const string& s, const string& suffix);

int main(int argc, char **argv)
{
    DIR *dir;
    struct dirent *ent;

    string dirname = ".";
    if (argc > 1)
        dirname = argv[1];
    vector<string> image_list;
    if ((dir = opendir(dirname.c_str())) != NULL) {
        string dirname;
        while ((ent = readdir(dir)) != NULL) {
            if (has_suffix(ent->d_name, ".tif") || has_suffix(ent->d_name, ".jpeg") ||
                has_suffix(ent->d_name, ".jpg") ||
                has_suffix(ent->d_name, ".bmp") || has_suffix(ent->d_name, ".png")) {
                image_list.push_back(ent->d_name);
                cout << ent->d_name << endl;
            }
        }
        closedir(dir);
    } else {
        cout << "Failed to open directory " << endl;
        return -1;
    }

    double max_val = -1, new_val;
    int most_similar_indexes[2];
    for (int i = 0; i < image_list.size() - 1; i++) {
        for (int j = i + 1; j < image_list.size(); j++) {
#ifdef DEBUG
            cout << "Image1: " << image_list[i] << ", Image2: " << image_list[j] << flush;
#endif
            new_val = compare_two_images(image_list[i], image_list[j]);
#ifdef DEBUG
            cout << ", Similarity: " << new_val << endl;
#endif
            if (new_val >= max_val) {
                most_similar_indexes[0] = i;
                most_similar_indexes[1] = j;
                max_val = new_val;
            }
        }
    }

    cout << "The most similar images (" << max_val << ") are: " << endl;
    for (int i = 0; i < 2; i++) {
        cout << "* " << image_list[most_similar_indexes[i]] << endl;
        imshow(image_list[most_similar_indexes[i]], imread(image_list[most_similar_indexes[i]], CV_LOAD_IMAGE_COLOR));
    }

    while ((char)waitKey(30) !='q');
}



double compare_two_images(const string& image1, const string& image2) {
    Mat mat_image1 = imread(image1, CV_LOAD_IMAGE_COLOR);
    Mat mat_image2 = imread(image2, CV_LOAD_IMAGE_COLOR);

    vector<Mat> hists_image1 = get_histograms_image(mat_image1);
    vector<Mat> hists_image2 = get_histograms_image(mat_image2);

    double result = 0;
    for (int i = 0; i < hists_image1.size(); i++) {
        result += compareHist(hists_image1[i], hists_image2[i], 0) / hists_image1.size();
    }
    return result;
}

vector<Mat> get_histograms_image(const Mat& image)
{
    Mat original_image = image;
    if (image.channels() == 1)
        cvtColor(original_image, original_image, CV_BGR2GRAY);

    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    vector<Mat> bgr_planes;
    Mat dst, b_hist, g_hist, r_hist, gray_hist;
    int histSize = 256;
    // Draw the histograms for B, G and R
    int hist_h = 480;

    split(image, bgr_planes );
    cvtColor(image, dst, CV_BGR2GRAY);

    bool uniform = true, accumulate = false;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &dst, 1, 0, Mat(), gray_hist, 1, &histSize, &histRange, uniform, accumulate );

    normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat() );
    normalize(gray_hist, gray_hist, 0, hist_h, NORM_MINMAX, -1, Mat() );

#ifdef DEBUG
    int hist_w = 640;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage2( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage3( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImage4( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(gray_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(gray_hist.at<float>(i)) ),
              Scalar( 255, 255, 255), 2, 8, 0  );
        line( histImage2, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage3, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage4, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    hconcat(histImage, histImage2, histImage);
    hconcat(histImage3, histImage4, histImage3);
    vconcat(histImage, histImage3, histImage);

    imshow("Histograms", histImage );

    waitKey(0);
#endif

    vector<Mat> histImages;
    histImages.push_back(b_hist);
    histImages.push_back(g_hist);
    histImages.push_back(r_hist);
    histImages.push_back(gray_hist);

    return histImages;
}

bool has_suffix(const string& s, const string& suffix)  {
    return (s.size() >= suffix.size()) && equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}