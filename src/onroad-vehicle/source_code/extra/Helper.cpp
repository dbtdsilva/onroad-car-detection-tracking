#include "Helper.h"

using namespace std;

double Helper::overlapPercentage(const cv::Rect &A, const cv::Rect &B) {
    int x_overlap = max(0, min(A.x+A.width,B.x + B.width) - max(A.x,B.x));
    int y_overlap = max(0, min(A.y+A.height,B.y + B.height) - max(A.y,B.y));
    int overlapArea = x_overlap * y_overlap;
    return (double)overlapArea/(A.width*A.height);
}
