#include "ImageUtils.h"

using namespace std;
using namespace cv;

float ImageUtils::compute_skew_angle(Mat image)
{
    Size imgsize = image.size();
    bitwise_not(image, image);
    vector<Vec4i> lines;
    HoughLinesP(image, lines, 1, CV_PI/180, 100, imgsize.width/2.f, 20);

}
