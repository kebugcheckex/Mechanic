#include "ImageUtils.h"
using namespace std;
using namespace cv;
Mat ImageUtils::Deskew(Mat inputImage)
{
    float angle = compute_skew_angle(inputImage);
    std::cout << "Angle = " << angle << std::endl;
    vector<Point> points;
    Mat_<uchar>::iterator it = inputImage.begin<uchar>();
    Mat_<uchar>::iterator end = inputImage.end<uchar>();
    for (; it != end; ++it)
        if (*it) points.push_back(it.pos());
    RotatedRect box = minAreaRect(Mat(points));
    Mat rotmat = getRotationMatrix2D(box.center, angle, 1);
    Mat rotated;
    warpAffine(inputImage, rotated, rotmat, inputImage.size(), INTER_CUBIC);
    Size box_size = box.size;
    if (box.angle < -45.)
        swap(box_size.width, box_size.height);
    Mat cropped;
    getRectSubPix(rotated, box_size, box.center, cropped);
    return cropped;
}

float ImageUtils::compute_skew_angle(Mat image)
{
    Size imgsize = image.size();
    vector<Vec4i> lines;
    HoughLinesP(image, lines, 1, CV_PI/180, 100, imgsize.width/2.f, 20);
    double angle = 0.0;
    unsigned nb_lines = lines.size();
    for (unsigned i = 0; i < nb_lines; ++i)
    {
        angle += atan2((double)lines[i][3] - lines[i][1],
                       (double)lines[i][2] - lines[i][0]);
    }

    angle /= nb_lines;
    angle = angle / M_PI * 180;
    return angle;
}
