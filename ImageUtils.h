#ifndef IMAGEUTILS_H_INCLUDED
#define IMAGEUTILS_H_INCLUDED

#include "stdafx.h"

class ImageUtils
{
public:
    static cv::Mat Deskew(const cv::Mat& inputImage);
private:
    static float compute_skew_angle(cv::Mat image);

};

#endif // IMAGEUTILS_H_INCLUDED
