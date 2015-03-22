#ifndef TEXTLOCALIZER_INCLUDED
#define TEXTLOCALIZER_INCLUDED

#include "stdafx.h"

class TextLocalizer
{
public:
    TextLocalizer(std::queue<cv::Mat>& outputBuffer);
    void PutFrame(cv::Mat frame);
};

#endif // TEXTLOCALIZER_INCLUDED
