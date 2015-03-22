#ifndef FACEDETECTOR_H_INCLUDED
#define FACEDETECTOR_H_INCLUDED

#include "stdafx.h"
#include "FrameProcessor.h"

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>

class FaceDetector : public FrameProcessor
{
public:
    const int BUFFER_SIZE = 25;
    const std::string CLS_XML = "/home/xinyu/Videos/haarcascade_frontalface_default.xml";
private:
    void process();
    cv::CascadeClassifier classifier;
};

#endif // FACEDETECTOR_H_INCLUDED
