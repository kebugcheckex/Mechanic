#ifndef FACEDETECTOR_H_INCLUDED
#define FACEDETECTOR_H_INCLUDED


#include "stdafx.h"
namespace lf = boost::lockfree;

class FaceDetector
{
public:
    FaceDetector(lf::queue<cv::Mat>& outputBuffer);
    const int BUFFER_SIZE = 25;
    const std::string CLS_XML = "/home/xinyu/Videos/haarcascade_frontalface_default.xml";
    void Run();
    void Stop();
    void PutFrame(cv::Mat inputFrame);
private:
    lf::queue<cv::Mat> m_inputBuffer;
    lf::queue<cv::Mat> m_outputBuffer;
    void detect();
    std::thread detectThread;
    bool running;
    CascadeClassifier classifier = CascadeClassifier(CLS_XML);
};

#endif // FACEDETECTOR_H_INCLUDED
