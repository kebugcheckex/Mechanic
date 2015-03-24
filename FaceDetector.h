#ifndef FACEDETECTOR_H_INCLUDED
#define FACEDETECTOR_H_INCLUDED

#include "stdafx.h"

class FaceDetector
{
public:
    static const int BUFFER_SIZE = 25;
    static const std::string CLS_XML;
    FaceDetector();
    bool PutFrame(cv::Mat& frame);
    bool GetFrame(cv::Mat& frame);
    void Run();
    void Stop();
private:
    cv::CascadeClassifier classifier;
    ConcurrentMatQueue m_inputBuffer;
    ConcurrentMatQueue m_outputBuffer;
    int inputBufferPointer;
    int outputBufferPointer;
    bool running;
    void process();
    std::thread* m_pThread;
};

#endif // FACEDETECTOR_H_INCLUDED
