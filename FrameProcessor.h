#ifndef FRAMEPROCESSOR_H_INCLUDED
#define FRAMEPROCESSOR_H_INCLUDED

#include "stdafx.h"

class FrameProcessor
{
public:
    FrameProcessor();
    FrameProcessor(ConcurrentMatQueue& outputBuffer);
    void PutFrame(cv::Mat frame);
    cv::Mat GetFrame();
    void Run();
    void Stop();

protected:
    virtual void process();
    bool running;
    tbb::concurrent_queue<cv::Mat> m_inputBuffer;
    tbb::concurrent_queue<cv::Mat> m_outputBuffer;
    std::thread* pThread;

};
#endif // FRAMEPROCESSOR_H_INCLUDED
