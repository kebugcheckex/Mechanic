#ifndef VIDEOREADER_H_INCLUDED
#define VIDEOREADER_H_INCLUDED

#include "stdafx.h"

class VideoReader
{
public:
    VideoReader(std::string fileName);
    bool GetFrame(cv::Mat& frame);
    void Run();
    void Stop();
    cv::Size GetSize();
private:
    cv::Size m_size;
    bool running;
    cv::Mat currentFrame;
    cv::VideoCapture vc;
    std::thread* pthread;
    std::mutex mtx;
    void ReadThread();
};
#endif // VIDEOREADER_H_INCLUDED
