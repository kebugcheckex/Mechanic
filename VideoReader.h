#ifndef VIDEOREADER_H_INCLUDED
#define VIDEOREADER_H_INCLUDED

#include "stdafx.h"
#include "FrameProcessor.h"
#include "FaceDetector.h"

class VideoReader
{
public:
    VideoReader(std::string fileName, FaceDetector* faceDetector);
    cv::Size GetFrameSize();
    void Run();
    void Stop();
private:
    void readFrame();
    cv::VideoCapture m_video;
    FaceDetector* m_pfaceDetector;
    int m_Width, m_Height;
    bool m_bRunning;
    std::thread* m_pThread;
};

#endif // VIDEOREADER_H_INCLUDED
