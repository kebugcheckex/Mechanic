#ifndef VIDEOREADER_H_INCLUDED
#define VIDEOREADER_H_INCLUDED

#include "stdafx.h"
#include "FrameProcessor.h"
#include "FaceDetector.h"
#include "TextLocalizer.h"

class VideoReader
{
public:
    VideoReader(std::string fileName, ConcurrentMatQueue* inputBuffer, FaceDetector* pFace, TextLocalizer* pText);
    cv::Size GetFrameSize();
    void Run();
    void Stop();
private:
    void readFrame();
    cv::VideoCapture m_video;
    ConcurrentMatQueue* m_pInputBuffer;
    FaceDetector* m_pfaceDetector;
    TextLocalizer* m_pTextLocalizer;
    int m_Width, m_Height;
    bool m_bRunning;
    std::thread* m_pThread;
};

#endif // VIDEOREADER_H_INCLUDED
