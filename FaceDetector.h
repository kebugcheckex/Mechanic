#ifndef FACEDETECTOR_H_INCLUDED
#define FACEDETECTOR_H_INCLUDED

#include "stdafx.h"
#include "VideoReader.h"

class FaceDetector
{
public:
    FaceDetector(VideoReader* pVideo, std::string xmlPath);
    bool GetResult(FaceResult& result);
    void Run();
    void Stop();
protected:
    void WorkingThread();
private:
    VideoReader* m_pVideoReader;
    cv::CascadeClassifier m_classifier;
    bool running;
    std::thread* m_pThread;
    std::mutex mtx;
    FaceResult m_results;
};

#endif // FACEDETECTOR_H_INCLUDED
