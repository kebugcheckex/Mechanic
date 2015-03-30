#include "FaceDetector.h"

using namespace std;
using namespace cv;

FaceDetector::FaceDetector(VideoReader* pVideo, string xmlPath)
{
    m_pVideoReader = pVideo;
    m_classifier = CascadeClassifier(xmlPath);
}

void FaceDetector::WorkingThread()
{
    Mat inputFrame;
    FaceResult objects;
    while (running)
    {
        m_pVideoReader->GetFrame(inputFrame);
        Mat grayImage;
        cvtColor(inputFrame, grayImage, COLOR_BGR2GRAY);
        m_classifier.detectMultiScale(grayImage, objects, 1.3, 5);
        cout << "Objects detected:" << objects.size() << "\t";
        mtx.lock();
        m_results = objects;
        mtx.unlock();
        objects.clear();
    }
}

void FaceDetector::Run()
{
    if (!running)
    {
        running = true;
        m_pThread = new thread(&FaceDetector::WorkingThread, this);
    }
}

void FaceDetector::Stop()
{
    if (running)
    {
        running = false;
        m_pThread = nullptr;
    }
}

bool FaceDetector::GetResult(FaceResult& result)
{
    mtx.lock();
    result = FaceResult(m_results);
    mtx.unlock();
    return true;
}
