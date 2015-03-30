#include "FaceDetector.h"

using namespace std;
using namespace cv;

FaceDetector::FaceDetector(VideoReader* pVideo, string xmlPath)
{
    running = false;
    m_pVideoReader = pVideo;
    m_classifier = CascadeClassifier("/home/xinyu/projects/Mechanic/haarcascade_frontalface_default.xml");
}

void FaceDetector::WorkingThread()
{
    cout << "Debug: Entering FaceDetector::WorkingThread" << endl;
    Mat inputFrame;
    FaceResult objects;
    while (running)
    {
        m_pVideoReader->GetFrame(inputFrame);
        //cout << "Debug: inputFrame size = " << inputFrame.cols << "x" << inputFrame.rows << endl;
        Mat grayImage;
        cvtColor(inputFrame, grayImage, COLOR_BGR2GRAY);
        m_classifier.detectMultiScale(grayImage, objects, 1.3, 5);
        //cout << "Objects detected:" << objects.size() << "\t";
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
    //cout << "Entering FaceDetector::GetResult" << endl;
    mtx.lock();
    result = FaceResult(m_results);
    //cout << "Got Face results" << m_results.size() << endl;
    mtx.unlock();
    return true;
}
