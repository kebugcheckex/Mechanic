#include "FaceDetector.h"

using namespace std;
using namespace cv;


FaceDetector::FaceDetector(lf::queue<Mat>& outputBuffer)
{
    running = false;
    m_outputBuffer = outputBuffer;
    lf::capacity(BUFFER_SIZE);
    m_inputBuffer = lf::queue(BUFFER_SIZE);
}

void FaceDetector::PutFrame(cv::Mat inputFrame)
{
    if (!m_inputBuffer.push(inputFrame))
    {
        cout << "Warning: Face detector input buffer overflow!" << endl;
    }
}

void FaceDetector::Run()
{
    running = true;
    detecThread = thread(&FaceDetector::detect, this);
}

void FaceDetector::Stop()
{
    if (running)
        running = false;
}

void FaceDetector::detect()
{
    cout << "Debug: Face detection thread is running." << endl;
    Mat inputFrame, greyImage, outputFrame;
    while (running)
    {
        vector<Rect> objects;
        bool detected = false;
        if (inputBuffer.pop(inputFrame))
        {
            outputFrame = inputFrame;
            cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
            classifier.detectMultiScale(grayImage, objects, 1.3, 5);
            cout << "Objects detected:" << objects.size() << "\t";
            for(int i = 0; i < objects.size(); i++)
                rectangle(outputFrame, objects[i], Scalar(0,255,0), 2);
            // TODO check if the queue is full
            if (!m_outputBuffer.push(outputFrame))
            {
                out << "Warning: Output buffer overflow!" << endl;
            }
        }
        else
        {
            cout << "Warning: Face detector input buffer under flow!" << endl;
        }
    }
}
