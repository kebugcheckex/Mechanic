#include "FaceDetector.h"

using namespace std;
using namespace cv;

const string FaceDetector::CLS_XML = "/home/xinyu/Videos/haarcascade_frontalface_default.xml";

FaceDetector::FaceDetector()
{
    classifier = CascadeClassifier(CLS_XML);
    inputBufferPointer = 0;
    outputBufferPointer = 0;
    running = false;
}

bool FaceDetector::PutFrame(Mat& frame)
{
    if (inputBufferPointer > BUFFER_SIZE)
    {
        cout << "Warning: Face detector input buffer overflow. Frame is dropped." << endl;
        return false;
    }
    else
    {
        m_inputBuffer.push(frame);
        inputBufferPointer++;
        cout << "Face: input buffer pointer = " << m_inputBuffer.unsafe_size() << endl;
    }
    return true;
}

bool FaceDetector::GetFrame(Mat& frame)
{
    if (!m_outputBuffer.try_pop(frame))
    {
        cout << "Warning: internal error!" << endl;
        return false;
    }
    else
    {
        outputBufferPointer--;
    }
    return true;
}

void FaceDetector::Run()
{
    running = true;
    m_pThread = new thread(&FaceDetector::process, this);
}

void FaceDetector::Stop()
{
    running = false;
}

void FaceDetector::process()
{
    cout << "Debug: Face detection thread is running." << endl;
    Mat inputFrame, grayImage, outputFrame;
    vector<Rect> objects;
    while (running)
    {
        cout << "Process: input buffer size = " << m_inputBuffer.unsafe_size() << endl;
        if (m_inputBuffer.try_pop(inputFrame))
        {
            outputFrame = inputFrame;
            cvtColor(inputFrame, grayImage, COLOR_BGR2GRAY);
            classifier.detectMultiScale(grayImage, objects, 1.3, 5);
            cout << "Objects detected:" << objects.size() << endl;
            for(int i = 0; i < objects.size(); i++)
                rectangle(outputFrame, objects[i], Scalar(0,255,0), 2);
            // TODO check if the queue is full
            m_outputBuffer.push(outputFrame);
        }
        else
        {
            cout << "Warning: Face detector input buffer under flow!" << endl;
        }
        objects.clear();
        this_thread::sleep_for(chrono::milliseconds(500));
    }
}
