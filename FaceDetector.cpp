#include "FaceDetector.h"

using namespace std;
using namespace cv;


//FaceDetector::FaceDetector(concurrent_queue<Mat>* outputBuffer)
//{
//    classifier = CascadeClassifier(CLS_XML);
//}

void FaceDetector::process()
{
    cout << "Debug: Face detection thread is running." << endl;
    Mat inputFrame, grayImage, outputFrame;
    while (running)
    {
        vector<Rect> objects;
        bool detected = false;
        if (m_inputBuffer.try_pop(inputFrame))
        {
            outputFrame = inputFrame;
            cvtColor(inputFrame, grayImage, COLOR_BGR2GRAY);
            classifier.detectMultiScale(grayImage, objects, 1.3, 5);
            cout << "Objects detected:" << objects.size() << "\t";
            for(int i = 0; i < objects.size(); i++)
                rectangle(outputFrame, objects[i], Scalar(0,255,0), 2);
            // TODO check if the queue is full
            m_outputBuffer->push(outputFrame);
        }
        else
        {
            cout << "Warning: Face detector input buffer under flow!" << endl;
        }
    }
}
