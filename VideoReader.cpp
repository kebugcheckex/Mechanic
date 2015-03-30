#include "VideoReader.h"

using namespace std;
using namespace cv;

VideoReader::VideoReader(string fileName)
{
    if (fileName == "0")
    {
        vc = VideoCapture(0);
    }
    else
    {
        vc = VideoCapture(fileName);
    }
    if (!vc.isOpened())
    {
        cerr << "Error: Failed to open the input file!" << endl;
        exit(1);
    }
    Mat temp;
    vc >> temp;
    m_size = Size(temp.cols, temp.rows);
    running = false;
    currentFrame = Scalar(0, 0, 0);
}

bool VideoReader::GetFrame(Mat& frame)
{
    if (currentFrame.rows == 0 || currentFrame.cols == 0)
    {
        cout << "currentFrame is empty!" << endl;
        return false;
    }
    mtx.lock();
    frame = currentFrame.clone();
    mtx.unlock();
    return true;
}

void VideoReader::ReadThread()
{
    Mat frame;
    while (running)
    {
        if (!vc.read(frame)) break;
        resize(frame, frame, Size(320, 240));
        mtx.lock();
        frame.copyTo(currentFrame);
        mtx.unlock();
    }
}

Size VideoReader::GetSize()
{
    return Size(320, 240);
}
void VideoReader::Run()
{
    if (!running)
    {
        running = true;
        pthread = new thread(&VideoReader::ReadThread, this);
    }
}

void VideoReader::Stop()
{
    if (running)
    {
        running = false;
        pthread = nullptr;
    }
}
