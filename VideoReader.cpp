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
        if (vc.read(frame)) break;
        mtx.lock();
        frame.copyTo(currentFrame);
        mtx.unlock();
    }
}

Size VideoReader::GetSize()
{
    return m_size;
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
