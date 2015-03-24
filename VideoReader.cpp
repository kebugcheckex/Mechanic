#include "VideoReader.h"

using namespace std;
using namespace cv;

VideoReader::VideoReader(string fileName, FaceDetector* faceDetector)
    : m_pfaceDetector(faceDetector)
{
    if (fileName == "0")   // If the input is string "0", we neet to convert it to number 0
        m_video = VideoCapture(0);
    else
        m_video = VideoCapture(fileName);

    if (!m_video.isOpened())
    {
        cerr << "Error: Failed to open the input file " << fileName << endl;
        exit(2);
    }

    Mat frame;
    m_video >> frame;   // Steal one frame for size detection
    m_Width = frame.cols;
    m_Height = frame.rows;
    m_bRunning = false;
}

Size VideoReader::GetFrameSize()
{
    Size s(m_Width, m_Height);
    return s;
}

void VideoReader::Run()
{
    m_bRunning = true;
    m_pThread = new thread(&VideoReader::readFrame, this);
}

void VideoReader::Stop()
{
    if (m_bRunning) m_bRunning = false;
}

void VideoReader::readFrame()
{
    cout << "Entering process thread function" << endl;
    Mat frame;
    while (m_video.read(frame) && m_bRunning)
    {
        m_pfaceDetector->PutFrame(frame);
        cout << "A frame is put into the Face detector's buffer." << endl;
        this_thread::sleep_for(chrono::milliseconds(400));
    }
}
