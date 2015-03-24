#include "VideoReader.h"

using namespace std;
using namespace cv;

VideoReader::VideoReader(std::string fileName, ConcurrentMatQueue* inputBuffer, FaceDetector* pFace, TextLocalizer* pText)
    : m_pInputBuffer(inputBuffer), m_pfaceDetector(pFace), m_pTextLocalizer(pText)
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

    m_Width = 320;
    m_Height = 240;
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
        cv::resize(frame, frame, Size(320, 240));
        m_pInputBuffer->push(frame);
        m_pfaceDetector->PutFrame(frame);
        m_pTextLocalizer->PutFrame(frame);
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}
