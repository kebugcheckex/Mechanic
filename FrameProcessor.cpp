#include "FrameProcessor.h"

using namespace std;
using namespace cv;
using namespace tbb;

FrameProcessor::FrameProcessor()
{
}

FrameProcessor::FrameProcessor(ConcurrentMatQueue& outputBuffer)
    : m_outputBuffer(outputBuffer)
{
    running = false;
}

void FrameProcessor::PutFrame(Mat frame)
{
    m_inputBuffer.push(frame);
    cout << "++++++++++++++++Face detector input frame size is " << m_inputBuffer.unsafe_size() << endl;
}

Mat FrameProcessor::GetFrame()
{
    Mat output;
    if (!m_outputBuffer.try_pop(output))
        cout << "Failed to pop" << endl;

    return output;
}

void FrameProcessor::Run()
{
    running = true;
    pThread = new thread(&FrameProcessor::process, this);
}

void FrameProcessor::Stop()
{
    if (running) running = false;
}

void FrameProcessor::process()
{
}
