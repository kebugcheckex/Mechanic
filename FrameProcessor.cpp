#include "FrameProcessor.h"

using namespace std;
using namespace cv;
using namespace tbb;

FrameProcessor::FrameProcessor(concurrent_queue<Mat>* outputBuffer)
{
    m_outputBuffer = outputBuffer;
    running = false;
}

void FrameProcessor::PutFrame(Mat frame)
{
    m_inputBuffer.push(frame);
}

void FrameProcessor::Run()
{
    running = true;
    theThread = thread(&FrameProcessor::process, this);
}

void FrameProcessor::Stop()
{
    if (running) running = false;
}
