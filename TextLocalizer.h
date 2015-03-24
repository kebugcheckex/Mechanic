#ifndef TEXTLOCALIZER_INCLUDED
#define TEXTLOCALIZER_INCLUDED

#include "stdafx.h"


#define NUM_FEATURES 11
#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

class TextLocalizer
{
public:
    TextLocalizer();
    static const int BUFFER_SIZE = 10;
    bool PutFrame(cv::Mat& frame);
    bool GetFrame(cv::Mat& frame);
    void Run();
    void Stop();
private:
    bool running;
    void process();
    ConcurrentMatQueue m_inputBuffer;
    ConcurrentMatQueue m_outputBuffer;
    int inputBufferPointer;
    int outputBufferPointer;
    std::thread* m_pThread;
};

#endif // TEXTLOCALIZER_INCLUDED
