#ifndef TEXTDETECTOR_H_INCLUDED
#define TEXTDETECTOR_H_INCLUDED

#include "stdafx.h"
#include "VideoReader.h"
#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"
#include "utils.h"

#define NUM_FEATURES 11
#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

class TextDetector
{
public:
    TextDetector(VideoReader* pVideo);
    bool GetResults(TextResult& pResults);
    TextResult ProcessOneFrame(const Mat& inputFrame);
    void Run();
    void Stop();
protected:
    void WorkingThread();
private:
    const int dims[NUM_FEATURES] = {3,3,3,3,3,3,3,3,3,5,5};
    bool running;
    std::thread* m_pThread;
    std::mutex mtx;
    TextResult m_results;
    VideoReader* m_pVideoReader;
    tesseract::TessBaseAPI api;
};
#endif // TEXTDETECTOR_H_INCLUDED
