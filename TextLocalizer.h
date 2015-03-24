#ifndef TEXTLOCALIZER_INCLUDED
#define TEXTLOCALIZER_INCLUDED

#include "stdafx.h"
#include "FrameProcessor.h"
#include "region.h"
#include "mser.h"
#include "max_meaningful_clustering.h"
#include "region_classifier.h"
#include "group_classifier.h"
#include "utils.h"

#define NUM_FEATURES 11
#define DECISION_THRESHOLD_EA 0.5
#define DECISION_THRESHOLD_SF 0.999999999

class TextLocalizer : public FrameProcessor
{
    void process();
};

#endif // TEXTLOCALIZER_INCLUDED
