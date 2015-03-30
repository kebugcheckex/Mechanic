#pragma once

#include "stdafx.h"
#include "region.h"
void accumulate_evidence(std::vector<int> *meaningful_cluster, int grow, cv::Mat *co_occurrence);

void get_gradient_magnitude(cv::Mat& _grey_img, cv::Mat& _gradient_magnitude);

static uchar bcolors[][3] =
{
    {0,0,255},
    {0,128,255},
    {0,255,255},
    {0,255,0},
    {255,128,0},
    {255,255,0},
    {255,0,0},
    {255,0,255},
    {255,255,255}
};

void drawClusters(cv::Mat& img, std::vector<Region> *regions, std::vector<std::vector<int> > *meaningful_clusters);
