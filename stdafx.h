#ifndef STDAFX_H_INCLUDED
#define STDAFX_H_INCLUDED

/* std headers */
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

/* opencv headers */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

/* boost headers */

/* TBB headers */
#include <tbb/tbb.h>
#include <tbb/concurrent_queue.h>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

/* alias */
typedef tbb::concurrent_queue<cv::Mat> ConcurrentMatQueue;

#endif // STDAFX_H_INCLUDED
