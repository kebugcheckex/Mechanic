#ifndef QRSCANNER_H_INCLUDED
#define QRSCANNER_H_INCLUDED

#include "stdafx.h"
#include "VideoReader.h"
class QRScanner
{
public:
    static const int CV_QR_NORTH = 0;
    static const int CV_QR_EAST = 1;
    static const int CV_QR_SOUTH = 2;
    static const int CV_QR_WEST = 3;
    QRScanner(VideoReader *video);
    bool GetResult(QRResult& result);
    void Run();
    void Stop();
    protected:
    void WorkingThread();
private:
    VideoReader *videoReader;
    bool running;
    std::mutex mtx;
    std::thread *pthread;
    QRResult result;
    cv::Mat qr,qr_raw,qr_gray,qr_thres;
    float cv_distance(cv::Point2f P, cv::Point2f Q);					// Get Distance between two points
    float cv_lineEquation(cv::Point2f L, cv::Point2f M, cv::Point2f J);		// Perpendicular Distance of a Point J from line formed by Points L and M; Solution to equation of the line Val = ax+by+c
    float cv_lineSlope(cv::Point2f L, cv::Point2f M, int& alignement);	// Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
    void cv_getVertices(std::vector<std::vector<cv::Point> > contours, int c_id,float slope, std::vector<cv::Point2f>& X);
    void cv_updateCorner(cv::Point2f P, cv::Point2f ref ,float& baseline,  cv::Point2f& corner);
    void cv_updateCornerOr(int orientation, std::vector<cv::Point2f> IN, std::vector<cv::Point2f> &OUT);
    bool getIntersectionPoint(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2, cv::Point2f& intersection);
    float cross(cv::Point2f v1, cv::Point2f v2);
};

#endif // QRSCANNER_H_INCLUDED
