#include <iostream>
#include <opencv2/opencv.hpp>



#include "stdafx.h"
#include "VideoReader.h"
#include "FaceDetector.h"
#include "TextDetector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)   // We need 2 parameters for input& output file
    {
        cout << "Usage: " << argv[0] << " input-file output-file" << endl;
        cout << "Input can be either a file on the disk or a URL to a video stream." << endl;
        return 1;
    }
    
    
    string xmlPath("./haarcascade_frontalface_default.xml");
    string inputFileName(argv[1]);
    string outputFileName(argv[2]);
    VideoReader videoReader(inputFileName);
    FaceDetector faceDetector(&videoReader, xmlPath);
    TextDetector textDetector(&videoReader);
    
    VideoCapture Vin(inputFileName);
    if(!Vin.isOpened()) return 1;
    
    Mat inputFrame;
    Vin >> inputFrame;
    int width = inputFrame.cols;
    int height = inputFrame.rows;
    
    
    VideoWriter vw(outputFileName, CV_FOURCC('M','P','E','G'), 25, Size(width/2, height/2));
    if (!vw.isOpened())
    {
        cerr << "Error: Failed to open the output file!" << endl;
        return 3;
    }
    
    CvFont font = cvFontQt("Helvetica", 12.0, CV_RGB(0, 255, 0) );
    CvFont prompt_font = cvFontQt("Arial", 12.0, CV_RGB(255, 0, 0));
    
    Mat outputFrame(height/2, width/2, CV_8UC3);
    
    int content = 0;
    int FrameNum = 0;
    while(Vin.read(inputFrame))
    {
        resize(inputFrame, inputFrame, Size(width/2, height/2));
        cout << "Frame " << FrameNum++ << endl;
        content = 0;
        inputFrame.copyTo(outputFrame);
        FaceResult fr = faceDetector.ProcessOneFrame(inputFrame);
        TextResult tr = textDetector.ProcessOneFrame(inputFrame);
        
        if (fr.size() > 0)
        {
            content++;
            for (FaceResult::iterator it = fr.begin(); it != fr.end(); it++)
            {
                rectangle(outputFrame, *it, Scalar(0, 255, 0), 2);
            }
        }
        
        if (tr.size() > 0)
        {
            content++;
            int line = 0;
            for (TextResult::iterator it = tr.begin(); it != tr.end(); it++)
            {
                rectangle(outputFrame, it->box, Scalar(0, 255, 255), 2);
                Point coord = Point(width - 100, 20 + line * 14);
                addText(outputFrame, it->text, coord, font);
                line++;
            }
        }
        
        imshow("Result", outputFrame);
        waitKey(20);
        vw << outputFrame;
    }
    
    return 0;
}

    